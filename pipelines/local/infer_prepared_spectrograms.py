#!/usr/bin/env python3
"""
Local GPU-side inference over prepared .npy spectrograms.

Inputs:
 - manifest CSV produced by the VM pipeline (npy_path, hydrophone, t0, t1, original_mat, ...)
 - trained model checkpoint + args.pkl

Process:
 - Load model
 - For each .npy: load [F,T], convert to tensor [1,F,T], run forward with optional AMP
 - Save predictions CSV (top-1 and per-class probabilities)
 - Optional: write dashboard-friendly structure under --dashboard-root:
    dashboards_root/YYYY-MM-DD/HYDROPHONE/
      - images/*.png (rendered spectrogram)
      - labels.json (filename -> [predicted labels])
"""
import argparse
import os
import sys
import csv
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.ssamba.utilities.training_utils import create_model


def load_model(model_dir: Path, checkpoint_path: Path, device: torch.device):
    import pickle
    with (model_dir / 'args.pkl').open('rb') as f:
        args = pickle.load(f)
    args.multiclass = True
    args.exp_dir = model_dir
    model = create_model(args).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, args


def main():
    ap = argparse.ArgumentParser(description='Infer prepared .npy spectrograms and write predictions CSV')
    ap.add_argument('--manifest', type=str, required=True, help='Path to manifest CSV with npy_path column')
    ap.add_argument('--model-dir', type=str, required=True, help='Directory containing args.pkl and checkpoint')
    ap.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint .pth')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--use-amp', action='store_true')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out-csv', type=str, required=True)
    # Dashboard outputs
    ap.add_argument('--dashboard-root', type=str, default=None, help='If set, write PNGs and labels.json grouped by day/hydro')
    ap.add_argument('--save-png', action='store_true', help='Render PNGs for dashboard display')
    ap.add_argument('--save-png-logfreq', action='store_true', help='Resample frequency axis to log scale before saving PNGs')
    ap.add_argument('--thresholds-file', type=str, default=None, help='JSON file mapping class name -> threshold (else 0.5)')
    args = ap.parse_args()

    device = torch.device(args.device)
    model, margs = load_model(Path(args.model_dir), Path(args.checkpoint), device)

    # Read manifest
    rows: List[dict] = []
    with open(args.manifest, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    paths = [r['npy_path'] for r in rows if 'npy_path' in r]
    if not paths:
        print('No npy_path entries in manifest')
        return

    # Batch inference
    probs_all: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            batch = []
            for p in paths[i:i+args.batch_size]:
                arr = np.load(p).astype(np.float32)  # [F,T]
                ten = torch.from_numpy(arr[None, ...])  # [1,F,T]
                batch.append(ten)
            if not batch:
                continue
            x = torch.stack(batch, dim=0).to(device)  # [B,1,F,T]
            if args.use_amp and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(x, task=getattr(margs, 'task', 'ft_cls'))
            else:
                logits = model(x, task=getattr(margs, 'task', 'ft_cls'))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(p)

    probs = np.concatenate(probs_all, axis=0)
    preds = probs.argmax(axis=1)

    # Recover class names from a small ONC dataset helper
    from src.ssamba.dataset import ONCSpectrogramDataset
    probe_h5 = "/home/sbialek/ONC/selfsupervision_anomalies_onc/data/different_locations_incl_backgroundpipelinenormals_multilabel.h5"
    probe_ds = ONCSpectrogramDataset(
        data_path=probe_h5,
        split='test',
        train_ratio=getattr(margs, 'train_ratio', 0.8),
        val_ratio=getattr(margs, 'val_ratio', 0.1),
        seed=getattr(margs, 'split_seed', 42),
        target_length=getattr(margs, 'target_length', 1024),
        num_mel_bins=getattr(margs, 'num_mel_bins', 128),
        supervised=True,
        subsample_test=False,
        multiclass=True,
        num_classes=getattr(margs, 'num_classes', 8),
    )
    _ = probe_ds[0]
    class_names = [probe_ds.index_to_label[i] for i in range(len(probe_ds.index_to_label))]

    # Write output CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        fieldnames = ['npy_path', 'hydrophone', 't0', 't1', 'pred_index', 'pred_label'] + [f'prob_{c}' for c in class_names]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(rows[:len(preds)]):
            d = {
                'npy_path': row.get('npy_path', ''),
                'hydrophone': row.get('hydrophone', ''),
                't0': row.get('t0', ''),
                't1': row.get('t1', ''),
                'pred_index': int(preds[i]),
                'pred_label': class_names[int(preds[i])] if int(preds[i]) < len(class_names) else str(preds[i]),
            }
            for j, cname in enumerate(class_names):
                d[f'prob_{cname}'] = float(probs[i, j])
            w.writerow(d)
    print(f"Wrote predictions: {out_path}")

    # Optional dashboard outputs
    if args.dashboard_root:
        dash_root = Path(args.dashboard_root)
        # thresholds
        thresholds: Dict[str, float] = {}
        if args.thresholds_file and Path(args.thresholds_file).exists():
            import json as _json
            with open(args.thresholds_file, 'r') as tf:
                thresholds = _json.load(tf)
        # Default global threshold=0.5 if not provided
        def is_on(label: str, val: float) -> bool:
            thr = thresholds.get(label, 0.5)
            try:
                thr_f = float(thr)
            except Exception:
                thr_f = 0.5
            return float(val) >= thr_f

        # Group rows by (day, hydrophone)
        from collections import defaultdict
        groups: Dict[tuple, List[int]] = defaultdict(list)
        for i, row in enumerate(rows[:len(preds)]):
            t0 = row.get('t0', '')
            # Expect formats like 20240628T025500.000Z -> day 2024-06-28
            day = ''
            if len(t0) >= 8:
                y, m, d = t0[:4], t0[4:6], t0[6:8]
                day = f"{y}-{m}-{d}"
            hydro = row.get('hydrophone', 'UNKNOWN')
            groups[(day, hydro)].append(i)

        import json
        import matplotlib.pyplot as plt
        import numpy as _np
        for (day, hydro), idxs in groups.items():
            day_dir = dash_root / day / hydro
            img_dir = day_dir / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)
            labels_json_path = day_dir / 'labels.json'
            labels_map: Dict[str, List[str]] = {}

            for i in idxs:
                row = rows[i]
                npy_path = Path(row.get('npy_path', ''))
                base_name = npy_path.stem + '.png'
                # Thresholded labels
                label_list: List[str] = []
                for j, cname in enumerate(class_names):
                    if is_on(cname, probs[i, j]):
                        label_list.append(cname)
                # Ensure at least the top-1 is present
                if not label_list:
                    label_list = [class_names[int(preds[i])]]
                labels_map[base_name] = label_list

                if args.save_png:
                    try:
                        arr = np.load(npy_path).astype(np.float32)  # [F,T]
                        if args.save_png_logfreq and arr.shape[0] > 0:
                            # Log-frequency resample: remap linear freq bins to log spacing
                            F, T = arr.shape
                            # Avoid zeros; assume lowest nonzero freq maps to index 1
                            eps = 1e-8
                            f_lin = _np.linspace(1, F, F)
                            f_log = _np.geomspace(max(1.0, f_lin[0]), f_lin[-1], F)
                            # Interpolate each time column along freq axis to log grid
                            arr_log = _np.zeros_like(arr)
                            for tcol in range(T):
                                arr_log[:, tcol] = _np.interp(f_log, f_lin, arr[:, tcol])
                            arr_to_save = arr_log
                        else:
                            arr_to_save = arr
                        plt.imsave(img_dir / base_name, arr_to_save, origin='lower', cmap='inferno')
                    except Exception as e:
                        print(f"Warning: failed to save PNG for {npy_path}: {e}")

            with labels_json_path.open('w') as jf:
                json.dump(labels_map, jf, indent=2)
            print(f"Wrote dashboard labels: {labels_json_path}")


if __name__ == '__main__':
    main()


