#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from src.finwhale_mat_dataset import FinWhaleMatDataset
from src.models.fin_models import create_model


def compute_metrics(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> dict:
    with torch.no_grad():
        probs = torch.softmax(y_pred_logits, dim=1)
        y_pred = torch.argmax(probs, dim=1)
        correct = (y_pred == y_true).sum().item()
        total = y_true.numel()
        acc = correct / max(total, 1)
        # precision, recall, f1 for positive class (label=1)
        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        tn = ((y_pred == 0) & (y_true == 0)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return dict(acc=acc, precision=prec, recall=rec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn, total=total)


def apply_contrast(x01: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    lo = np.percentile(x01, pmin)
    hi = np.percentile(x01, pmax)
    if hi <= lo:
        return x01
    y = (x01 - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def to_colormap_rgb(x01: np.ndarray, cmap_name: str = 'inferno') -> np.ndarray:
    cmap = cm.get_cmap(cmap_name)
    rgb = (cmap(x01)[..., :3] * 255.0).astype(np.uint8)
    return rgb


def save_png(x: torch.Tensor, out_path: Path, overlay_text: str = "", scale: int = 3,
             cmap: str = 'inferno', pmin: float = 2.0, pmax: float = 98.0, marker_x: int = None) -> None:
    # x is [1, F, T] in [0,1]
    arr01 = x.squeeze(0).detach().cpu().numpy().astype(np.float32)
    arr01 = apply_contrast(arr01, pmin, pmax)
    rgb = to_colormap_rgb(arr01, cmap_name=cmap)
    img = Image.fromarray(rgb)
    # draw marker before scaling
    if marker_x is not None and 0 <= marker_x < arr01.shape[1]:
        draw = ImageDraw.Draw(img)
        h = img.size[1]
        draw.line([(marker_x, 0), (marker_x, h)], fill=(255, 255, 255), width=1)
    if scale > 1:
        w, h = img.size
        img = img.resize((w * scale, h * scale), resample=Image.BICUBIC)
    if overlay_text:
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((5, 5), overlay_text, fill=(255, 255, 255), font=font)
    img.save(str(out_path))


def main():
    ap = argparse.ArgumentParser(description="Test CNN on Fin Whale MAT spectrograms (supports multiple models)")
    ap.add_argument('--pos-dir', type=str, required=True, help='Directory with positive MAT files')
    ap.add_argument('--neg-dir', type=str, required=True, help='Directory with negative MAT files')
    # Backward-compat single checkpoint; can be omitted when using --checkpoints
    ap.add_argument('--checkpoint', type=str, default="", help='Path to trained model checkpoint (.pt)')
    # New: multiple checkpoints
    ap.add_argument('--checkpoints', type=str, nargs='*', help='Paths to multiple checkpoints')
    ap.add_argument('--labels', type=str, nargs='*', help='Optional labels for checkpoints (same order)')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--crop-size', type=int, default=96)
    ap.add_argument('--min-db', type=float, default=-80.0)
    ap.add_argument('--max-db', type=float, default=0.0)
    ap.add_argument('--train-ratio', type=float, default=0.8)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--augment-test', action='store_true', help='Jitter test crops like training')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out-dir', type=str, required=True, help='Output directory for this test run')
    ap.add_argument('--ignore-checkpoint-seed', action='store_true', help='Do not load seed from args.pkl next to checkpoint')
    ap.add_argument('--png-scale', type=int, default=3, help='Scale factor for saved spectrogram PNGs')
    ap.add_argument('--png-cmap', type=str, default='inferno', help='Colormap for saved PNGs')
    ap.add_argument('--png-pmin', type=float, default=2.0, help='Lower percentile for PNG contrast')
    ap.add_argument('--png-pmax', type=float, default=98.0, help='Upper percentile for PNG contrast')
    args = ap.parse_args()

    # Build checkpoint list (support comma-separated and backward compat)
    ckpts: List[str] = []
    if args.checkpoints:
        for c in args.checkpoints:
            ckpts.extend([s for s in c.split(',') if s.strip()])
    if args.checkpoint:
        ckpts.append(args.checkpoint)
    if not ckpts:
        raise SystemExit("Please provide at least one checkpoint via --checkpoint or --checkpoints")
    labels: List[str] = []
    if args.labels:
        for l in args.labels:
            labels.extend([s for s in l.split(',') if s.strip()])

    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build test dataset once
    # Derive seed from the first checkpoint if not ignored
    seed_to_use = args.seed
    if not args.ignore_checkpoint_seed:
        try:
            import pickle
            sidecar = Path(ckpts[0]).parent / 'args.pkl'
            if sidecar.exists():
                with open(sidecar, 'rb') as f:
                    sargs = pickle.load(f)
                if hasattr(sargs, 'seed'):
                    seed_to_use = int(getattr(sargs, 'seed'))
                elif isinstance(sargs, dict) and 'seed' in sargs:
                    seed_to_use = int(sargs['seed'])
        except Exception:
            pass

    test_ds = FinWhaleMatDataset(
        args.pos_dir, args.neg_dir,
        split='test', train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        crop_size=args.crop_size, min_db=args.min_db, max_db=args.max_db,
        seed=seed_to_use, augment_eval=bool(args.augment_test), return_path=True, return_meta=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Helper to derive a short label per checkpoint
    def derive_label(ckpt_path: str, model_name: str) -> str:
        p = Path(ckpt_path)
        # prefer directory name (e.g., resnet18) or model name
        parent = p.parent.name
        base = p.stem
        for cand in [parent, model_name, base]:
            if cand:
                return cand
        return 'model'

    # Store results per model
    all_results: Dict[str, Dict[str, np.ndarray]] = {}

    for idx, ckpt_path in enumerate(ckpts):
        # Determine model name from args.pkl (preferred) or checkpoint
        model_name = 'SmallCNN'
        try:
            import pickle
            sidecar_args = Path(ckpt_path).parent / 'args.pkl'
            if sidecar_args.exists():
                with open(sidecar_args, 'rb') as f:
                    saved_args = pickle.load(f)
                if hasattr(saved_args, 'model'):
                    model_name = str(getattr(saved_args, 'model'))
                elif isinstance(saved_args, dict) and 'model' in saved_args:
                    model_name = str(saved_args['model'])
        except Exception:
            pass
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'args' in checkpoint and isinstance(checkpoint['args'], dict):
            model_name = checkpoint['args'].get('model', model_name)
        model = create_model(model_name, num_classes=2, in_ch=1).to(device)
        state_dict = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        # Per-model output directory
        label = labels[idx] if (len(labels) > idx) else derive_label(ckpt_path, model_name)
        model_dir = out_dir / label
        (model_dir / 'pngs' / 'tp').mkdir(parents=True, exist_ok=True)
        (model_dir / 'pngs' / 'tn').mkdir(parents=True, exist_ok=True)
        (model_dir / 'pngs' / 'fp').mkdir(parents=True, exist_ok=True)
        (model_dir / 'pngs' / 'fn').mkdir(parents=True, exist_ok=True)

        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_paths: List[str] = []
        all_meta: List[dict] = []

        def normalize_meta(meta_obj, batch_size: int):
            if isinstance(meta_obj, dict):
                out = []
                for i in range(batch_size):
                    item = {}
                    for k, v in meta_obj.items():
                        try:
                            item[k] = v[i].item() if hasattr(v, 'shape') else v[i]
                        except Exception:
                            item[k] = None
                    out.append(item)
                return out
            elif isinstance(meta_obj, (list, tuple)):
                return list(meta_obj)
            else:
                return [None] * batch_size

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 4:
                    x, y, paths, meta = batch
                    meta_list = normalize_meta(meta, x.size(0))
                elif len(batch) == 3:
                    x, y, paths = batch
                    meta_list = [None] * x.size(0)
                else:
                    x, y = batch
                    paths = ["?"] * x.size(0)
                    meta_list = [None] * x.size(0)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
                all_paths.extend(list(paths))
                all_meta.extend(meta_list)

                # Save PNGs for this batch
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                for i in range(x.size(0)):
                    truth = int(y[i].item())
                    pred = int(preds[i].item())
                    cls = 'tp' if (pred == 1 and truth == 1) else \
                          'tn' if (pred == 0 and truth == 0) else \
                          'fp' if (pred == 1 and truth == 0) else 'fn'
                    src = Path(all_paths[-x.size(0) + i]).name
                    overlay = f"{label} pred={pred} truth={truth} file={src}"
                    out_path = model_dir / 'pngs' / cls / f"{Path(src).stem}.png"
                    marker_x = None
                    m = meta_list[i]
                    try:
                        if isinstance(m, dict) and 'crop_start' in m and 'full_T' in m and 'crop_size' in m:
                            marker_x = int((int(m['full_T']) // 2) - int(m['crop_start']))
                            marker_x = max(0, min(marker_x, int(m['crop_size']) - 1))
                    except Exception:
                        marker_x = None
                    save_png(x[i].detach().cpu(), out_path, overlay_text=overlay, scale=int(args.png_scale),
                             cmap=args.png_cmap, pmin=args.png_pmin, pmax=args.png_pmax, marker_x=marker_x)

        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(labels_cat, logits_cat)
        # Save per-model report
        with open(model_dir / 'report.txt', 'w') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        # Collect for combined plots
        probs_pos = torch.softmax(logits_cat, dim=1)[:, 1].numpy()
        y_true = labels_cat.numpy().astype(np.int32)
        all_results[label] = {
            'probs': probs_pos,
            'y_true': y_true,
            'meta': np.array([m.get('dist_from_center_frac', np.nan) if isinstance(m, dict) else np.nan for m in all_meta], dtype=float)
        }

        # Also save individual PR/ROC per model
        fpr, tpr, _ = roc_curve(y_true, probs_pos)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 5))
        plt.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.3f})')
        plt.plot([0,1],[0,1],'k--',alpha=0.5)
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC: {label}'); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(model_dir / 'roc_curve.png', dpi=150); plt.close()

        prec, rec, _ = precision_recall_curve(y_true, probs_pos)
        plt.figure(figsize=(8,5))
        plt.plot(rec, prec, label=label)
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR: {label}'); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(model_dir / 'pr_curve.png', dpi=150); plt.close()

    # Combined plots across models
    # Use y_true from first model as reference (should be identical)
    # ROC
    plt.figure(figsize=(8, 5))
    for label, res in all_results.items():
        y_true = res['y_true']; probs = res['probs']
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC (combined)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_dir / 'roc_curve_all.png', dpi=150); plt.close()

    # PR combined
    plt.figure(figsize=(8,5))
    for label, res in all_results.items():
        y_true = res['y_true']; probs = res['probs']
        prec, rec, _ = precision_recall_curve(y_true, probs)
        plt.plot(rec, prec, label=label)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve (combined)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_dir / 'pr_curve_all.png', dpi=150); plt.close()

    # Precision/Recall vs threshold combined
    plt.figure(figsize=(8,5))
    thresholds = np.linspace(0.0, 1.0, 101)
    for label, res in all_results.items():
        y_true = res['y_true']; probs = res['probs']
        prec_at=[]; rec_at=[]
        for thr in thresholds:
            pred = (probs >= thr).astype(np.int32)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            prec_at.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            rec_at.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        plt.plot(thresholds, prec_at, label=f'{label} Precision')
        plt.plot(thresholds, rec_at, linestyle='--', label=f'{label} Recall')
    plt.xlabel('Threshold'); plt.ylabel('Score'); plt.title('Precision/Recall vs Threshold (combined)')
    plt.legend(ncol=2, fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_dir / 'precision_recall_vs_threshold_all.png', dpi=150); plt.close()

    # Accuracy vs offset combined
    plt.figure(figsize=(8,5))
    bins = np.linspace(0, 1.0, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for label, res in all_results.items():
        dist_fracs = res['meta']
        y_true = res['y_true']; probs = res['probs']
        preds_bin = (probs >= 0.5).astype(np.int32)
        correct = (preds_bin == y_true).astype(np.float32)
        acc_by_bin = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (dist_fracs >= b0) & (dist_fracs < b1) & (~np.isnan(dist_fracs))
            if mask.sum() > 0:
                acc_by_bin.append(float(correct[mask].mean()))
            else:
                acc_by_bin.append(np.nan)
        plt.plot(bin_centers, acc_by_bin, marker='o', label=label)
    plt.xlabel('Distance from Center (fraction of half-length)'); plt.ylabel('Accuracy'); plt.title('Accuracy vs Call Offset (combined)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_dir / 'accuracy_vs_center_offset_all.png', dpi=150); plt.close()


if __name__ == '__main__':
    main()
