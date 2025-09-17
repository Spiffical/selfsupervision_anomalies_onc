#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, List

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
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from src.finwhale_mat_dataset import FinWhaleMatDataset
from scripts.train_cnn import SmallCNN  # reuse the model class


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


def save_png(x: torch.Tensor, out_path: Path, overlay_text: str = "") -> None:
    # x is [1, F, T] in [0,1]
    arr = (x.squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    # Convert to (H, W) grayscale image; map F to H
    img = Image.fromarray(arr)
    img = img.convert('L')
    # Add overlay text
    if overlay_text:
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((5, 5), overlay_text, fill=(255, 0, 0), font=font)
    img.save(str(out_path))


def main():
    ap = argparse.ArgumentParser(description="Test CNN on Fin Whale MAT spectrograms")
    ap.add_argument('--pos-dir', type=str, required=True, help='Directory with positive MAT files')
    ap.add_argument('--neg-dir', type=str, required=True, help='Directory with negative MAT files')
    ap.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint (.pt)')
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
    args = ap.parse_args()

    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))

    out_dir = Path(args.out_dir)
    (out_dir / 'pngs' / 'tp').mkdir(parents=True, exist_ok=True)
    (out_dir / 'pngs' / 'tn').mkdir(parents=True, exist_ok=True)
    (out_dir / 'pngs' / 'fp').mkdir(parents=True, exist_ok=True)
    (out_dir / 'pngs' / 'fn').mkdir(parents=True, exist_ok=True)

    # Prefer seed from checkpoint's args.pkl if available (unless ignored)
    seed_to_use = args.seed
    try:
        if not args.ignore_checkpoint_seed:
            ckpt_path = Path(args.checkpoint)
            ckpt_dir = ckpt_path.parent
            sidecar_args = ckpt_dir / 'args.pkl'
            if sidecar_args.exists():
                import pickle
                with open(sidecar_args, 'rb') as f:
                    saved_args = pickle.load(f)
                if hasattr(saved_args, 'seed'):
                    seed_to_use = int(getattr(saved_args, 'seed'))
                elif isinstance(saved_args, dict) and 'seed' in saved_args:
                    seed_to_use = int(saved_args['seed'])
                print(f"Using seed from checkpoint args.pkl: {seed_to_use}")
            else:
                print("No args.pkl next to checkpoint; using CLI seed")
    except Exception as e:
        print(f"Warning: failed to load seed from checkpoint args.pkl: {e}. Using CLI seed {seed_to_use}")

    # Build test dataset (optionally jittered)
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

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SmallCNN().to(device)
    state_dict = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_paths: List[str] = []
    all_meta: List[dict] = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                x, y, paths, meta = batch
            elif len(batch) == 3:
                x, y, paths = batch
                meta = [None] * x.size(0)
            else:
                x, y = batch
                paths = ["?"] * x.size(0)
                meta = [None] * x.size(0)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
            all_paths.extend(list(paths))
            all_meta.extend(list(meta))

            # Save PNGs for this batch
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            for i in range(x.size(0)):
                truth = int(y[i].item())
                pred = int(preds[i].item())
                cls = 'tp' if (pred == 1 and truth == 1) else \
                      'tn' if (pred == 0 and truth == 0) else \
                      'fp' if (pred == 1 and truth == 0) else 'fn'
                # Construct overlay text with source filename
                src = Path(all_paths[-x.size(0) + i]).name
                overlay = f"pred={pred} truth={truth} file={src}"
                out_path = out_dir / 'pngs' / cls / f"{Path(src).stem}.png"
                # Save spectrogram image
                save_png(x[i].detach().cpu(), out_path, overlay_text=overlay)

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(labels_cat, logits_cat)

    # Save metrics report
    report_txt = out_dir / 'report.txt'
    with open(report_txt, 'w') as f:
        f.write(f"seed_used: {seed_to_use}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    # Also CSV
    report_csv = out_dir / 'report.csv'
    import csv
    with open(report_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(metrics.keys()))
        writer.writerow([metrics[k] for k in metrics.keys()])

    print(f"Saved report to {report_txt} and {report_csv}")

    # Threshold sweep plots
    probs_pos = torch.softmax(logits_cat, dim=1)[:, 1].numpy()
    y_true = labels_cat.numpy().astype(np.int32)

    # Precision-Recall vs threshold
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, probs_pos)
    plt.figure()
    plt.plot(recalls, precisions, label='PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'pr_curve.png', dpi=150)
    plt.close()

    # ROC curve and AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, probs_pos)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_curve.png', dpi=150)
    plt.close()

    # Precision and Recall as function of threshold
    # Convert PR arrays (which may not include threshold for last point)
    thresholds = np.linspace(0.0, 1.0, 101)
    prec_at = []
    rec_at = []
    for thr in thresholds:
        pred = (probs_pos >= thr).astype(np.int32)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec_at.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        rec_at.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    plt.figure()
    plt.plot(thresholds, prec_at, label='Precision')
    plt.plot(thresholds, rec_at, label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision/Recall vs Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'precision_recall_vs_threshold.png', dpi=150)
    plt.close()

    # Performance vs distance from center (only for positives where meta present)
    dist_fracs = []
    preds_bin = (probs_pos >= 0.5).astype(np.int32)
    correct = (preds_bin == y_true).astype(np.int32)
    for m in all_meta:
        if isinstance(m, dict) and 'dist_from_center_frac' in m:
            dist_fracs.append(float(m['dist_from_center_frac']))
        else:
            dist_fracs.append(np.nan)
    dist_fracs = np.array(dist_fracs)
    # Bin distances
    bins = np.linspace(0, 1.0, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    acc_by_bin = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (dist_fracs >= b0) & (dist_fracs < b1)
        if mask.sum() > 0:
            acc_by_bin.append(float(correct[mask].mean()))
        else:
            acc_by_bin.append(np.nan)
    plt.figure()
    plt.plot(bin_centers, acc_by_bin, marker='o')
    plt.xlabel('Distance from Center (fraction of half-length)')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs Call Offset from Center')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'accuracy_vs_center_offset.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    main()
