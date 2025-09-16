#!/usr/bin/env python3
"""
Train a simple CNN baseline on Fin Whale MAT spectrograms.

Uses the FinWhaleMatDataset to read .mat spectrograms directly from disk,
apply center-jitter augmentation for positives, square crop, and normalization.

Supports class imbalance via either:
 - WeightedRandomSampler (balance='weighted' or 'oversample'), or
 - Class-weighted CrossEntropyLoss (balance='none': default).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

# Ensure repo root is on sys.path so `src` is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.finwhale_mat_dataset import make_dataloaders, FinWhaleMatDataset
from src.ssamba.utilities.wandb_utils import (
    init_wandb,
    log_training_metrics,
    log_validation_metrics,
    finish_run,
)


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, 1, 96, 96]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 32, 48, 48]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 64, 24, 24]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [B, 128, 12, 12]
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # [B, 256, 6, 6]
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 256]
        x = self.dropout(x)
        logits = self.head(x)  # [B, 2]
        return logits


def compute_metrics(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> dict:
    with torch.no_grad():
        probs = torch.softmax(y_pred_logits, dim=1)
        y_pred = torch.argmax(probs, dim=1)
        correct = (y_pred == y_true).sum().item()
        total = y_true.numel()
        acc = correct / max(total, 1)

        # precision, recall, f1 for positive class (label=1)
        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return dict(acc=acc, precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn, total=total)


def train_one_epoch(model, loader: DataLoader, optimizer, device, loss_fn, scaler=None, log_interval: int = 100) -> Tuple[float, dict]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []
    start = time.time()
    for step, batch in enumerate(loader, 1):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

        if step % log_interval == 0:
            print(f"  [train] step {step:5d} | avg loss {total_loss/max(total_samples,1):.4f} | {total_samples} samples | {time.time()-start:.1f}s")
            start = time.time()

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(labels_cat, logits_cat)
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, loss_fn) -> Tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []
    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(labels_cat, logits_cat)
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, metrics


def get_device(arg: str) -> torch.device:
    if arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(arg)


def main():
    ap = argparse.ArgumentParser(description="Train CNN on Fin Whale MAT spectrograms")
    ap.add_argument('--pos-dir', type=str, required=True, help='Directory with positive MAT files')
    ap.add_argument('--neg-dir', type=str, required=True, help='Directory with negative MAT files')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--pin-memory', action='store_true')
    ap.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--crop-size', type=int, default=96)
    ap.add_argument('--min-db', type=float, default=-80.0)
    ap.add_argument('--max-db', type=float, default=0.0)
    ap.add_argument('--train-ratio', type=float, default=0.8)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--balance', type=str, default='none', choices=['none', 'weighted', 'oversample'])
    ap.add_argument('--save-path', type=str, default='checkpoints/finwhale_cnn.pt')
    # WandB args
    ap.add_argument('--use_wandb', action='store_true', help='Enable logging to Weights & Biases')
    ap.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username or team) to use')
    ap.add_argument('--wandb_group', type=str, default=None, help='WandB group name for organizing runs')
    ap.add_argument('--wandb_project', type=str, default='finwhale_cnn', help='WandB project name')
    ap.add_argument('--exp_dir', type=str, default='exp/finwhale_cnn', help='Experiment directory for logs and checkpoints')
    ap.add_argument('--use-amp', action='store_true', help='Enable mixed precision on CUDA')
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"Device: {device}")

    # Create loaders
    train_loader, val_loader, test_loader = make_dataloaders(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        crop_size=args.crop_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_db=args.min_db,
        max_db=args.max_db,
        balance=args.balance,
        seed=args.seed,
    )

    # Class counts from train dataset
    train_ds: FinWhaleMatDataset = train_loader.dataset  # type: ignore
    labels = torch.tensor([lbl for _, lbl in train_ds.files], dtype=torch.long)
    counts = torch.bincount(labels, minlength=2).float()
    print(f"Train class counts: neg={int(counts[0].item())}, pos={int(counts[1].item())}")

    # Prepare experiment directory and wandb
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Persist full args for reproducibility
    try:
        import pickle
        with open(exp_dir / 'args.pkl', 'wb') as f:
            pickle.dump(args, f)
        print(f"Saved args to {exp_dir / 'args.pkl'}")
    except Exception as e:
        print(f"Warning: failed to save args.pkl: {e}")

    # Save exact split file lists for reproducibility
    try:
        split_dir = exp_dir / 'splits'
        split_dir.mkdir(parents=True, exist_ok=True)
        # Access datasets behind loaders
        val_ds: FinWhaleMatDataset = val_loader.dataset  # type: ignore
        test_ds: FinWhaleMatDataset = test_loader.dataset  # type: ignore
        def write_split(ds: FinWhaleMatDataset, name: str):
            with open(split_dir / f"{name}.txt", 'w') as f:
                for p, lbl in ds.files:
                    f.write(f"{p}\t{lbl}\n")
        write_split(train_ds, 'train')
        write_split(val_ds, 'val')
        write_split(test_ds, 'test')
        print(f"Saved split file lists under {split_dir}")
    except Exception as e:
        print(f"Warning: failed to save split files: {e}")

    # Attach additional attributes expected by wandb_utils
    args.use_wandb = bool(args.use_wandb)
    args.exp_dir = str(exp_dir)
    args.model = 'SmallCNN'
    args.dataset = 'FinWhaleMAT'
    args.n_epochs = args.epochs
    args.task = 'finetune_classification'

    run_id_file = exp_dir / 'wandb_run_id.txt'
    run_id = None
    if run_id_file.exists():
        try:
            run_id = run_id_file.read_text().strip()
        except Exception:
            pass
    if args.use_wandb:
        try:
            run = init_wandb(
                args,
                project_name=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                run_id=run_id,
            )
            if run is not None and (not run_id_file.exists()):
                # Persist run id for potential resume
                try:
                    run_id_file.write_text(run.id)
                except Exception:
                    pass
            print(f"[wandb] init ok: project={args.wandb_project}, group={args.wandb_group}")
        except Exception as e:
            print(f"[wandb] init failed: {e}. Disabling wandb.")
            args.use_wandb = False

    # Model, optimizer, loss
    model = SmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # If we are not using sampler-based balancing, use class-weighted CE
    if args.balance == 'none':
        weights = (counts.sum() / torch.clamp(counts, min=1.0))
        weights = (weights / weights.mean()).to(device)  # normalize for stability
        print(f"Class weights for CE: {weights.tolist()}")
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == 'cuda' else None

    best_val_f1 = -1.0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t0 = time.time()
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_fn, scaler)
        val_loss, val_metrics = evaluate(model, val_loader, device, loss_fn)
        dt = time.time() - t0
        print(f"[epoch {epoch}] train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"train acc {train_metrics['acc']:.3f} f1 {train_metrics['f1']:.3f} | "
              f"val acc {val_metrics['acc']:.3f} f1 {val_metrics['f1']:.3f} | {dt:.1f}s")

        # Log to wandb
        if args.use_wandb:
            log_training_metrics({
                'ft_train_loss': train_loss,
                'ft_acc': train_metrics['acc'],
                'ft_precision': train_metrics['precision'],
                'ft_recall': train_metrics['recall'],
                'ft_f1': train_metrics['f1'],
            }, use_wandb=True)
            # Validation metrics
            log_validation_metrics({
                'ft_val_loss': val_loss,
                'ft_acc': val_metrics['acc'],
                'ft_precision': val_metrics['precision'],
                'ft_recall': val_metrics['recall'],
                'ft_f1': val_metrics['f1'],
            }, task=args.task, epoch=epoch, prefix='', use_wandb=True)

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_metrics': val_metrics}, save_path)
            print(f"  [checkpoint] Saved new best to {save_path} (val f1={best_val_f1:.3f})")

    # Final test
    # Load best checkpoint
    if save_path.exists():
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')} for test evaluation")
    test_loss, test_metrics = evaluate(model, test_loader, device, loss_fn)
    print(f"\nTest: loss {test_loss:.4f} | acc {test_metrics['acc']:.3f} | "
          f"prec {test_metrics['precision']:.3f} | rec {test_metrics['recall']:.3f} | f1 {test_metrics['f1']:.3f}")

    if args.use_wandb:
        # Log final test metrics
        log_training_metrics({
            'ft_test_loss': test_loss,
            'ft_test_acc': test_metrics['acc'],
            'ft_test_precision': test_metrics['precision'],
            'ft_test_recall': test_metrics['recall'],
            'ft_test_f1': test_metrics['f1'],
        }, use_wandb=True)
        finish_run()


if __name__ == '__main__':
    main()
