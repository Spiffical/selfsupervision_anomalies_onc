#!/usr/bin/env python3
"""
Daily VM-side pipeline:
 - Scan for recent plotRes .mat spectrograms (default last 24h)
 - Convert each to prepared .npy using H5-like preprocessing (resize -> normalize), shape [F, T]
 - Write a manifest.csv with metadata for downstream inference
 - Optionally delete original .mat after successful conversion

Usage example:
  python pipelines/daily/prepare_daily_spectrograms.py \
    --mat-root /data/onc/matfiles \
    --output-root /data/onc/prepared \
    --since-hours 24 --num-workers 8 --delete-raw
"""
import argparse
import os
import sys
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure repository src/ path is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
from src.ssamba.utilities.spectrogram_utils import (
    load_mat_spectrogram,
    resize_to_target,
    normalize_spectrogram,
)


def find_recent_mat_files(root: Path, pattern: str, since_hours: int) -> List[Path]:
    cutoff = time.time() - since_hours * 3600
    out: List[Path] = []
    for p in root.rglob(pattern):
        try:
            if p.is_file() and p.stat().st_mtime >= cutoff:
                out.append(p)
        except FileNotFoundError:
            continue
    return sorted(out)


def prepare_one(
    mat_path: Path,
    output_root: Path,
    expected_shape: Tuple[int, int],
    target_size: Tuple[int, int],
    dataset_mean: float = None,
    dataset_std: float = None,
    amount: float = 1.0,
    by_day: bool = False,
) -> Tuple[Path, dict]:
    """
    Convert a single .mat to prepared .npy using H5-like preprocessing.
    Returns (output_npy_path, manifest_row_dict).
    """
    # Try parse hydrophone and timestamps from filename
    base = mat_path.stem
    parts = base.split('_')
    hydro = parts[0] if parts else ''
    t0 = ''
    t1 = ''
    if len(parts) >= 3:
        t0 = parts[1]
        t1 = parts[2].split('-')[0]
    # Day string
    day_str = ''
    if len(t0) >= 8:
        y, m, d = t0[:4], t0[4:6], t0[6:8]
        day_str = f"{y}-{m}-{d}"
    if not day_str:
        ts = time.localtime(mat_path.stat().st_mtime)
        day_str = time.strftime('%Y-%m-%d', ts)

    # Decide output directory
    if by_day:
        out_dir = output_root / day_str / hydro
    else:
        # Shallow mirror: keep last 2 parent components
        keep_parts = mat_path.parts[-3:]
        out_dir = output_root.joinpath(*keep_parts[:-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(base).with_suffix('.npy').name)

    arr = load_mat_spectrogram(str(mat_path), expected_shape)
    # H5-like: resize -> normalize
    arr = resize_to_target(arr, target_size)
    arr = normalize_spectrogram(arr, dataset_mean=dataset_mean, dataset_std=dataset_std, amount=amount)
    np.save(out_path, arr.astype(np.float32))

    meta = {
        'npy_path': str(out_path),
        'hydrophone': hydro,
        't0': t0,
        't1': t1,
        'original_mat': str(mat_path),
        'fdim': target_size[0],
        'tdim': target_size[1],
        'pipeline': 'h5_like_v1',
        'day': day_str,
    }
    return out_path, meta


def main():
    ap = argparse.ArgumentParser(description='Prepare recent plotRes .mat files into normalized/resized .npy spectrograms')
    ap.add_argument('--mat-root', type=str, required=True, help='Root directory containing raw .mat files')
    ap.add_argument('--output-root', type=str, required=True, help='Root directory to write prepared .npy and manifest.csv')
    ap.add_argument('--since-hours', type=int, default=24, help='How many hours back to include (default: 24)')
    ap.add_argument('--pattern', type=str, default='*spect_plotRes.mat', help='Glob pattern to find plotRes mats')
    ap.add_argument('--target-size', type=int, nargs=2, default=[512, 512], metavar=('F', 'T'))
    ap.add_argument('--expected-shape', type=int, nargs=2, default=[854, 1000], metavar=('F', 'T'))
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--delete-raw', action='store_true', help='Delete .mat after successful conversion')
    ap.add_argument('--dataset-mean', type=float, default=None)
    ap.add_argument('--dataset-std', type=float, default=None)
    ap.add_argument('--amount', type=float, default=1.0)
    ap.add_argument('--args-pkl', type=str, default=None, help='Path to model args.pkl to auto-fill dataset_mean/std')
    ap.add_argument('--by-day', action='store_true', help='Organize prepared output as YYYY-MM-DD/HYDROPHONE')
    args = ap.parse_args()

    mat_root = Path(args.mat_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Optional: load dataset stats from args.pkl
    if args.args_pkl and (args.dataset_mean is None or args.dataset_std is None):
        try:
            import pickle
            with open(args.args_pkl, 'rb') as f:
                margs = pickle.load(f)
            if args.dataset_mean is None and getattr(margs, 'dataset_mean', None) is not None:
                args.dataset_mean = float(getattr(margs, 'dataset_mean'))
            if args.dataset_std is None and getattr(margs, 'dataset_std', None) is not None:
                args.dataset_std = float(getattr(margs, 'dataset_std'))
            print(f"Using stats from args.pkl - mean: {args.dataset_mean}, std: {args.dataset_std}")
        except Exception as e:
            print(f"Warning: failed to read args.pkl ({e}); proceeding with provided stats")

    print(f"Scanning {mat_root} for recent mats (pattern={args.pattern}, last {args.since_hours}h)...")
    mats = find_recent_mat_files(mat_root, args.pattern, args.since_hours)
    print(f"Found {len(mats)} files")

    manifest_rows = []
    if mats:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futs = [
                ex.submit(
                    prepare_one,
                    m,
                    out_root,
                    tuple(args.expected_shape),
                    tuple(args.target_size),
                    args.dataset_mean,
                    args.dataset_std,
                    args.amount,
                    args.by_day,
                ) for m in mats
            ]
            for fut in as_completed(futs):
                try:
                    out_path, meta = fut.result()
                    manifest_rows.append(meta)
                except Exception as e:
                    print(f"Error preparing file: {e}")

    # Write manifest CSV
    if manifest_rows:
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        manifest_path = out_root / f'manifest_{ts}.csv'
        with manifest_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            for row in manifest_rows:
                w.writerow(row)
        print(f"Wrote manifest: {manifest_path}")

    # Optionally delete raw mats after success
    if args.delete_raw and manifest_rows:
        produced = {Path(r['original_mat']).resolve() for r in manifest_rows}
        deleted = 0
        for p in produced:
            try:
                os.remove(p)
                deleted += 1
            except Exception:
                pass
        print(f"Deleted {deleted} original .mat files")


if __name__ == '__main__':
    main()


