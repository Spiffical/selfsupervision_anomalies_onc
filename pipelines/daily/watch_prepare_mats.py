#!/usr/bin/env python3
"""
Watch a directory tree for newly downloaded plotRes .mat spectrograms and prepare them on the fly.

Behavior:
 - Periodically scans --mat-root for files matching --pattern (default *spect_plotRes.mat)
 - Processes files in batches (up to --batch-size) with a ThreadPoolExecutor of --num-workers
 - For each .mat: loads, resizes, normalizes (H5-like), saves .npy to --output-root, then deletes the .mat
 - Exits after being idle for --exit-when-idle seconds (no new files found), unless --no-exit is set

Example:
  python pipelines/daily/watch_prepare_mats.py \
    --mat-root /data/onc \
    --output-root /data/onc/prepared \
    --args-pkl /path/to/model/args.pkl \
    --num-workers 8 --batch-size 64 --scan-interval 10
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np

from src.ssamba.utilities.spectrogram_utils import (
    load_mat_spectrogram,
    resize_to_target,
    normalize_spectrogram,
)


def prepare_one(mat_path: Path,
                output_root: Path,
                expected_shape: Tuple[int, int],
                target_size: Tuple[int, int],
                dataset_mean: float | None,
                dataset_std: float | None,
                amount: float,
) -> Path:
    """Convert one .mat -> .npy (H5-like). Returns output npy path on success."""
    # Create a mirrored yet shallow structure under output_root
    keep_parts = mat_path.parts[-3:]
    out_dir = output_root.joinpath(*keep_parts[:-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(keep_parts[-1]).with_suffix('.npy').name)

    arr = load_mat_spectrogram(str(mat_path), expected_shape)
    arr = resize_to_target(arr, target_size)
    arr = normalize_spectrogram(arr, dataset_mean=dataset_mean, dataset_std=dataset_std, amount=amount)
    np.save(out_path, arr.astype(np.float32))
    # Delete raw on success
    try:
        os.remove(mat_path)
    except Exception:
        pass
    return out_path


def load_stats_from_args_pkl(args_pkl: str) -> tuple[float | None, float | None]:
    if not args_pkl:
        return None, None
    try:
        import pickle
        with open(args_pkl, 'rb') as f:
            margs = pickle.load(f)
        mean = float(getattr(margs, 'dataset_mean')) if getattr(margs, 'dataset_mean', None) is not None else None
        std = float(getattr(margs, 'dataset_std')) if getattr(margs, 'dataset_std', None) is not None else None
        return mean, std
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser(description='Watch and prepare .mat spectrograms as they arrive')
    ap.add_argument('--mat-root', type=str, required=True)
    ap.add_argument('--output-root', type=str, required=True)
    ap.add_argument('--pattern', type=str, default='*spect_plotRes.mat')
    ap.add_argument('--expected-shape', type=int, nargs=2, default=[854, 1000])
    ap.add_argument('--target-size', type=int, nargs=2, default=[512, 512])
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--scan-interval', type=int, default=10, help='Seconds between scans')
    ap.add_argument('--exit-when-idle', type=int, default=600, help='Exit after this many idle seconds')
    ap.add_argument('--no-exit', action='store_true', help='Do not exit when idle')
    ap.add_argument('--dataset-mean', type=float, default=None)
    ap.add_argument('--dataset-std', type=float, default=None)
    ap.add_argument('--args-pkl', type=str, default=None, help='Load dataset_mean/std from model args.pkl if not provided')
    ap.add_argument('--amount', type=float, default=1.0)
    args = ap.parse_args()

    mat_root = Path(args.mat_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Init stats
    dmean = args.dataset_mean
    dstd = args.dataset_std
    if (dmean is None or dstd is None) and args.args_pkl:
        amean, astd = load_stats_from_args_pkl(args.args_pkl)
        if dmean is None:
            dmean = amean
        if dstd is None:
            dstd = astd

    idle_since = time.time()
    print(f"Watching {mat_root} for '{args.pattern}' ... (batch={args.batch_size}, workers={args.num_workers})")
    while True:
        # Find current .mat candidates
        mats: List[Path] = [p for p in mat_root.rglob(args.pattern) if p.is_file()]
        if mats:
            idle_since = time.time()
            batch = mats[: args.batch_size]
            print(f"Preparing batch of {len(batch)} .mat files...")
            futures = []
            with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                for m in batch:
                    futures.append(ex.submit(
                        prepare_one,
                        m,
                        out_root,
                        tuple(args.expected_shape),
                        tuple(args.target_size),
                        dmean,
                        dstd,
                        args.amount,
                    ))
                for fut in as_completed(futures):
                    try:
                        outp = fut.result()
                        print(f"✓ Prepared: {outp}")
                    except Exception as e:
                        print(f"✗ Error preparing file: {e}")
        else:
            # No files found
            if not args.no_exit and (time.time() - idle_since) > args.exit_when_idle:
                print("Idle timeout reached. Exiting.")
                break
            time.sleep(args.scan_interval)


if __name__ == '__main__':
    main()


