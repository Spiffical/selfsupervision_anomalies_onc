#!/usr/bin/env python3
"""
Quick inspector for MATLAB spectrogram files (.mat) to understand keys, shapes, and counts.

It scans positive and negative folders (e.g., 'mat_files' and 'neg_mat_files'),
summarizes available variables (PdB_norm, P, F, T, etc.), reports shapes and
orientation (freq x time vs time x freq), and loads a few samples to compute
value ranges. Designed to help build a robust Dataset/DataLoader.

Example:
  python scripts/inspect_mat_files.py \
    --root "/Volumes/HydrophoneData/FinWhalesProject/data" \
    --pos-subdir mat_files --neg-subdir neg_mat_files \
    --sample 40 --load-stats 6

Optionally save a JSON report with --save-json report.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import scipy.io as sio
except Exception as e:
    sio = None  # We will error later with a clear message

try:
    import h5py
except Exception:
    h5py = None  # Optional; only needed for v7.3 MAT files


SPECTRO_KEYS = [
    'PdB_norm', 'power_db_norm', 'PdB', 'P_db',
    'P', 'PSD', 'psd', 'Sxx', 'S', 'spec', 'spectrogram', 'power_spectrogram'
]
FREQ_KEYS = ['F', 'frequencies', 'freqs', 'freq', 'f']
TIME_KEYS = ['T', 'times', 'time', 't']


def is_mat_v73(path: Path) -> bool:
    """Heuristically detect if the .mat file is v7.3 (HDF5-based)."""
    if h5py is None:
        return False
    try:
        with h5py.File(path, 'r') as f:  # noqa: F841
            return True
    except Exception:
        return False


def whosmat_shapes(path: Path) -> Dict[str, Tuple[int, ...]]:
    """Return variable shapes using scipy.io.whosmat for v5 mat files without loading data."""
    if sio is None:
        raise RuntimeError("scipy is required to inspect .mat files. Please install scipy.")
    try:
        info = sio.whosmat(str(path))
        return {name: tuple(shape) for name, shape, _ in info}
    except Exception:
        # Fall back to loading (last resort)
        data = sio.loadmat(str(path), simplify_cells=True)
        out = {}
        for k, v in data.items():
            if k.startswith('__'):
                continue
            try:
                arr = np.asarray(v)
                out[k] = arr.shape
            except Exception:
                pass
        return out


def h5_shapes(path: Path) -> Dict[str, Tuple[int, ...]]:
    """Return dataset shapes for HDF5-based (v7.3) .mat files."""
    shapes: Dict[str, Tuple[int, ...]] = {}
    assert h5py is not None
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                shapes[name.split('/')[-1]] = tuple(obj.shape)
        f.visititems(visit)
    return shapes


def reservoir_sample_mat_files(folder: Path, k: int, rng: random.Random, scan_limit: int = 0) -> Tuple[List[Path], int, bool]:
    """Reservoir sample up to k .mat files by streaming the directory (O(k) memory).

    Returns (sampled_paths, total_count).
    """
    reservoir: List[Path] = []
    n = 0
    limited = False
    # Use os.scandir for performance on large dirs
    for entry in os.scandir(folder):
        try:
            if not entry.is_file():
                continue
        except FileNotFoundError:
            continue
        name = entry.name
        if not name.lower().endswith('.mat'):
            continue
        n += 1
        if scan_limit > 0 and n > scan_limit:
            limited = True
            break
        if k <= 0:
            # Caller requested "all"; we'll collect them in a list
            reservoir.append(Path(entry.path))
            continue
        if len(reservoir) < k:
            reservoir.append(Path(entry.path))
        else:
            # Pick a random index in [1, n]
            j = rng.randint(1, n)
            if j <= k:
                reservoir[j - 1] = Path(entry.path)
    return reservoir, n, limited


def pick_key(candidates: List[str], available: Dict[str, Tuple[int, ...]]) -> Optional[str]:
    for k in candidates:
        if k in available and len(available[k]) >= 2:
            return k
    # Try case-insensitive
    lowered = {k.lower(): k for k in available.keys()}
    for k in candidates:
        if k.lower() in lowered and len(available[lowered[k.lower()]]) >= 2:
            return lowered[k.lower()]
    return None


def infer_axes_orientation(shape: Tuple[int, ...], f_len: Optional[int], t_len: Optional[int]) -> Tuple[str, Tuple[int, int]]:
    """Infer whether data is (F, T) or (T, F). Returns orientation label and (F, T) dims."""
    if len(shape) < 2:
        return ("unknown", (shape[0] if shape else 0, 0))
    r, c = shape[:2]
    # If we know lengths, prefer matching
    if f_len and t_len:
        if (r, c) == (f_len, t_len):
            return ("F x T", (r, c))
        if (r, c) == (t_len, f_len):
            return ("T x F", (c, r))
    # Otherwise, guess F is the smaller dimension (often fewer freq bins than time frames)
    if r <= c:
        return ("F x T?", (r, c))
    else:
        return ("T x F?", (c, r))


def load_for_stats(path: Path) -> Dict[str, Union[str, float, int, Tuple[int, ...]]]:
    """Fully load one file to compute value stats for the best-guess spectrogram array."""
    if sio is None:
        raise RuntimeError("scipy is required to inspect .mat files. Please install scipy.")
    data = sio.loadmat(str(path), simplify_cells=True)

    # Try direct key match
    spec_key = None
    for k in SPECTRO_KEYS:
        if k in data:
            spec_key = k
            break
    if spec_key is None:
        # Try case-insensitive
        lowered = {k.lower(): k for k in data.keys()}
        for k in SPECTRO_KEYS:
            if k.lower() in lowered:
                spec_key = lowered[k.lower()]
                break

    # If still none, pick the largest 2D array as a fallback
    arr = None
    if spec_key is not None:
        arr = np.asarray(data[spec_key])
    else:
        best = None
        for k, v in data.items():
            if k.startswith('__'):
                continue
            a = np.asarray(v)
            if a.ndim == 2:
                size = a.shape[0] * a.shape[1]
                if best is None or size > best[0]:
                    best = (size, k, a)
        if best is not None:
            _, spec_key, arr = best

    result = {
        'file': str(path),
        'chosen_key': spec_key or 'unknown',
        'ndim': int(arr.ndim) if arr is not None else -1,
        'shape': tuple(arr.shape) if arr is not None else tuple(),
        'dtype': str(arr.dtype) if arr is not None else 'unknown',
    }
    if arr is not None:
        arr = np.asarray(arr)
        finite = np.isfinite(arr)
        if finite.any():
            result.update(
                min=float(np.nanmin(arr)),
                max=float(np.nanmax(arr)),
                mean=float(np.nanmean(arr)),
                nan_frac=float(np.mean(~finite)),
            )
        else:
            result.update(min=float('nan'), max=float('nan'), mean=float('nan'), nan_frac=1.0)
    return result


def inspect_folder(folder: Path, sample_n: int, load_stats_n: int, rng: random.Random, quiet: bool = False, scan_limit: int = 0) -> Dict:
    if not quiet:
        print(f"[scan] Scanning and sampling in: {folder} (this may take a while)", flush=True)

    # Reservoir-sample for structure inspection
    t0 = time.time()
    inspect_files, total_count, limited_inspect = reservoir_sample_mat_files(folder, sample_n if sample_n > 0 else 0, rng, scan_limit=scan_limit)
    # Separate reservoir-sample for value stats
    stats_files, total_count2, limited_stats = reservoir_sample_mat_files(folder, load_stats_n if load_stats_n > 0 else 0, rng, scan_limit=scan_limit)
    # If we scanned twice, total_count may differ only if limit hit during one pass
    total_count = max(total_count, total_count2)
    limited = bool(limited_inspect or limited_stats)
    t1 = time.time()

    out: Dict = {
        'folder': str(folder),
        'count': int(total_count),
        'shapes_summary': {},
        'orientations': Counter(),
        'spec_keys_seen': Counter(),
        'freq_len_samples': [],
        'time_len_samples': [],
        'examples': [],
        'value_stats': [],
    }
    if total_count == 0:
        return out

    if not quiet:
        print(
            f"[scan] Found {total_count}{' (partial)' if limited else ''} .mat files | "
            f"inspecting {len(inspect_files)} | stats on {len(stats_files)} "
            f"(scan {t1 - t0:.1f}s)",
            flush=True,
        )

    for i, fp in enumerate(inspect_files, 1):
        if not quiet:
            print(f"[inspect] {folder.name}: {i}/{len(inspect_files)} {fp.name}", flush=True)
        try:
            if is_mat_v73(fp):
                shapes = h5_shapes(fp)
            else:
                shapes = whosmat_shapes(fp)
        except Exception:
            # As a fallback, attempt full load and infer
            shapes = {}
            try:
                if sio is not None:
                    data = sio.loadmat(str(fp), simplify_cells=True)
                    for k, v in data.items():
                        if k.startswith('__'):
                            continue
                        try:
                            arr = np.asarray(v)
                            shapes[k] = tuple(arr.shape)
                        except Exception:
                            pass
            except Exception:
                pass

        spec_key = pick_key(SPECTRO_KEYS, shapes) or 'unknown'
        freq_key = pick_key(FREQ_KEYS, shapes)
        time_key = pick_key(TIME_KEYS, shapes)

        f_len = shapes.get(freq_key, (None,))[0] if freq_key else None
        t_len = shapes.get(time_key, (None,))[0] if time_key else None

        if spec_key != 'unknown' and spec_key in shapes:
            orient, (F, T) = infer_axes_orientation(shapes[spec_key], f_len, t_len)
            out['orientations'][orient] += 1
            out['shapes_summary'][(F, T)] = out['shapes_summary'].get((F, T), 0) + 1
            if f_len:
                out['freq_len_samples'].append(int(F))
            if t_len:
                out['time_len_samples'].append(int(T))
        else:
            out['orientations']['unknown'] += 1

        out['spec_keys_seen'][spec_key] += 1
        out['examples'].append({
            'file': str(fp),
            'spec_key': spec_key,
            'freq_key': freq_key,
            'time_key': time_key,
            'shapes': {k: list(v) for k, v in shapes.items() if k in (spec_key, freq_key, time_key) and v},
        })

    # Load a few files fully for value stats
    if load_stats_n > 0:
        for j, fp in enumerate(stats_files, 1):
            if not quiet:
                print(f"[values] {folder.name}: {j}/{len(stats_files)} {fp.name}", flush=True)
            try:
                out['value_stats'].append(load_for_stats(fp))
            except Exception as e:
                out['value_stats'].append({'file': str(fp), 'error': str(e)})

    return out


def summarise_report(pos: Dict, neg: Dict) -> Dict:
    def basic(d: Dict) -> Dict:
        shapes_counter = Counter(d.get('shapes_summary', {}))
        common_shapes = shapes_counter.most_common(5)
        freq_lens = d.get('freq_len_samples', [])
        time_lens = d.get('time_len_samples', [])
        summary = {
            'folder': d.get('folder'),
            'count': d.get('count', 0),
            'top_shapes_(F,T)': [(tuple(map(int, s)), int(c)) for s, c in common_shapes],
            'orientations': dict(d.get('orientations', {})),
            'spec_keys_seen': dict(d.get('spec_keys_seen', {})),
        }
        if freq_lens:
            summary['freq_len_stats'] = {
                'min': int(min(freq_lens)),
                'median': int(statistics.median(freq_lens)),
                'max': int(max(freq_lens)),
            }
        if time_lens:
            summary['time_len_stats'] = {
                'min': int(min(time_lens)),
                'median': int(statistics.median(time_lens)),
                'max': int(max(time_lens)),
            }
        # Value range aggregate
        vals = [vs for vs in d.get('value_stats', []) if 'min' in vs]
        if vals:
            summary['value_range'] = {
                'min': float(min(v['min'] for v in vals)),
                'max': float(max(v['max'] for v in vals)),
                'mean_mean': float(statistics.mean(v['mean'] for v in vals)),
                'samples': len(vals)
            }
        return summary

    rep = {
        'positive': basic(pos),
        'negative': basic(neg),
        'class_balance_ratio_pos_to_neg': (
            float(pos.get('count', 0)) / float(neg.get('count', 1)) if neg.get('count', 0) else None
        )
    }

    # Recommend a square crop size based on shared smallest dimension across common shapes
    def smallest_dim(d: Dict) -> Optional[int]:
        shapes = [s for s, _ in Counter(d.get('shapes_summary', {})).most_common(10)]
        if not shapes:
            return None
        # shapes are (F,T); pick min across their mins
        mins = [min(F, T) for (F, T) in shapes]
        return int(min(mins)) if mins else None

    pos_min = smallest_dim(pos)
    neg_min = smallest_dim(neg)
    if pos_min and neg_min:
        rep['recommended_square_crop'] = int(min(pos_min, neg_min))
    elif pos_min:
        rep['recommended_square_crop'] = int(pos_min)

    return rep


def main():
    ap = argparse.ArgumentParser(description="Inspect MATLAB spectrogram .mat files")
    ap.add_argument('--root', type=str, required=True, help='Root directory containing subfolders')
    ap.add_argument('--pos-subdir', type=str, default='mat_files', help='Subfolder with positive (whale) .mat files')
    ap.add_argument('--neg-subdir', type=str, default='neg_mat_files', help='Subfolder with negative (no whale) .mat files')
    ap.add_argument('--sample', type=int, default=40, help='Number of files to sample per class for structural inspection (0 = all)')
    ap.add_argument('--load-stats', type=int, default=6, help='Number of files to fully load per class for value stats (min/max/mean)')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for sampling')
    ap.add_argument('--quiet', action='store_true', help='Reduce progress output')
    ap.add_argument('--scan-limit', type=int, default=0, help='Limit the number of .mat files scanned per folder (0 = no limit)')
    ap.add_argument('--save-json', type=str, default=None, help='Optional path to save JSON report')
    args = ap.parse_args()

    root = Path(args.root)
    pos_dir = root / args.pos_subdir
    neg_dir = root / args.neg_subdir

    if not pos_dir.exists():
        raise SystemExit(f"Positive directory not found: {pos_dir}")
    if not neg_dir.exists():
        raise SystemExit(f"Negative directory not found: {neg_dir}")
    if sio is None:
        raise SystemExit("scipy is required (pip install scipy)")

    rng = random.Random(args.seed)

    pos_report = inspect_folder(pos_dir, args.sample, args.load_stats, rng, quiet=args.quiet, scan_limit=args.scan_limit)
    neg_report = inspect_folder(neg_dir, args.sample, args.load_stats, rng, quiet=args.quiet, scan_limit=args.scan_limit)

    report = summarise_report(pos_report, neg_report)

    # Pretty print
    def pp(d):
        print(json.dumps(d, indent=2))

    print("=== INSPECTION SUMMARY ===")
    pp(report)

    print("\n=== POSITIVE (examples) ===")
    for ex in pos_report.get('examples', [])[:5]:
        print(json.dumps(ex, indent=2))

    print("\n=== NEGATIVE (examples) ===")
    for ex in neg_report.get('examples', [])[:5]:
        print(json.dumps(ex, indent=2))

    print("\n=== VALUE STATS (positive) ===")
    for vs in pos_report.get('value_stats', []):
        print(json.dumps(vs, indent=2))

    print("\n=== VALUE STATS (negative) ===")
    for vs in neg_report.get('value_stats', []):
        print(json.dumps(vs, indent=2))

    if args.save_json:
        out = {
            'summary': report,
            'positive': pos_report,
            'negative': neg_report,
        }
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nSaved JSON report to: {out_path}")


if __name__ == '__main__':
    main()
