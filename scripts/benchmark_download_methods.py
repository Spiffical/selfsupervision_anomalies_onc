#!/usr/bin/env python3
"""
Benchmark ONC MAT downloads: blocking (waitComplete) vs submit-then-poll (parallel).

Defaults: up to 10 hydrophones, 100 spectrograms each (≈8h20m).

Usage examples:
  python scripts/benchmark_download_methods.py --data-dir /data/onc --method both
  python scripts/benchmark_download_methods.py --data-dir /data/onc --method parallel --max-devices 5 --spectrograms-per-device 60
"""
import sys
import os
import time
import glob
import statistics
import argparse
import random
import warnings
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Repo path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.download_spectrograms import load_config, print_status
from utils.data.spectrogram_downloader import SpectrogramDownloader


def device_has_mat(dl: SpectrogramDownloader, device: str, start_dt: datetime, end_dt: datetime) -> bool:
    """Quick archive check for any MAT files within window."""
    time_delta = end_dt - start_dt
    start_time, end_time = dl.start_and_end_strings(start_dt, time_delta)
    try:
        filters = {
            'deviceCode': device,
            'dateFrom': start_time,
            'dateTo': end_time,
            'extension': 'mat',
        }
        result = dl.onc.getListByDevice(filters, allPages=True)
        files = result.get('files') or []
        return len(files) > 0
    except Exception:
        return False


def choose_devices(dl: SpectrogramDownloader, max_devices: int, window_hours: float) -> List[str]:
    """Pick up to max_devices with MAT data in the recent window."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=window_hours)
    deps = dl._get_cached_deployments()
    all_devices = sorted(list({d.device_code for d in deps}))
    random.shuffle(all_devices)

    picked: List[str] = []
    for dev in all_devices:
        if len(picked) >= max_devices:
            break
        if device_has_mat(dl, dev, start_dt, end_dt):
            picked.append(dev)
    return picked


def _count_processed_mat(path: str) -> int:
    return len(glob.glob(os.path.join(path, 'processed', '*.mat')))


def _count_rejects_mat(path: str) -> int:
    return len(glob.glob(os.path.join(path, 'rejects', '*.mat')))


def run_blocking(dl: SpectrogramDownloader, devices: List[str], spectrograms_per_device: int, fixed_start: datetime, fixed_end: datetime) -> Tuple[float, Dict[str, str]]:
    """Blocking baseline: sequential per-device downloads using waitComplete path."""
    start_wall = time.time()
    statuses: Dict[str, str] = {}
    details: Dict[str, Dict[str, Any]] = {}

    # Use fixed window to match parallel method exactly
    start_dt = fixed_start
    end_dt = fixed_end

    for dev in devices:
        try:
            print_status(f"[blocking] {dev}: requesting {spectrograms_per_device} spectrograms", 'PROGRESS')
            dl.setup_directories('mat', dev, 'benchmark_blocking')
            # count before (processed subdir)
            before_proc = _count_processed_mat(dl.input_path)
            before_rej = _count_rejects_mat(dl.input_path)
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Metadata file.*', category=RuntimeWarning)
                dl.download_MAT_or_PNG(
                    deviceCode=dev,
                    start_date_object=start_dt,
                    filetype='mat',
                    spectrograms_per_batch=spectrograms_per_device,
                    download_flac=False,
                )
            elapsed = time.time() - t0
            after_proc = _count_processed_mat(dl.input_path)
            after_rej = _count_rejects_mat(dl.input_path)
            files_dl = max(0, (after_proc + after_rej) - (before_proc + before_rej))
            statuses[dev] = 'downloaded'
            details[dev] = {
                'status': 'downloaded',
                'seconds': elapsed,
                'files': files_dl,
            }
            print_status(f"[blocking] {dev}: done", 'SUCCESS')
        except Exception as e:
            statuses[dev] = f'error: {e}'
            print_status(f"[blocking] {dev}: failed: {e}", 'ERROR')
            details[dev] = {
                'status': statuses[dev],
                'seconds': None,
                'files': None,
            }

    return time.time() - start_wall, statuses, details


def run_parallel(dl: SpectrogramDownloader, devices: List[str], spectrograms_per_device: int, max_wait_minutes: int, poll_interval_seconds: int, max_download_workers: int, fixed_start: datetime, fixed_end: datetime) -> Tuple[float, Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Submit all runs (no wait), then poll/download in parallel until done or timeout."""
    start_wall = time.time()
    statuses: Dict[str, str] = {}
    details: Dict[str, Dict[str, Any]] = {}

    # Use fixed window provided by caller
    start_dt = fixed_start
    end_dt = fixed_end

    # Submit
    run_records: List[Dict[str, Any]] = []
    print_status('Submitting runs (parallel test)...', 'INFO')
    for dev in devices:
        try:
            dl.setup_directories('mat', dev, 'benchmark_parallel')
            # track baseline counts for later verification
            base_counts = _count_processed_mat(dl.input_path) + _count_rejects_mat(dl.input_path)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Metadata file.*', category=RuntimeWarning)
                rec = dl.submit_mat_run_no_wait(
                    deviceCode=dev,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    spectrograms_per_batch=spectrograms_per_device,
                )
            run_records.append(rec)
            statuses[dev] = 'submitted'
            details[dev] = {
                'status': 'submitted',
                'seconds': None,
                'files': base_counts,  # baseline count before downloads
                'dpRequestId': rec.get('dpRequestId'),
                'createdAt': rec.get('createdAt'),
            }
        except Exception as e:
            statuses[dev] = f'error: {e}'
            print_status(f"[parallel] submit failed {dev}: {e}", 'ERROR')
            details[dev] = {
                'status': statuses[dev],
                'seconds': None,
                'files': None,
                'dpRequestId': None,
                'createdAt': None,
            }

    # Persist queue for observability
    dl.save_runs('last24h', run_records)

    # Poll loop
    deadline = time.time() + (max_wait_minutes * 60)
    pass_num = 0
    def outstanding():
        return [r for r in run_records if r.get('status') != 'downloaded']

    while time.time() < deadline and outstanding():
        pass_num += 1
        todo = outstanding()
        print_status(f"Poll pass {pass_num}: attempting {len(todo)} downloads", 'PROGRESS')
        with ThreadPoolExecutor(max_workers=max_download_workers) as ex:
            futures = {ex.submit(dl.try_download_run, r): r for r in todo}
            for fu in as_completed(futures):
                try:
                    status, updated = fu.result()
                    # update run_records
                    for i, r in enumerate(run_records):
                        if r['dpRequestId'] == updated['dpRequestId']:
                            run_records[i] = updated
                            break
                    if updated.get('deviceCode'):
                        dev = updated['deviceCode']
                        if status == 'downloaded':
                            statuses[dev] = 'downloaded'
                        else:
                            statuses[dev] = status
                        # compute per-device elapsed if possible
                        created = updated.get('createdAt')
                        completed = updated.get('completedAt')
                        elapsed = None
                        if created and completed:
                            try:
                                # parse ISO with Z
                                def parse_iso_z(s: str) -> datetime:
                                    # support both ms and us
                                    try:
                                        return datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
                                    except ValueError:
                                        return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                                elapsed = (parse_iso_z(completed) - parse_iso_z(created)).total_seconds()
                            except Exception:
                                elapsed = None
                        details[dev] = {
                            'status': statuses[dev],
                            'seconds': elapsed,
                            'files': None,  # will fill after loop
                            'dpRequestId': updated.get('dpRequestId'),
                            'createdAt': created,
                            'completedAt': completed,
                        }
                except Exception as e:
                    base = futures[fu]
                    dev = base.get('deviceCode')
                    statuses[dev] = f'error: {e}'
                    details[dev] = {
                        'status': statuses[dev],
                        'seconds': None,
                        'files': None,
                        'dpRequestId': base.get('dpRequestId'),
                        'createdAt': base.get('createdAt'),
                    }
        # persist after each pass
        dl.save_runs('last24h', run_records)
        if outstanding():
            time.sleep(poll_interval_seconds)

    # finalize statuses for any remaining
    for r in run_records:
        dev = r.get('deviceCode')
        if r.get('status') == 'downloaded':
            statuses[dev] = 'downloaded'
        else:
            statuses[dev] = r.get('status', 'pending')
        # Keep any missing details filled from record
        if dev not in details:
            details[dev] = {
                'status': statuses[dev],
                'seconds': None,
                'files': None,
                'dpRequestId': r.get('dpRequestId'),
                'createdAt': r.get('createdAt'),
                'completedAt': r.get('completedAt'),
            }

    # After polling completes, compute file counts from processed subdir for each device
    for dev in devices:
        # Recreate directory path used earlier
        dl.setup_directories('mat', dev, 'benchmark_parallel')
        details[dev]['files'] = _count_processed_mat(dl.input_path) + _count_rejects_mat(dl.input_path)

    return time.time() - start_wall, statuses, details


def summarize(label: str, wall_seconds: float, statuses: Dict[str, str]) -> None:
    ok = sum(1 for s in statuses.values() if s == 'downloaded')
    pending = sum(1 for s in statuses.values() if s not in ('downloaded',) and not str(s).startswith('error'))
    errors = sum(1 for s in statuses.values() if str(s).startswith('error'))
    print_status(f"{label}: time={wall_seconds:.1f}s, downloaded={ok}, pending={pending}, errors={errors}", 'INFO')


def detailed_summary(blocking: Tuple[float, Dict[str, str], Dict[str, Dict[str, Any]]] = None,
                     parallel: Tuple[float, Dict[str, str], Dict[str, Dict[str, Any]]] = None) -> None:
    print('\n================ Detailed Summary ================')
    if blocking is not None:
        t_block, st_block, det_block = blocking
        print(f"Blocking total time: {t_block:.1f}s")
        if det_block:
            secs = [v['seconds'] for v in det_block.values() if isinstance(v.get('seconds'), (int, float))]
            if secs:
                print(f"  Per-device time: median={statistics.median(secs):.1f}s, min={min(secs):.1f}s, max={max(secs):.1f}s")
        print('  Devices:')
        for dev in sorted(det_block.keys()):
            info = det_block[dev]
            print(f"    - {dev}: status={info.get('status')}, seconds={info.get('seconds')}, files={info.get('files')}")

    if parallel is not None:
        t_par, st_par, det_par = parallel
        print(f"\nParallel total time: {t_par:.1f}s")
        if det_par:
            secs = [v['seconds'] for v in det_par.values() if isinstance(v.get('seconds'), (int, float))]
            if secs:
                print(f"  Per-device elapsed (submit→complete): median={statistics.median(secs):.1f}s, min={min(secs):.1f}s, max={max(secs):.1f}s")
        print('  Devices:')
        for dev in sorted(det_par.keys()):
            info = det_par[dev]
            print(f"    - {dev}: status={info.get('status')}, seconds={info.get('seconds')}, dpRequestId={info.get('dpRequestId')}")

    # Cross-method comparison if both present
    if blocking is not None and parallel is not None:
        _, _, det_block = blocking
        _, _, det_par = parallel
        print('\nPer-device file count comparison (processed/*.mat):')
        for dev in sorted(set(list(det_block.keys()) + list(det_par.keys()))):
            b = det_block.get(dev, {}).get('files')
            p = det_par.get(dev, {}).get('files')
            match = (b == p) if (b is not None and p is not None) else False
            print(f"  - {dev}: blocking={b} | parallel={p} | match={match}")

    if blocking is not None and parallel is not None:
        t_block, _, _ = blocking
        t_par, _, _ = parallel
        if t_par > 0:
            speedup = t_block / t_par
            print(f"\nOverall speedup (blocking/parallel): {speedup:.2f}x")
    print('================================================\n')


def main():
    ap = argparse.ArgumentParser(description='Benchmark blocking vs parallel submit-then-poll MAT downloads')
    ap.add_argument('--data-dir', type=str, help='Parent directory for downloads (overrides .env DATA_DIR)')
    ap.add_argument('--max-devices', type=int, default=10, help='Max hydrophones to test')
    ap.add_argument('--spectrograms-per-device', type=int, default=100, help='5-min spectrograms per device')
    ap.add_argument('--method', type=str, default='both', choices=['both', 'blocking', 'parallel'], help='Which method to run')
    ap.add_argument('--max-wait-minutes', type=int, default=45, help='Parallel: max wait minutes')
    ap.add_argument('--poll-interval-seconds', type=int, default=30, help='Parallel: polling interval')
    ap.add_argument('--max-download-workers', type=int, default=4, help='Parallel: max concurrent downloads')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for device sampling')
    args = ap.parse_args()

    random.seed(args.seed)

    onc_token, data_dir = load_config(data_dir_override=args.data_dir)
    dl = SpectrogramDownloader(onc_token, data_dir)

    # Pick devices with recent data (use 24h to maximize hit rate)
    print_status('Enumerating devices with MAT files in last 24h...', 'INFO')
    devices = choose_devices(dl, max_devices=args.max_devices, window_hours=24)
    if not devices:
        raise SystemExit('No devices with MAT files found in last 24h')
    print_status(f"Testing devices ({len(devices)}): {', '.join(devices)}", 'INFO')

    blocking_result = None
    parallel_result = None

    # Fixed window used by both methods
    duration_seconds = (args.spectrograms_per_device - 1) * 300
    fixed_end = datetime.now(timezone.utc)
    fixed_start = fixed_end - timedelta(seconds=duration_seconds)

    if args.method in ('both', 'blocking'):
        t_block, st_block, det_block = run_blocking(dl, devices, args.spectrograms_per_device, fixed_start, fixed_end)
        summarize('Blocking', t_block, st_block)
        blocking_result = (t_block, st_block, det_block)

    if args.method in ('both', 'parallel'):
        t_par, st_par, det_par = run_parallel(
            dl,
            devices,
            args.spectrograms_per_device,
            args.max_wait_minutes,
            args.poll_interval_seconds,
            args.max_download_workers,
            fixed_start,
            fixed_end,
        )
        summarize('Parallel', t_par, st_par)
        parallel_result = (t_par, st_par, det_par)

    print_status('Benchmark complete', 'SUCCESS')
    # Detailed summary at the end
    detailed_summary(blocking_result, parallel_result)


if __name__ == '__main__':
    main()


