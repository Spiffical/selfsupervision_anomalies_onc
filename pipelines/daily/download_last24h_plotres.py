#!/usr/bin/env python3
"""
Download last 24 hours of plotRes MAT spectrograms from ONC for one or more hydrophones.

Leverages the existing SpectrogramDownloader logic in utils/data/spectrogram_downloader.py
so filters (OD diversion mode, spectral downsample) match the plotRes settings we use elsewhere.

Example:
  # All active hydrophones discovered from deployments
  python pipelines/daily/download_last24h_plotres.py --data-dir /data/onc --all

  # Specific list of devices
  python pipelines/daily/download_last24h_plotres.py --data-dir /data/onc --devices ICLISTENHF1951,ICLISTENHF1354
"""
import os
import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import warnings
from pathlib import Path
from typing import List

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.download_spectrograms import load_config, print_status  # reuse .env loader and status printer
from utils.data.spectrogram_downloader import SpectrogramDownloader


def list_devices_from_deployments(dl: SpectrogramDownloader) -> List[str]:
    # Use cached deployments to enumerate device codes
    deps = dl._get_cached_deployments()
    return sorted(list({d.device_code for d in deps}))


def device_has_last24h_mat(dl: SpectrogramDownloader, device: str, start_dt: datetime, end_dt: datetime) -> bool:
    """Fast check: query archive for any MAT files in the window before launching data product."""
    # Reuse downloader's time formatting helper
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


def main():
    ap = argparse.ArgumentParser(description='Download last 24h plotRes MAT spectrograms from ONC')
    ap.add_argument('--data-dir', type=str, help='Parent directory for downloads (overrides .env DATA_DIR)')
    ap.add_argument('--devices', type=str, help='Comma-separated device codes (e.g. ICLISTENHF1951,ICLISTENHF1354)')
    ap.add_argument('--all', action='store_true', help='Download for all devices discovered from deployments')
    ap.add_argument('--spectrograms-per-batch', type=int, default=288, help='5-min spectrograms per request (24h=288)')
    ap.add_argument('--max-wait-minutes', type=int, default=45, help='Max time to wait for runs to be downloadable')
    ap.add_argument('--poll-interval-seconds', type=int, default=30, help='Polling interval between download attempts')
    ap.add_argument('--max-download-workers', type=int, default=4, help='Max parallel downloads during polling')
    args = ap.parse_args()

    onc_token, data_dir = load_config(data_dir_override=args.data_dir)
    dl = SpectrogramDownloader(onc_token, data_dir)

    # Decide devices
    devices: List[str] = []
    if args.devices:
        devices = [d.strip() for d in args.devices.split(',') if d.strip()]
    elif args.all:
        print_status('Enumerating devices from deployments...', 'INFO')
        devices = list_devices_from_deployments(dl)
    else:
        raise SystemExit('Specify --devices or --all')

    if not devices:
        raise SystemExit('No devices found to download')

    print_status(f'Target devices ({len(devices)}): {", ".join(devices)}', 'INFO')

    # Start/end window (UTC)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=24)

    # Pre-check which devices actually have MAT files in last 24h
    print_status('Checking which devices have MAT files in last 24h...', 'INFO')
    ok_devices: List[str] = []
    skipped_devices: List[str] = []
    for dev in devices:
        has_files = device_has_last24h_mat(dl, dev, start_dt, end_dt)
        if has_files:
            ok_devices.append(dev)
        else:
            skipped_devices.append(dev)
    print_status(f"Devices with data: {len(ok_devices)} | skipped (no data): {len(skipped_devices)}", 'INFO')
    if skipped_devices:
        print_status(f"Skipping: {', '.join(skipped_devices[:10])}{' ...' if len(skipped_devices)>10 else ''}", 'INFO')
    # Phase A: submit all runs (no-wait) and persist queue
    print_status('Submitting data product runs (no-wait)...', 'INFO')
    run_records = []
    for dev in ok_devices:
        try:
            # Ensure per-device output directories are set
            dl.setup_directories('mat', dev, 'last24h')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Metadata file.*', category=RuntimeWarning)
                rec = dl.submit_mat_run_no_wait(
                    deviceCode=dev,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    spectrograms_per_batch=args.spectrograms_per_batch,
                )
            run_records.append(rec)
            print_status(f"Submitted {dev} (dpRequestId={rec['dpRequestId']})", 'SUCCESS')
        except Exception as e:
            print_status(f"Submit failed {dev}: {e}", 'ERROR')

    # Save queue
    dl.save_runs('last24h', run_records)
    print_status(f"Saved run queue: {dl.runs_file_path('last24h')}", 'INFO')

    # Phase B: poll and download until all complete or timeout
    deadline = time.time() + (args.max_wait_minutes * 60)
    remaining = lambda: [r for r in run_records if r.get('status') != 'downloaded']
    pass_num = 0
    while time.time() < deadline and remaining():
        pass_num += 1
        todo = remaining()
        print_status(f"Poll pass {pass_num}: attempting {len(todo)} downloads", 'PROGRESS')

        with ThreadPoolExecutor(max_workers=args.max_download_workers) as ex:
            futures = {ex.submit(dl.try_download_run, r): r for r in todo}
            for fu in as_completed(futures):
                try:
                    status, updated = fu.result()
                    # Update in-memory record
                    for i, r in enumerate(run_records):
                        if r['dpRequestId'] == updated['dpRequestId']:
                            run_records[i] = updated
                            break
                except Exception as e:
                    base = futures[fu]
                    print_status(f"Download attempt error for {base.get('deviceCode')} dpRequestId={base.get('dpRequestId')}: {e}", 'ERROR')

        # Persist after each pass
        dl.save_runs('last24h', run_records)

        # If still pending, sleep
        if remaining():
            time.sleep(args.poll_interval_seconds)

    # Summary
    done = [r for r in run_records if r.get('status') == 'downloaded']
    pend = remaining()
    print_status(f"Completed: {len(done)} | Pending: {len(pend)}", 'INFO')


if __name__ == '__main__':
    main()


