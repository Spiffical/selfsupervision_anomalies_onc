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
    # Iterate devices; for each, request MAT data product covering 24 hours
    # SpectrogramDownloader handles path setup and processing
    for dev in ok_devices:
        try:
            print_status(f"Downloading last 24h for {dev} starting {start_dt.isoformat(timespec='seconds')}Z", 'PROGRESS')
            # Only create directories for devices we will actually download
            dl.setup_directories('mat', dev, 'last24h')
            # ONC client sometimes returns a harmless RuntimeWarning for metadata index; suppress it
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Metadata file.*', category=RuntimeWarning)
                dl.download_MAT_or_PNG(
                    deviceCode=dev,
                    start_date_object=start_dt,
                    filetype='mat',
                    spectrograms_per_batch=args.spectrograms_per_batch,
                    download_flac=False,
                )
            print_status(f"Finished {dev}", 'SUCCESS')
        except Exception as e:
            print_status(f"Failed {dev}: {e}", 'ERROR')


if __name__ == '__main__':
    main()


