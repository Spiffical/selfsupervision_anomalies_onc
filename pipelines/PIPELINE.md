# Daily ONC Spectrogram Pipeline (Download → Prepare → Infer → Dashboard)

This document describes the production pipeline that:
- downloads ONC plotRes spectrogram `.mat` files for the last 24 hours
- prepares them into normalized/resized `.npy` arrays for inference
- optionally runs local inference and emits dashboard-friendly outputs
- supports continuous scheduling and per-day folder organization

You can copy/paste sections below to give an LLM full context for dashboard changes.

---

## Components

### 1) VM-side downloader
File: `pipelines/daily/download_last24h_plotres.py`
- Uses existing `SpectrogramDownloader` (see `utils/data/spectrogram_downloader.py`).
- Pre-checks which devices have MATs in last 24h (via `getListByDevice`).
- Runs one last-24h data product per device with data.
- Output goes into `DATA_DIR` under the downloader’s structure (e.g. `data/onc/ICLISTENHF1951/last24h/mat/processed/*.mat`).

Usage:
```bash
python pipelines/daily/download_last24h_plotres.py \
  --data-dir /data/onc \
  --all                       # or --devices ICLISTENHF1951,ICLISTENHF1354
```
Notes:
- UTC window [now-24h, now].
- Suppresses harmless ONC metadata warnings.
- Skips devices with no data; no empty folders are created.

### 2) VM-side continuous watcher (prepare + delete raw)
File: `pipelines/daily/watch_prepare_mats.py`
- Periodically scans `--mat-root` for new plotRes `.mat` files.
- Processes batches in parallel: load → resize (to target) → normalize → save `.npy`.
- Deletes `.mat` after successful conversion.
- Optional idle timeout or `--no-exit` for continuous mode.

Example:
```bash
python pipelines/daily/watch_prepare_mats.py \
  --mat-root /data/onc \
  --output-root /data/onc/prepared \
  --num-workers 8 --batch-size 64 --scan-interval 10 --no-exit \
  --args-pkl /path/to/model/args.pkl      # auto-fill dataset_mean/std
```

### 3) VM-side one-shot preparer (by-day folders)
File: `pipelines/daily/prepare_daily_spectrograms.py`
- Converts all matching `.mat` in the last N hours to `.npy`.
- With `--by-day`, organizes as `YYYY-MM-DD/HYDROPHONE/*.npy` and writes a manifest CSV.
- Manifest columns: `npy_path, hydrophone, t0, t1, original_mat, fdim, tdim, pipeline, day`.

Example:
```bash
python pipelines/daily/prepare_daily_spectrograms.py \
  --mat-root /data/onc \
  --output-root /data/onc/prepared \
  --since-hours 24 --num-workers 8 --by-day \
  --args-pkl /path/to/model/args.pkl
```

### 4) Local GPU-side inference
File: `pipelines/local/infer_prepared_spectrograms.py`
- Reads VM manifest (`manifest_*.csv`), loads trained model, runs batched inference (optional AMP).
- Outputs:
  - Predictions CSV (one row per spectrogram): `npy_path, hydrophone, t0, t1, pred_index, pred_label, prob_<class>...`.
  - Optional dashboard outputs when `--dashboard-root` is set:
    - `DASH_ROOT/YYYY-MM-DD/HYDROPHONE/images/*.png` (rendered from `.npy`)
    - `DASH_ROOT/YYYY-MM-DD/HYDROPHONE/labels.json` (filename → list of labels)

Example:
```bash
python pipelines/local/infer_prepared_spectrograms.py \
  --manifest /mnt/whalestor/prepared/manifest_2025...csv \
  --model-dir /home/user/ONC/.../finetune/amba-base-... \
  --checkpoint /home/user/ONC/.../models/ft-avgtok_best_checkpoint.pth \
  --batch-size 32 --use-amp --device cuda \
  --out-csv /mnt/whalestor/inference/predictions_2025-09-25.csv \
  --dashboard-root /mnt/whalestor/dashboard \
  --save-png \
  --thresholds-file /mnt/whalestor/config/thresholds.json
```
Thresholds file example (`thresholds.json`):
```json
{
  "normal": 0.61,
  "Anomaly": 0.02,
  "Data Gap": 0.52,
  "Dropout": 0.09,
  "Engine Noise": 0.45,
  "Rain": 0.03,
  "Sensitivity": 0.03,
  "Tonal": 0.04
}
```
Label selection uses per-class thresholds when present; otherwise 0.5. Top-1 is included if none cross threshold.

### 5) VM starter scripts
- `pipelines/daily/run_daily_vm_pipeline.sh`: one-shot download + prepare.
- `pipelines/daily/start_daily_vm_pipeline.sh`: full, schedulable starter.
  - Starts background watcher (with `timeout` unless `--continuous`).
  - Kicks off last-24h download for `--all` or specific `--devices`.
  - Options: `--max-hours 24`, `--continuous`, `--start-at HH:MM` (local).

Examples:
```bash
# Single-day run; watcher stops after 24h
./pipelines/daily/start_daily_vm_pipeline.sh \
  --repo-root /home/onc/ONC/selfsupervision_anomalies_onc \
  --data-dir /data/onc \
  --prepared-dir /data/onc/prepared \
  --args-pkl /path/to/model/args.pkl \
  --all --max-hours 24

# Continuous daily schedule; begin at 04:00 each day
./pipelines/daily/start_daily_vm_pipeline.sh \
  --repo-root /home/onc/ONC/selfsupervision_anomalies_onc \
  --data-dir /data/onc \
  --prepared-dir /data/onc/prepared \
  --args-pkl /path/to/model/args.pkl \
  --all --max-hours 24 --continuous --start-at 04:00
```

---

## Directory Conventions

- Downloader:
```
DATA_DIR/
  ICLISTENHF1951/last24h/mat/processed/*.mat
  ICLISTENHF1354/last24h/mat/processed/*.mat
  ...
```

- Prepared (with `--by-day`):
```
PREPARED_DIR/
  2025-09-25/ICLISTENHF1951/*.npy
  2025-09-25/ICLISTENHF1354/*.npy
  manifest_20250925T040000Z.csv
```

- Dashboard outputs (local inference with `--dashboard-root`):
```
DASH_ROOT/
  2025-09-25/ICLISTENHF1951/
    images/
      ICLISTENHF1951_....png
    labels.json
  2025-09-25/ICLISTENHF1354/
    images/
      ICLISTENHF1354_....png
    labels.json
```

### labels.json schema
A flat map: `image_filename.png` → array of predicted labels.
Example:
```json
{
  "ICLISTENHF1951_20240902T175035.989Z_20240902T175535.989Z-OD-spect_plotRes.png": ["Engine Noise"],
  "ICLISTENHF1951_20240902T175535.989Z_20240902T180035.989Z-OD-spect_plotRes.png": ["normal", "Tonal"]
}
```

### Predictions CSV columns
`npy_path, hydrophone, t0, t1, pred_index, pred_label, prob_<class1>, prob_<class2>, ...`

### Manifest CSV columns (VM prep)
`npy_path, hydrophone, t0, t1, original_mat, fdim, tdim, pipeline, day`

---

## Preprocessing Summary (H5-like)
- Resize original spectrogram `[F,T]` to target size (default `[512,512]`).
- Normalize using dataset statistics `(x-mean)/(2*std)` when provided, else percentile-log-minmax.
- Save `.npy` as float32 `[F,T]`.

---

## Performance Tips
- VM prep: increase `--num-workers` for more parallelism.
- Local inference: `--use-amp` on CUDA reduces forward time.

---

## Dashboard Integration Checklist
1) Point the dashboard generator at `DASH_ROOT/YYYY-MM-DD/*/`.
2) For each hydro:
   - read `labels.json` (filename → label list)
   - display PNGs from `images/`.
3) Optionally merge the predictions CSV for analytics.
4) Keep filename conventions consistent so labeling app and dashboard align.

---

## Troubleshooting
- ONC metadata warnings are suppressed; spectrogram files still download.
- Devices without last-24h data are skipped.
- Watcher deletes `.mat` only after a successful `.npy` write.
- Pass `--args-pkl` wherever available to keep normalization consistent with training.
