#!/usr/bin/env bash
set -euo pipefail

# Start the full daily VM pipeline:
# 1) Launch a background watcher that prepares new .mat files to .npy and deletes raws
# 2) Kick off a last-24h download for either --all devices or a provided list
#
# Lifetime control:
#  - Default: max runtime 24h (watcher is killed after --max-hours)
#  - --continuous: keep watcher running and repeat download every --max-hours (e.g., 24h)
#  - --start-at HH:MM (24h) waits until the next occurrence (today or tomorrow) before starting
#
# Usage:
#   ./pipelines/daily/start_daily_vm_pipeline.sh \
#     --repo-root /home/onc/ONC/selfsupervision_anomalies_onc \
#     --data-dir /data/onc \
#     --prepared-dir /data/onc/prepared \
#     [--args-pkl /path/to/model/args.pkl] \
#     [--devices ICLISTENHF1951,ICLISTENHF1354 | --all]

REPO_ROOT=""
DATA_DIR=""
PREPARED_DIR=""
ARGS_PKL=""
DEVICES=""
USE_ALL=false
WATCH_WORKERS=8
WATCH_BATCH=64
WATCH_SCAN=10
MAX_HOURS=24
CONTINUOUS=false
START_AT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2;;
    --data-dir) DATA_DIR="$2"; shift 2;;
    --prepared-dir) PREPARED_DIR="$2"; shift 2;;
    --args-pkl) ARGS_PKL="$2"; shift 2;;
    --devices) DEVICES="$2"; shift 2;;
    --all) USE_ALL=true; shift 1;;
    --watch-workers) WATCH_WORKERS="$2"; shift 2;;
    --watch-batch) WATCH_BATCH="$2"; shift 2;;
    --watch-scan) WATCH_SCAN="$2"; shift 2;;
    --max-hours) MAX_HOURS="$2"; shift 2;;
    --continuous) CONTINUOUS=true; shift 1;;
    --start-at) START_AT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$REPO_ROOT" || -z "$DATA_DIR" || -z "$PREPARED_DIR" ]]; then
  echo "Missing required args. See header for usage." >&2
  exit 1
fi

cd "$REPO_ROOT"

# Optional delayed start until next HH:MM
if [[ -n "$START_AT" ]]; then
  # Compute seconds until next occurrence of START_AT in local time
  NOW_EPOCH=$(date +%s)
  TARGET_TODAY=$(date -d "today $START_AT" +%s 2>/dev/null || true)
  if [[ -z "$TARGET_TODAY" ]]; then
    echo "Invalid --start-at format. Use HH:MM (24h)." >&2
    exit 1
  fi
  if (( TARGET_TODAY <= NOW_EPOCH )); then
    TARGET=$(date -d "tomorrow $START_AT" +%s)
  else
    TARGET=$TARGET_TODAY
  fi
  SLEEP_SECS=$(( TARGET - NOW_EPOCH ))
  echo "Scheduling start at $START_AT (sleeping ${SLEEP_SECS}s) ..."
  sleep "$SLEEP_SECS"
fi

echo "Starting watcher: mat-root=$DATA_DIR prepared=$PREPARED_DIR (max-hours=$MAX_HOURS continuous=$CONTINUOUS)"
WATCH_ARGS=(
  --mat-root "$DATA_DIR" \
  --output-root "$PREPARED_DIR" \
  --num-workers "$WATCH_WORKERS" \
  --batch-size "$WATCH_BATCH" \
  --scan-interval "$WATCH_SCAN" \
  --no-exit
)
if [[ -n "$ARGS_PKL" ]]; then
  WATCH_ARGS+=(--args-pkl "$ARGS_PKL")
fi

# Launch watcher with timeout unless running continuous
if $CONTINUOUS; then
  python pipelines/daily/watch_prepare_mats.py "${WATCH_ARGS[@]}" &
else
  # Use GNU timeout to enforce max lifetime
  timeout "${MAX_HOURS}h" python pipelines/daily/watch_prepare_mats.py "${WATCH_ARGS[@]}" &
fi
WATCH_PID=$!
echo "Watcher PID: $WATCH_PID"

run_download() {
  echo "Kicking off last-24h download..."
  DL_ARGS=( --data-dir "$DATA_DIR" )
  if $USE_ALL; then
    DL_ARGS+=( --all )
  elif [[ -n "$DEVICES" ]]; then
    DL_ARGS+=( --devices "$DEVICES" )
  else
    echo "Either --all or --devices must be provided" >&2
    kill "$WATCH_PID" || true
    exit 1
  fi
  python pipelines/daily/download_last24h_plotres.py "${DL_ARGS[@]}"
}

if $CONTINUOUS; then
  # Repeat forever at MAX_HOURS cadence
  while true; do
    run_download
    echo "Sleeping ${MAX_HOURS}h before next cycle..."
    sleep "${MAX_HOURS}h"
  done
else
  # Single-shot download and exit; watcher auto-terminates via timeout
  run_download
  echo "Download launched. Watcher will stop after ${MAX_HOURS}h or when you kill PID $WATCH_PID"
fi


