#!/usr/bin/env bash
set -euo pipefail

# Daily VM pipeline:
# 1) Download last 24h plotRes MAT for all active devices
# 2) Prepare (resize -> normalize) .npy spectrograms with optional stats from args.pkl
#
# Usage:
#   ./pipelines/daily/run_daily_vm_pipeline.sh \
#     --repo-root /home/onc/ONC/selfsupervision_anomalies_onc \
#     --data-dir /data/onc \
#     --prepared-dir /data/onc/prepared \
#     [--args-pkl /path/to/model/args.pkl]

REPO_ROOT=""
DATA_DIR=""
PREPARED_DIR=""
ARGS_PKL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2;;
    --data-dir) DATA_DIR="$2"; shift 2;;
    --prepared-dir) PREPARED_DIR="$2"; shift 2;;
    --args-pkl) ARGS_PKL="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$REPO_ROOT" || -z "$DATA_DIR" || -z "$PREPARED_DIR" ]]; then
  echo "Missing required args. See header for usage." >&2
  exit 1
fi

cd "$REPO_ROOT"

# 1) Download last 24h MATs (pre-check devices so we avoid empty folders)
python pipelines/daily/download_last24h_plotres.py \
  --data-dir "$DATA_DIR" \
  --all

# 2) Prepare recent .mat to .npy under PREPARED_DIR with parallel workers
PREP_ARGS=(
  --mat-root "$DATA_DIR" \
  --output-root "$PREPARED_DIR" \
  --since-hours 24 \
  --num-workers "8"
)
if [[ -n "$ARGS_PKL" ]]; then
  PREP_ARGS+=(--args-pkl "$ARGS_PKL")
fi

python pipelines/daily/prepare_daily_spectrograms.py "${PREP_ARGS[@]}"

echo "Daily VM pipeline complete. Prepared files and manifest are in: $PREPARED_DIR"


