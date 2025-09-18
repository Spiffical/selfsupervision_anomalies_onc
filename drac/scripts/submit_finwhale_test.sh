#!/bin/bash
#SBATCH --account=def-kmoran                    # DRAC project account
#SBATCH --job-name=finwhale_test                # Job name
#SBATCH --output=out/finwhale_test_%j.out       # Standard output log
#SBATCH --error=err/finwhale_test_%j.err        # Standard error log
#SBATCH --time=02:00:00                         # Max runtime (HH:MM:SS)
#SBATCH --gres=gpu:h100:1                      # GPU (optional for speed)
#SBATCH --cpus-per-task=4                       # CPU cores
#SBATCH --mem=16G                               # Memory per node

# Parameters
POS_DIR=""
NEG_DIR=""
TAR_PATH=""
CHECKPOINT=""
OUT_DIR=""
BATCH_SIZE=128
NUM_WORKERS=4
CROP_SIZE=96
MIN_DB=-80
MAX_DB=0
TRAIN_RATIO=0.8
VAL_RATIO=0.1
SEED=42
AUGMENT_TEST="false"
DEVICE="cuda"
PROJECT_PATH="$HOME/ssamba"
VENV_PATH="${VENV_PATH:-$HOME/ssamba/myenv}"
PNG_SCALE=3
PNG_CMAP="inferno"
PNG_PMIN=2
PNG_PMAX=98

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pos-dir) POS_DIR="$2"; shift 2 ;;
    --neg-dir) NEG_DIR="$2"; shift 2 ;;
    --tar-path) TAR_PATH="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --crop-size) CROP_SIZE="$2"; shift 2 ;;
    --min-db) MIN_DB="$2"; shift 2 ;;
    --max-db) MAX_DB="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --augment-test) AUGMENT_TEST="true"; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --project-path) PROJECT_PATH="$2"; shift 2 ;;
    --venv|--venv-path) VENV_PATH="$2"; shift 2 ;;
    --png-scale) PNG_SCALE="$2"; shift 2 ;;
    --png-cmap) PNG_CMAP="$2"; shift 2 ;;
    --png-pmin) PNG_PMIN="$2"; shift 2 ;;
    --png-pmax) PNG_PMAX="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" || -z "$OUT_DIR" ]]; then
  echo "Error: --checkpoint and --out-dir are required"
  exit 1
fi

mkdir -p out err

module load python/3.10
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Error: venv not found at $VENV_PATH/bin/activate"
  exit 2
fi
source "$VENV_PATH/bin/activate"

# Copy project to node-local
rsync -a --delete --exclude='.git' "$PROJECT_PATH/" "$SLURM_TMPDIR/ssamba_project/"

# Handle data sources: either TAR extraction or raw dirs
if [[ -n "$TAR_PATH" ]]; then
  mkdir -p "$SLURM_TMPDIR/finwhale_data"
  if [[ "$TAR_PATH" == *.tar.gz || "$TAR_PATH" == *.tgz ]]; then
    tar -xzf "$TAR_PATH" -C "$SLURM_TMPDIR/finwhale_data"
  elif [[ "$TAR_PATH" == *.tar ]]; then
    tar -xf "$TAR_PATH" -C "$SLURM_TMPDIR/finwhale_data"
  elif [[ "$TAR_PATH" == *.zip ]]; then
    unzip -q "$TAR_PATH" -d "$SLURM_TMPDIR/finwhale_data"
  else
    echo "Unsupported archive format: $TAR_PATH"; exit 1
  fi
  if [[ -d "$SLURM_TMPDIR/finwhale_data/mat_files" && -d "$SLURM_TMPDIR/finwhale_data/neg_mat_files" ]]; then
    POS_ARG="$SLURM_TMPDIR/finwhale_data/mat_files"
    NEG_ARG="$SLURM_TMPDIR/finwhale_data/neg_mat_files"
  else
    ROOT_SUBDIR=$(find "$SLURM_TMPDIR/finwhale_data" -maxdepth 2 -type d -name mat_files -print -quit)
    if [[ -n "$ROOT_SUBDIR" ]]; then
      POS_ARG="$ROOT_SUBDIR"
      CAND_NEG="$(dirname "$ROOT_SUBDIR")/neg_mat_files"
      [[ -d "$CAND_NEG" ]] || { echo "Missing neg_mat_files next to $ROOT_SUBDIR"; exit 1; }
      NEG_ARG="$CAND_NEG"
    else
      echo "Could not locate mat_files/neg_mat_files in archive"; exit 1
    fi
  fi
else
  [[ -n "$POS_DIR" && -n "$NEG_DIR" ]] || { echo "Provide --pos-dir and --neg-dir or --tar-path"; exit 1; }
  POS_ARG="$POS_DIR"
  NEG_ARG="$NEG_DIR"
fi

# Ensure output directory
RUN_OUT_DIR="$OUT_DIR/finwhale_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_OUT_DIR"

export PYTHONPATH="$PYTHONPATH:$SLURM_TMPDIR/ssamba_project/src"
cd "$SLURM_TMPDIR/ssamba_project"

CMD=(
  python -u scripts/test_cnn.py \
    --pos-dir "$POS_ARG" --neg-dir "$NEG_ARG" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$RUN_OUT_DIR" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --crop-size "$CROP_SIZE" --min-db "$MIN_DB" --max-db "$MAX_DB" \
    --train-ratio "$TRAIN_RATIO" --val-ratio "$VAL_RATIO" \
    --seed "$SEED" --device "$DEVICE" \
    --png-scale "$PNG_SCALE" --png-cmap "$PNG_CMAP" --png-pmin "$PNG_PMIN" --png-pmax "$PNG_PMAX"
)
if [[ "$AUGMENT_TEST" == "true" ]]; then
  CMD+=( --augment-test )
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
