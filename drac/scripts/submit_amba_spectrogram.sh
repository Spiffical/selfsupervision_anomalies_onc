#!/bin/bash
#SBATCH --account=def-kmoran                   # DRAC project account
#SBATCH --job-name=amba_spectrogram            # Job name
#SBATCH --output=out/amba_spectrogram_%j.out   # Standard output log
#SBATCH --error=err/amba_spectrogram_%j.err    # Standard error log
#SBATCH --time=08:00:00                        # Max runtime (HH:MM:SS)
#SBATCH --cpus-per-task=4                      # CPU cores
#SBATCH --mem=32G                              # Memory per node
# (GPU flags are intentionally omitted; wrapper adds them)

# ---------- your script (unchanged except venv support) ----------
TRAINING_DATA_PATH=""
WANDB_PROJECT="amba_spectrogram"
WANDB_GROUP="default_experiment"
TRAIN_RATIO=0.8
PROJECT_PATH="$HOME/selfsupervision_anomalies_onc"
RESUME="true"
TASK="pretrain_joint"
EXP_DIR="/exp"
WANDB_ENTITY=""
PRETRAINED_PATH=""
declare -a EXCLUDE_LABELS=()
DRY_RUN="false"
MULTICLASS="false"
NUM_CLASSES=2

# Virtualenv (can be overridden via --venv or --venv-path, or env VENV_PATH)
VENV_PATH="${VENV_PATH:-$HOME/selfsupervision_anomalies_onc/myenv}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) TRAINING_DATA_PATH="$2"; shift 2;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2;;
        --wandb-group) WANDB_GROUP="$2"; shift 2;;
        --train-ratio) TRAIN_RATIO="$2"; shift 2;;
        --project-path) PROJECT_PATH="$2"; shift 2;;
        --resume) RESUME="$2"; shift 2;;
        --task) TASK="$2"; shift 2;;
        --exp-dir) EXP_DIR="$2"; shift 2;;
        --wandb-entity) WANDB_ENTITY="$2"; shift 2;;
        --exclude-label) EXCLUDE_LABELS+=("$2"); shift 2;;
        --pretrained-path) PRETRAINED_PATH="$2"; shift 2;;
        --dry-run) DRY_RUN="true"; shift;;
        --multiclass) MULTICLASS="true"; NUM_CLASSES="$2"; shift 2;;
        --venv|--venv-path) VENV_PATH="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [ -z "$TRAINING_DATA_PATH" ]; then
    echo "Error: --dataset is required"
    exit 1
fi

echo "Excluded labels: ${EXCLUDE_LABELS[*]}"
echo -e "\nOperations that would be performed:"
echo "1. Loading Python 3.10"
echo "2. Activating virtual environment: $VENV_PATH/bin/activate"
echo "3. Loading W&B API key from: $PROJECT_PATH/.env (if exists)"
echo "4. Copying training data:"
echo "   From: $TRAINING_DATA_PATH"
echo "   To: \$SLURM_TMPDIR/$(basename "$TRAINING_DATA_PATH")"
echo "5. Copying project files:"
echo "   From: $PROJECT_PATH"
echo "   To: \$SLURM_TMPDIR/ssamba_project"

TRAINING_DATA_FILENAME=$(basename "$TRAINING_DATA_PATH")

RUN_CMD="$SLURM_TMPDIR/ssamba_project/src/run_amba_spectrogram.sh \
    --python-script \"\$SLURM_TMPDIR/ssamba_project/src/run_amba_spectrogram.py\" \
    --dataset \"\$SLURM_TMPDIR/$TRAINING_DATA_FILENAME\" \
    --wandb-project \"$WANDB_PROJECT\" \
    --wandb-group \"$WANDB_GROUP\" \
    --train-ratio \"$TRAIN_RATIO\" \
    --resume \"$RESUME\" \
    --exp-dir \"$EXP_DIR\" \
    --task \"$TASK\""

[ -n "$WANDB_ENTITY" ]     && RUN_CMD+=" --wandb-entity \"$WANDB_ENTITY\""
for label in "${EXCLUDE_LABELS[@]}"; do RUN_CMD+=" --exclude-label \"$label\""; done
[ -n "$PRETRAINED_PATH" ]  && RUN_CMD+=" --pretrained-path \"$PRETRAINED_PATH\""
[ "$MULTICLASS" = "true" ] && RUN_CMD+=" --multiclass \"$NUM_CLASSES\""
[ "$DRY_RUN" = "true" ]    && RUN_CMD+=" --dry-run"

echo -e "\nFinal command that would be executed:\n$RUN_CMD\n"
[ "$DRY_RUN" = "true" ] && { echo "Dry run completed. Exiting."; exit 0; }

module load python/3.10

# Activate chosen virtual environment
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Error: venv not found at $VENV_PATH/bin/activate"
    exit 2
fi
source "$VENV_PATH/bin/activate"

# Load W&B API key if present
if [ -f "$PROJECT_PATH/.env" ]; then
    export $(grep -v '^#' "$PROJECT_PATH/.env" | xargs)
fi

echo "Copying training data to temporary directory..."
cp "$TRAINING_DATA_PATH" "$SLURM_TMPDIR/$TRAINING_DATA_FILENAME"

echo "Copying project files to temporary directory..."
cp -ru "$PROJECT_PATH" "$SLURM_TMPDIR/ssamba_project"

cd "$SLURM_TMPDIR/ssamba_project"
eval "$RUN_CMD"
