#!/bin/bash
#SBATCH --account=def-kmoran                   # DRAC project account
#SBATCH --job-name=amba_spectrogram            # Job name
#SBATCH --output=out/amba_spectrogram_%j.out   # Standard output log (%j adds job ID)
#SBATCH --error=err/amba_spectrogram_%j.err    # Standard error log
#SBATCH --time=08:00:00                        # Maximum runtime (HH:MM:SS)
#SBATCH --gres=gpu:v100l:1                     # Request 1 V100 GPU (or P100 if needed)
#SBATCH --cpus-per-task=4                      # Number of CPU cores
#SBATCH --mem=32G                              # Memory per node

# Initialize variables with defaults
TRAINING_DATA_PATH=""
WANDB_PROJECT="amba_spectrogram"
WANDB_GROUP="default_experiment"
TRAIN_RATIO=0.8
PROJECT_PATH="$HOME/ssamba"
RESUME="true"
TASK="pretrain_joint"
EXP_DIR="/exp"
WANDB_ENTITY=""
PRETRAINED_PATH=""
declare -a EXCLUDE_LABELS=()
DRY_RUN="false"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            TRAINING_DATA_PATH="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-group)
            WANDB_GROUP="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --project-path)
            PROJECT_PATH="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --exp-dir)
            EXP_DIR="$2"
            shift 2
            ;;
        --wandb-entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --exclude-label)
            EXCLUDE_LABELS+=("$2")
            shift 2
            ;;
        --pretrained-path)
            PRETRAINED_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$TRAINING_DATA_PATH" ]; then
    echo "Error: --dataset is required"
    exit 1
fi

# Print out the excluded labels
echo "Excluded labels: ${EXCLUDE_LABELS[*]}"

# Print out the operations that would be performed
echo -e "\nOperations that would be performed:"
echo "1. Loading Python 3.10"
echo "2. Activating virtual environment: $HOME/ssamba/myenv/bin/activate"
echo "3. Loading W&B API key from: $PROJECT_PATH/.env (if exists)"
echo "4. Copying training data:"
echo "   From: $TRAINING_DATA_PATH"
echo "   To: \$SLURM_TMPDIR/$(basename "$TRAINING_DATA_PATH")"
echo "5. Copying project files:"
echo "   From: $PROJECT_PATH"
echo "   To: \$SLURM_TMPDIR/ssamba_project"

# Extract the filename from the training data path
TRAINING_DATA_FILENAME=$(basename "$TRAINING_DATA_PATH")

# Construct the run_amba_spectrogram.sh command
RUN_CMD="$SLURM_TMPDIR/ssamba_project/src/run_amba_spectrogram.sh \
    --python-script \"\$SLURM_TMPDIR/ssamba_project/src/run_amba_spectrogram.py\" \
    --dataset \"\$SLURM_TMPDIR/$TRAINING_DATA_FILENAME\" \
    --wandb-project \"$WANDB_PROJECT\" \
    --wandb-group \"$WANDB_GROUP\" \
    --train-ratio \"$TRAIN_RATIO\" \
    --resume \"$RESUME\" \
    --exp-dir \"$EXP_DIR\" \
    --task \"$TASK\""

# Add optional arguments to command
if [ ! -z "$WANDB_ENTITY" ]; then
    RUN_CMD+=" --wandb-entity \"$WANDB_ENTITY\""
fi

# Add exclude labels as separate arguments
for label in "${EXCLUDE_LABELS[@]}"; do
    RUN_CMD+=" --exclude-label \"$label\""
done

if [ ! -z "$PRETRAINED_PATH" ]; then
    RUN_CMD+=" --pretrained-path \"$PRETRAINED_PATH\""
fi

if [ "$DRY_RUN" = "true" ]; then
    RUN_CMD+=" --dry-run"
fi

# Print the final command that would be executed
echo -e "\nFinal command that would be executed:"
echo "$RUN_CMD"
echo

# If this is a dry run, exit here
if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run completed. Exiting without executing."
    exit 0
fi

# Load required modules
module load python/3.10

# Activate your virtual environment
source $HOME/ssamba/myenv/bin/activate

# Load W&B API key from .env file if needed
if [ -f $PROJECT_PATH/.env ]; then
    export $(grep -v '^#' $PROJECT_PATH/.env | xargs)
fi

# Copy training data to SLURM temporary directory while preserving filename
echo "Copying training data to temporary directory..."
cp "$TRAINING_DATA_PATH" "$SLURM_TMPDIR/$TRAINING_DATA_FILENAME"

# Copy project files to SLURM temporary directory
echo "Copying project files to temporary directory..."
cp -ru "$PROJECT_PATH" "$SLURM_TMPDIR/ssamba_project"

# Navigate to the copied project directory
cd "$SLURM_TMPDIR/ssamba_project"

# Execute the command
eval "$RUN_CMD"
