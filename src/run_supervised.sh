#!/bin/bash
# Load necessary modules
module load python/3.10

# Activate your virtual environment
source $HOME/ssamba/myenv/bin/activate

# Load environment variables from .env file
if [ -f ~/ssamba/.env ]; then
    export $(grep -v '^#' ~/ssamba/.env | xargs)
fi

# Initialize variables with defaults
PYTHON_SCRIPT="$SLURM_TMPDIR/ssamba_project/src/run_supervised.py"
DATA_TRAIN_PATH=""
WANDB_PROJECT="amba_spectrogram_supervised"
WANDB_GROUP="supervised_experiment"
TRAIN_RATIO=0.8
EXP_DIR="/exp"
WANDB_ENTITY=""
declare -a EXCLUDE_LABELS=()
DRY_RUN="false"
RESUME=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python-script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --dataset)
            DATA_TRAIN_PATH="$2"
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
if [ -z "$DATA_TRAIN_PATH" ]; then
    echo "Error: --dataset is required"
    exit 1
fi

# Set fixed parameters
batch_size=16
lr=1e-4
lr_patience=3
epoch=200
freqm=48
timem=192
mixup=0.0
bal=balanced
fstride=10
tstride=10
fshape=16
tshape=16

# Dataset parameters
dataset=custom
dataset_mean=51.506817  # Set to "none" to calculate from dataset
dataset_std=13.638703   # Set to "none" to calculate from dataset
target_length=512  # Your spectrogram time dimension
num_mel_bins=512  # Your spectrogram frequency dimension

# Dataset split parameters
train_ratio=$TRAIN_RATIO
val_ratio=0.1
split_seed=42

# Model architecture
model_size=base
embed_dim=256
depth=8
in_chans=1

# Modify experiment directory name to include excluded labels if any
exclude_labels_str=""
if (( ${#EXCLUDE_LABELS[@]} > 0 )); then
    # Join array elements with underscores, replacing spaces with underscores
    labels_joined=""
    for label in "${EXCLUDE_LABELS[@]}"; do
        if [ -z "$labels_joined" ]; then
            labels_joined="${label// /_}"
        else
            labels_joined="${labels_joined}_${label// /_}"
        fi
    done
    exclude_labels_str="-excl${labels_joined}"
fi

# Base experiment folder name
base_folder=amba-supervised-b${batch_size}-lr${lr}-${dataset}-tr$(printf "%.1f" ${TRAIN_RATIO})-${WANDB_GROUP}${exclude_labels_str}

echo "Base folder: $base_folder"

# Create experiment directory
exp_dir=${EXP_DIR}/supervised/${base_folder}
mkdir -p ${exp_dir}/models

# Construct the Python command
PYTHON_CMD="python -W ignore $PYTHON_SCRIPT --use_wandb --wandb_entity \"${WANDB_ENTITY:-spencer-bialek}\" \
--wandb_project ${WANDB_PROJECT} \
--wandb_group ${WANDB_GROUP} \
--dataset ${dataset} \
--data-train \"$DATA_TRAIN_PATH\" \
--exp-dir $exp_dir \
--dataset_mean ${dataset_mean} \
--dataset_std ${dataset_std} \
--train_ratio ${train_ratio} \
--val_ratio ${val_ratio} \
--split_seed ${split_seed} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
--target_length ${target_length} --num_mel_bins ${num_mel_bins} \
--model_size ${model_size} --n-print-steps 100 \
--embed_dim ${embed_dim} --depth ${depth} \
--in_chans ${in_chans}"

# Add resume argument if provided
if [ ! -z "$RESUME" ]; then
    PYTHON_CMD+=" --resume \"$RESUME\""
fi

# Add exclude labels as a single argument with multiple values
if [ ${#EXCLUDE_LABELS[@]} -gt 0 ]; then
    PYTHON_CMD+=" --exclude_labels"
    for label in "${EXCLUDE_LABELS[@]}"; do
        PYTHON_CMD+=" \"$label\""
    done
fi

# Print the command that would be executed
echo "Python command that will be executed:"
echo "$PYTHON_CMD"
echo

# If this is a dry run, exit here
if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run completed. Exiting without executing."
    exit 0
fi

# Set up environment variables
set -x
export TORCH_HOME=../../pretrained_models
export PYTHONPATH=$PYTHONPATH:$HOME/ssamba
export PYTHONPATH=$PYTHONPATH:$SCRATCH/ssamba_project/src
export PYTHONPATH=$PYTHONPATH:$SLURM_TMPDIR/ssamba_project/src

# Execute the Python command
eval "$PYTHON_CMD" 