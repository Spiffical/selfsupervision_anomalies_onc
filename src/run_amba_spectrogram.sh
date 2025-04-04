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
PYTHON_SCRIPT=""
DATA_TRAIN_PATH=""
WANDB_PROJECT="amba_spectrogram"
WANDB_GROUP="default_experiment"
TRAIN_RATIO=0.8
RESUME="true"  # Default is to resume training if a checkpoint exists
EXP_DIR="/exp"
TASK="pretrain_joint"
WANDB_ENTITY=""
declare -a EXCLUDE_LABELS=()
PRETRAINED_PATH=""
DRY_RUN="false"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python-script)
            PYTHON_SCRIPT="$2"
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
        --resume)
            # Only set RESUME to false if explicitly specified as false/False/FALSE
            if [[ "${2,,}" == "false" ]]; then
                RESUME="false"
            fi
            shift 2
            ;;
        --exp-dir)
            EXP_DIR="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
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
if [ -z "$PYTHON_SCRIPT" ]; then
    echo "Error: --python-script is required"
    exit 1
fi
if [ -z "$DATA_TRAIN_PATH" ]; then
    echo "Error: --dataset is required"
    exit 1
fi

# Set fixed parameters for experiment folder name (always use pretraining values)
folder_mask_patch=300
folder_batch_size=16
folder_lr=1e-4
folder_fstride=16
folder_tstride=16

# Print out the excluded labels
echo "Excluded labels: ${EXCLUDE_LABELS[*]}"

# Set task-specific parameters for actual training
if [[ $TASK == *"pretrain"* ]]; then
    # Pretraining parameters
    mask_patch=300  # Number of patches to mask during pretraining
    batch_size=16
    lr=1e-4
    lr_patience=2
    epoch=200
    freqm=0
    timem=0
    mixup=0
    bal=none
    fstride=16  # No overlap in pretraining
    tstride=16  # No overlap in pretraining
else
    # Finetuning parameters
    mask_patch=0  # No masking in finetuning
    batch_size=16
    lr=5e-5
    lr_patience=3
    epoch=200
    freqm=48
    timem=192
    mixup=0.5
    bal=balanced
    fstride=10  # Use overlap in finetuning
    tstride=10  # Use overlap in finetuning
fi

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
patch_size=16
embed_dim=768
depth=24

# Patch parameters
fshape=16
tshape=16

# Model configuration
rms_norm='false'
residual_in_fp32='false'
fused_add_norm='false'
if_rope='false'
if_rope_residual='false'
bimamba_type="v2"
drop_path_rate=0.1
stride=16
channels=1
num_classes=2
drop_rate=0.
norm_epsilon=1e-5
if_bidirectional='true'
final_pool_type='none'
if_abs_pos_embed='true'
if_bimamba='false'
if_cls_token='true'
if_devide_out='true'
use_double_cls_token='false'
use_middle_cls_token='false'

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

# Base experiment folder name - use pretraining parameters for consistent naming
base_folder=amba-${model_size}-f${fshape}-t${tshape}-b${folder_batch_size}-lr${folder_lr}-m${folder_mask_patch}-${dataset}-tr$(printf "%.1f" ${TRAIN_RATIO})-${WANDB_GROUP}${exclude_labels_str}

echo "Base folder: $base_folder"

# Create separate directories for pretraining and finetuning
if [[ $TASK == *"pretrain"* ]]; then
    # For pretraining, save in pretrain directory
    exp_dir=${EXP_DIR}/pretrain/${base_folder}
else
    # For finetuning, save in finetune directory
    exp_dir=${EXP_DIR}/finetune/${base_folder}
fi

# Create directories
mkdir -p ${exp_dir}/models

# Construct the Python command that would be executed
PYTHON_CMD="python -W ignore $PYTHON_SCRIPT --use_wandb --wandb_entity \"${WANDB_ENTITY:-spencer-bialek}\" \
--wandb_project ${WANDB_PROJECT} \
--wandb_group ${WANDB_GROUP} \
--dataset ${dataset} \
--data-train \"$DATA_TRAIN_PATH\" \
--exp-dir $exp_dir \
$([ ! -z "$PRETRAINED_PATH" ] && echo "--pretrained_path $PRETRAINED_PATH") \
--dataset_mean ${dataset_mean} \
--dataset_std ${dataset_std} \
--n_class 2 \
--train_ratio ${train_ratio} \
--val_ratio ${val_ratio} \
--split_seed ${split_seed} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
--target_length ${target_length} --num_mel_bins ${num_mel_bins} \
--model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
--task ${TASK} --lr_patience ${lr_patience} --epoch_iter 1 \
--patch_size ${patch_size} --embed_dim ${embed_dim} --depth ${depth} \
--rms_norm ${rms_norm} --residual_in_fp32 ${residual_in_fp32} \
--fused_add_norm ${fused_add_norm} --if_rope ${if_rope} --if_rope_residual ${if_rope_residual} \
--bimamba_type ${bimamba_type} --use_middle_cls_token ${use_middle_cls_token} \
--drop_path_rate ${drop_path_rate} --stride ${stride} --channels ${channels} \
--num_classes ${num_classes} --drop_rate ${drop_rate} --norm_epsilon ${norm_epsilon} \
--if_bidirectional ${if_bidirectional} --final_pool_type ${final_pool_type} \
--if_abs_pos_embed ${if_abs_pos_embed} --if_bimamba ${if_bimamba} \
--if_cls_token ${if_cls_token} --if_devide_out ${if_devide_out} \
--use_double_cls_token ${use_double_cls_token} --use_middle_cls_token ${use_middle_cls_token}"

# Add exclude labels as a single argument with multiple values
if [ ${#EXCLUDE_LABELS[@]} -gt 0 ]; then
    PYTHON_CMD+=" --exclude_labels"
    for label in "${EXCLUDE_LABELS[@]}"; do
        PYTHON_CMD+=" \"$label\""
    done
fi

# Add resume flag if needed (default behavior is to resume)
if [ "$RESUME" != "false" ]; then
    PYTHON_CMD+=" --resume"
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