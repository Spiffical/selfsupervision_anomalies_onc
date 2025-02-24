#!/bin/bash
# Load necessary modules
module load python/3.10

# Activate your virtual environment
source $HOME/ssamba/myenv/bin/activate

# Load environment variables from .env file
if [ -f ~/ssamba/.env ]; then
    export $(grep -v '^#' ~/ssamba/.env | xargs)
fi

# Get command line arguments
DATA_TRAIN_PATH=${1:-$SCRATCH/different_locations_incl_backgroundpipelinenormals_multilabel.h5}
PRETRAINED_MODEL_PATH=${2:-""}  # Path to pretrained model
TRAIN_RATIO=${3:-0.8}  # Default to 0.8 if not provided
WANDB_GROUP=${4:-"default_finetune"}  # Default group if not provided

set -x
export TORCH_HOME=../../pretrained_models
export PYTHONPATH=$PYTHONPATH:$HOME/ssamba
export PYTHONPATH=$PYTHONPATH:$SCRATCH/ssamba_project
export PYTHONPATH=$PYTHONPATH:$SLURM_TMPDIR/ssamba_project

# Create experiment directory in scratch
mkdir -p $SCRATCH/exp

# Dataset parameters
dataset=custom
dataset_mean=51.506817  # Set to "none" to calculate from dataset
dataset_std=13.638703   # Set to "none" to calculate from dataset
target_length=512
num_mel_bins=512

# Dataset split parameters
train_ratio=$TRAIN_RATIO
val_ratio=0.1
split_seed=42

# Training parameters
task=ft_cls  # finetuning task
mask_patch=0  # No masking during finetuning

# Model architecture (should match pretrained model)
model_size=small
patch_size=16
embed_dim=768
depth=24

# Patch parameters
fshape=16
tshape=16
fstride=16
tstride=16

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

# Finetuning hyperparameters
bal=none
batch_size=32  # Can be larger than pretraining since no masking
lr=5e-5  # Lower learning rate for finetuning
lr_patience=5  # More patience for finetuning
epoch=50
freqm=0
timem=0
mixup=0

# Experiment directory
exp_dir=$SCRATCH/exp/amba-ft-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-${task}-${dataset}-tr$(printf "%.1f" ${TRAIN_RATIO})-${WANDB_GROUP}

# Run the training script
python -W ignore src/run_amba_spectrogram.py --use_wandb --wandb_entity "spencer-bialek" \
--wandb_group ${WANDB_GROUP} \
--dataset ${dataset} \
--data-train "$DATA_TRAIN_PATH" \
--exp-dir $exp_dir \
--pretrained_model_path "$PRETRAINED_MODEL_PATH" \
--n_class 2 \
--train_ratio ${train_ratio} \
--val_ratio ${val_ratio} \
--split_seed ${split_seed} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
--target_length ${target_length} --num_mel_bins ${num_mel_bins} \
--model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
--task ${task} --lr_patience ${lr_patience} --epoch_iter 1 \
--patch_size ${patch_size} --embed_dim ${embed_dim} --depth ${depth} \
--rms_norm ${rms_norm} --residual_in_fp32 ${residual_in_fp32} \
--fused_add_norm ${fused_add_norm} --if_rope ${if_rope} --if_rope_residual ${if_rope_residual} \
--bimamba_type ${bimamba_type} --use_middle_cls_token ${use_middle_cls_token} \
--drop_path_rate ${drop_path_rate} --stride ${stride} --channels ${channels} \
--num_classes ${num_classes} --drop_rate ${drop_rate} --norm_epsilon ${norm_epsilon} \
--if_bidirectional ${if_bidirectional} --final_pool_type ${final_pool_type} \
--if_abs_pos_embed ${if_abs_pos_embed} --if_bimamba ${if_bimamba} \
--if_cls_token ${if_cls_token} --if_devide_out ${if_devide_out} \
--use_double_cls_token ${use_double_cls_token} --use_middle_cls_token ${use_middle_cls_token} \
--adaptschedule true --main_metric f2  # Use F2 score for model selection \
--dataset_mean ${dataset_mean} \
--dataset_std ${dataset_std} 