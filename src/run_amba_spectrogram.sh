#!/bin/bash
# Load necessary modules
module load python/3.10

# Activate your virtual environment
source ~/ssamba/myenv/bin/activate
set -x
export TORCH_HOME=../../pretrained_models
export PYTHONPATH=$PYTHONPATH:/home/merileo/ssamba

mkdir -p exp

# Dataset parameters
dataset=custom
dataset_mean=0  # You should calculate this from your dataset
dataset_std=1   # You should calculate this from your dataset
target_length=512  # Your spectrogram time dimension
num_mel_bins=512  # Your spectrogram frequency dimension

# Dataset split parameters
train_ratio=0.8
val_ratio=0.1
split_seed=42

# Training parameters
task=pretrain_joint  # or ft_cls for fine-tuning
mask_patch=300  # Number of patches to mask during pretraining

# Model architecture
model_size=small
patch_size=16
embed_dim=768
depth=24

# Patch parameters - no overlap in pretraining
fshape=16
tshape=16
fstride=${fshape}
tstride=${tshape}

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
num_classes=1000
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

# Training hyperparameters
bal=none
batch_size=6
lr=1e-4
lr_patience=2
epoch=30
freqm=0
timem=0
mixup=0

# Experiment directory
exp_dir=./exp/amba-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}

# Run the training script
python -W ignore /home/merileo/ssamba/src/run_amba_spectrogram.py --use_wandb --wandb_entity "spencer-bialek" \
--dataset ${dataset} \
--data-train /scratch/merileo/different_locations_incl_backgroundpipelinenormals_multilabel.h5 \
--exp-dir $exp_dir \
--n_class 2 \
--train_ratio ${train_ratio} \
--val_ratio ${val_ratio} \
--split_seed ${split_seed} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
--target_length ${target_length} --num_mel_bins ${num_mel_bins} \
--model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
--task ${task} --lr_patience ${lr_patience} --epoch_iter 4000 \
--patch_size ${patch_size} --embed_dim ${embed_dim} --depth ${depth} \
--rms_norm ${rms_norm} --residual_in_fp32 ${residual_in_fp32} \
--fused_add_norm ${fused_add_norm} --if_rope ${if_rope} --if_rope_residual ${if_rope_residual} \
--bimamba_type ${bimamba_type} --use_middle_cls_token ${use_middle_cls_token} \
--drop_path_rate ${drop_path_rate} --stride ${stride} --channels ${channels} \
--num_classes ${num_classes} --drop_rate ${drop_rate} --norm_epsilon ${norm_epsilon} \
--if_bidirectional ${if_bidirectional} --final_pool_type ${final_pool_type} \
--if_abs_pos_embed ${if_abs_pos_embed} --if_bimamba ${if_bimamba} \
--if_cls_token ${if_cls_token} --if_devide_out ${if_devide_out} \
--use_double_cls_token ${use_double_cls_token} --use_middle_cls_token ${use_middle_cls_token} 