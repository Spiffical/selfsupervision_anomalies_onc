import os
import argparse
import torch
from utilities.wandb_utils import init_wandb, finish_run
import dataloader
import numpy as np
from traintest_mask import trainmask
from traintest import train
import sys

# Import AMBAModel after setting up our own argument parser to avoid conflicts
def import_model():
    from run_amba_spectrogram import AMBAModel
    return AMBAModel

def setup_test_args():
    """Create a minimal set of arguments for testing with a small model."""
    args = argparse.Namespace()
    
    # Dataset parameters
    args.data_train = None  # Will be set from command line
    args.train_ratio = 0.8
    args.val_ratio = 0.1
    args.split_seed = 42
    args.dataset = "custom"
    args.num_mel_bins = 128  # Reduced from 512
    args.target_length = 128  # Reduced from 512
    
    # Model parameters - significantly smaller
    args.patch_size = 8  # Smaller patches
    args.embed_dim = 96  # Reduced from 768
    args.depth = 4  # Reduced from 24
    args.fshape = 8  # Reduced from 16
    args.tshape = 8  # Reduced from 16
    args.fstride = 8  # Reduced from 16
    args.tstride = 8  # Reduced from 16
    
    # Training parameters
    args.batch_size = 8  # Reduced from 12
    args.num_workers = 2  # Reduced from 4
    args.n_epochs = 3  # Reduced from 5
    args.lr = 1e-4
    args.warmup = True
    args.n_print_steps = 20  # More frequent printing
    args.epoch_iter = 0.5  # Save every half epoch
    args.lr_patience = 2
    args.adaptschedule = True
    args.mask_patch = 50  # Reduced from 300
    args.optim = "adam"
    args.save_model = True
    args.freqm = 0
    args.timem = 0
    args.mixup = 0
    args.bal = None
    
    # Model architecture parameters - simplified
    args.rms_norm = False
    args.residual_in_fp32 = False
    args.fused_add_norm = False
    args.if_rope = False
    args.if_rope_residual = False
    args.if_bidirectional = True
    args.final_pool_type = 'none'
    args.if_abs_pos_embed = True
    args.if_bimamba = False
    args.if_cls_token = True
    args.if_devide_out = True
    args.use_double_cls_token = False
    args.use_middle_cls_token = False
    args.drop_path_rate = 0.1
    args.stride = 8  # Reduced from 16
    args.channels = 1
    args.num_classes = 10  # Reduced from 1000
    args.drop_rate = 0.0
    args.norm_epsilon = 1e-5
    args.bimamba_type = "v2"
    
    return args

def setup_tiny_model_args():
    """Create an extremely small model configuration for laptop testing."""
    args = setup_test_args()
    
    # Even smaller model configuration
    args.num_mel_bins = 64
    args.target_length = 64
    args.patch_size = 4
    args.embed_dim = 32
    args.depth = 2
    args.fshape = 4
    args.tshape = 4
    args.fstride = 4
    args.tstride = 4
    args.batch_size = 4
    args.mask_patch = 20
    args.stride = 4
    
    return args

def test_training(data_path, exp_dir, task='pretrain_joint', use_wandb=True, tiny_model=False):
    """
    Run a test training loop with the specified configuration.
    
    Args:
        data_path: Path to the test h5 dataset
        exp_dir: Directory to save experiment results
        task: Training task (pretrain_mpc, pretrain_mpg, pretrain_joint, or ft_cls)
        use_wandb: Whether to log to wandb
        tiny_model: Whether to use an extremely small model for laptop testing
    """
    # Import the model here to avoid argument parser conflicts
    AMBAModel = import_model()
    
    # Create experiment directory
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    
    # Setup arguments
    args = setup_tiny_model_args() if tiny_model else setup_test_args()
    args.data_train = data_path
    args.exp_dir = exp_dir
    args.task = task
    args.use_wandb = use_wandb
    
    if use_wandb:
        run = init_wandb(
            args,
            project_name="ssamba_test",
            group="testing"
        )
        # Set run name manually if needed
        if run:
            model_size = "tiny" if tiny_model else "small"
            run.name = f"test_{model_size}_{task}"
            run.save()
    
    # Calculate dataset statistics
    print("Calculating dataset statistics...")
    mean, std = dataloader.calculate_dataset_stats(args.data_train)
    print(f"Dataset mean: {mean:.6f}")
    print(f"Dataset std: {std:.6f}")
    args.dataset_mean = mean
    args.dataset_std = std
    
    # Create data loaders
    train_dataset = dataloader.HDF5Dataset(
        h5_path=args.data_train,
        split='train',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
        target_length=args.target_length,
        num_mel_bins=args.num_mel_bins,
        freqm=0,
        timem=0,
        dataset_mean=args.dataset_mean,
        dataset_std=args.dataset_std,
        mixup=0
    )

    val_dataset = dataloader.HDF5Dataset(
        h5_path=args.data_train,
        split='val',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
        target_length=args.target_length,
        num_mel_bins=args.num_mel_bins,
        freqm=0,
        timem=0,
        dataset_mean=args.dataset_mean,
        dataset_std=args.dataset_std,
        mixup=0
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset splits:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    # Print model size information
    model_size = "tiny" if tiny_model else "small"
    print(f"\nUsing {model_size} model configuration:")
    print(f"Embedding dimension: {args.embed_dim}")
    print(f"Depth (layers): {args.depth}")
    print(f"Patch size: {args.patch_size}")
    print(f"Input dimensions: {args.num_mel_bins}x{args.target_length}")
    
    # Initialize model
    vision_mamba_config = {
        'img_size': (args.num_mel_bins, args.target_length),
        'patch_size': args.patch_size,
        'stride': args.stride,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'channels': args.channels,
        'num_classes': args.num_classes,
        'drop_rate': args.drop_rate,
        'drop_path_rate': args.drop_path_rate,
        'norm_epsilon': args.norm_epsilon,
        'rms_norm': args.rms_norm,
        'residual_in_fp32': args.residual_in_fp32,
        'fused_add_norm': args.fused_add_norm,
        'if_rope': args.if_rope,
        'if_rope_residual': args.if_rope_residual,
        'if_bidirectional': args.if_bidirectional,
        'final_pool_type': args.final_pool_type,
        'if_abs_pos_embed': args.if_abs_pos_embed,
        'if_bimamba': args.if_bimamba,
        'if_cls_token': args.if_cls_token,
        'if_devide_out': args.if_devide_out,
        'use_double_cls_token': args.use_double_cls_token,
        'use_middle_cls_token': args.use_middle_cls_token,
        'bimamba_type': args.bimamba_type
    }
    
    audio_model = AMBAModel(
        fshape=args.fshape, tshape=args.tshape,
        fstride=args.fstride, tstride=args.tstride,
        input_fdim=args.num_mel_bins,
        input_tdim=args.target_length,
        model_size='base',
        pretrain_stage=True if 'pretrain' in task else False,
        vision_mamba_config=vision_mamba_config
    )
    
    # Calculate and print approximate model size
    model_parameters = sum(p.numel() for p in audio_model.parameters())
    print(f"Approximate model parameters: {model_parameters:,}")
    model_size_mb = model_parameters * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)
    print(f"Approximate model size: {model_size_mb:.2f} MB")
    
    # Start training
    if 'pretrain' in task:
        print(f'Starting test pretraining for {args.n_epochs} epochs')
        trainmask(audio_model, train_loader, val_loader, args)
    else:
        print(f'Starting test fine-tuning for {args.n_epochs} epochs')
        train(audio_model, train_loader, val_loader, args)
    
    if use_wandb:
        finish_run()

if __name__ == "__main__":
    # Create our own argument parser
    parser = argparse.ArgumentParser(description='Test AMBA training pipeline with small models')
    parser.add_argument('data_path', type=str, help='Path to the test h5 dataset')
    parser.add_argument('--exp_dir', type=str, default='test_exp',
                      help='Directory to save experiment results')
    parser.add_argument('--task', type=str, default='pretrain_joint',
                      choices=['pretrain_mpc', 'pretrain_mpg', 'pretrain_joint', 'ft_cls'],
                      help='Training task')
    parser.add_argument('--no_wandb', action='store_true',
                      help='Disable wandb logging')
    parser.add_argument('--tiny', action='store_true',
                      help='Use an extremely small model for laptop testing')
    
    # Parse arguments before importing any modules that might define their own argument parser
    args = parser.parse_args()
    
    # Run the test training
    test_training(args.data_path, args.exp_dir, args.task, not args.no_wandb, args.tiny) 