#!/usr/bin/env python3

print("Starting imports...")
import sys
import time
import os

def log_import(module_name):
    start = time.time()
    print(f"Attempting to import {module_name}...")
    return start

def end_import(module_name, start):
    end = time.time()
    print(f"Finished importing {module_name} in {end - start:.2f} seconds")

# Add the correct path to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Added {project_root} to Python path")

# Basic imports
start = log_import("torch")
import torch
end_import("torch", start)

start = log_import("torch.nn")
from torch import nn
end_import("torch.nn", start)

start = log_import("argparse")
import argparse
end_import("argparse", start)

start = log_import("json")
import json
end_import("json", start)

# Import checkpoint utilities
start = log_import("utilities.checkpoint_utils")
from utilities.checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint, setup_model_from_checkpoint
end_import("utilities.checkpoint_utils", start)

print("\nPython path:")
for path in sys.path:
    print(f"  {path}")

print("\nChecking models package location...")
models_dir = os.path.join(script_dir, "models")
print(f"Looking for models in: {models_dir}")
if os.path.exists(models_dir):
    print("models directory exists")
    init_file = os.path.join(models_dir, "__init__.py")
    if os.path.exists(init_file):
        print("__init__.py exists")
        print("Contents of __init__.py:")
        with open(init_file, 'r') as f:
            print(f.read())
    else:
        print("__init__.py does not exist")
else:
    print("models directory does not exist")

# Import AMBAModel with detailed logging
print("\nStarting AMBAModel import process...")
start_total = time.time()

start = log_import("timm")
import timm
end_import("timm", start)

start = log_import("models.AMBAModel")
from models import AMBAModel
end_import("models.AMBAModel", start)

print(f"Total import time: {time.time() - start_total:.2f} seconds")

def print_state_dict_info(state_dict, name="State Dict"):
    """Helper function to print information about a state dict."""
    print(f"\n{name} Information:")
    print(f"Number of keys: {len(state_dict)}")
    print("First few keys:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"  {key}")
    if 'module.' in next(iter(state_dict.keys()), ''):
        print("Contains 'module.' prefix")
    else:
        print("Does not contain 'module.' prefix")

def create_model(device, use_data_parallel=True):
    """Create a new AMBAModel instance."""
    print("\nCreating new AMBAModel...")
    
    vision_mamba_config = {
        'img_size': (512, 512),
        'patch_size': 16,
        'stride': 16,
        'embed_dim': 768,
        'depth': 24,
        'channels': 1,
        'num_classes': 1000,
        'drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'norm_epsilon': 1e-5,
        'rms_norm': False,
        'residual_in_fp32': False,
        'fused_add_norm': False,
        'if_rope': False,
        'if_rope_residual': False,
        'if_bidirectional': True,
        'final_pool_type': 'none',
        'if_abs_pos_embed': True,
        'if_bimamba': False,
        'if_cls_token': True,
        'if_devide_out': True,
        'use_double_cls_token': False,
        'use_middle_cls_token': False
    }

    model = AMBAModel(
        fshape=16,
        tshape=16,
        fstride=16,
        tstride=16,
        input_fdim=512,
        input_tdim=512,
        model_size='base',
        pretrain_stage=True,
        vision_mamba_config=vision_mamba_config
    )
    
    # Wrap in DataParallel if using CUDA and requested
    if use_data_parallel and torch.cuda.is_available():
        model = nn.DataParallel(model)
    
    model = model.to(device)
    print("Model created successfully")
    return model

def create_optimizer(model):
    """Create optimizer for the model."""
    print("\nCreating optimizer...")
    
    # Get all parameters
    audio_trainables = [p for p in model.parameters() if p.requires_grad]
    print(f"Total number of parameters: {len(list(model.parameters()))}")
    print(f"Total number of trainable parameters: {len(audio_trainables)}")
    print(f"Total parameter elements: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million")
    print(f"Total trainable parameter elements: {sum(p.numel() for p in audio_trainables) / 1e6:.2f} million")
    
    # Create optimizer with single parameter group
    optimizer = torch.optim.AdamW(
        audio_trainables,
        lr=1e-4,
        weight_decay=5e-8,
        betas=(0.95, 0.999)
    )
    
    print("Optimizer created successfully")
    return optimizer

def create_scheduler(optimizer):
    """Create scheduler for the optimizer."""
    print("\nCreating scheduler...")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    
    print("Scheduler created successfully")
    return scheduler

class MockMetricsTracker:
    """Mock metrics tracker for testing."""
    def __init__(self):
        self.best_metrics = {'acc': 0.0, 'loss': float('inf')}
        
    def should_save_best(self, acc):
        if acc > self.best_metrics['acc']:
            self.best_metrics['acc'] = acc
            return True
        return False

def test_existing_checkpoint(checkpoint_path, device, use_data_parallel=True):
    """Test loading an existing checkpoint."""
    print(f"\n===== Testing existing checkpoint: {checkpoint_path} =====")
    
    try:
        # First, try loading with weights_only=False (old behavior)
        print("\nAttempting to load checkpoint with weights_only=False...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("Successfully loaded checkpoint with weights_only=False")
    except Exception as e:
        print(f"Error loading with weights_only=False: {str(e)}")
        print("\nTrying to add numpy.core.multiarray.scalar to safe globals...")
        try:
            import numpy as np
            from torch.serialization import add_safe_globals
            add_safe_globals(['numpy.core.multiarray.scalar'])
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print("Successfully loaded checkpoint after adding scalar to safe globals")
        except Exception as e2:
            print(f"Error after adding safe globals: {str(e2)}")
            print("\nFalling back to weights_only=False with explicit setting...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                print("Successfully loaded checkpoint with explicit weights_only=False")
            except Exception as e3:
                print(f"All loading attempts failed: {str(e3)}")
                print("Cannot proceed with testing this checkpoint")
                return
    
    # Print checkpoint contents
    print("\nCheckpoint contents:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: {type(checkpoint[key])} with {len(checkpoint[key])} items")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # Print model state dict info
    if 'model_state_dict' in checkpoint:
        print_state_dict_info(checkpoint['model_state_dict'], "Checkpoint Model State")
    
    # Create a new model for loading
    new_model = create_model(device, use_data_parallel=use_data_parallel)
    
    # Define functions to create optimizer and scheduler for setup_model_from_checkpoint
    def create_optimizer_fn(model):
        return create_optimizer(model)
        
    def create_scheduler_fn(optimizer):
        return create_scheduler(optimizer)
    
    # Try to set up model from checkpoint
    print("\nSetting up model from checkpoint using checkpoint_utils.setup_model_from_checkpoint...")
    try:
        new_model, new_optimizer, new_scheduler, start_epoch = setup_model_from_checkpoint(
            checkpoint, new_model, create_optimizer_fn, create_scheduler_fn
        )
        print(f"Successfully set up model from checkpoint. Starting from epoch {start_epoch}")
        
        # Print loaded model state dict info
        print_state_dict_info(new_model.state_dict(), "Loaded Model State")
        
        return True
    except Exception as e:
        print(f"Error setting up model from checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--use_data_parallel", action="store_true", help="Whether to use DataParallel for the model")
    parser.add_argument("--test_checkpoint", type=str, default=None, help="Path to an existing checkpoint to test")
    parser.add_argument("--test_model_dir", type=str, default=None, help="Path to a model directory to test the latest checkpoint")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test existing checkpoint if provided
    if args.test_checkpoint:
        success = test_existing_checkpoint(args.test_checkpoint, device, args.use_data_parallel)
        if success:
            print("\nSuccessfully tested existing checkpoint!")
        else:
            print("\nFailed to test existing checkpoint.")
        return
    
    # Test latest checkpoint in model directory if provided
    if args.test_model_dir:
        print(f"\nLooking for latest checkpoint in {args.test_model_dir}...")
        last_epoch, checkpoint_path = find_latest_checkpoint(args.test_model_dir)
        
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"Found checkpoint for epoch {last_epoch} at {checkpoint_path}")
            success = test_existing_checkpoint(checkpoint_path, device, args.use_data_parallel)
            if success:
                print("\nSuccessfully tested latest checkpoint in model directory!")
            else:
                print("\nFailed to test latest checkpoint in model directory.")
        else:
            print(f"No checkpoint found in {args.test_model_dir}")
        return
    
    # Create a mock args object for checkpoint saving
    class Args:
        def __init__(self):
            self.task = 'pretrain_mpc'
            self.lr = 1e-4
            self.n_epochs = 100
            self.epoch_iter = 1000
            self.mask_patch = 400
            self.num_mel_bins = 512
            self.fshape = 16
            self.use_wandb = False
    
    mock_args = Args()
    
    try:
        # Create model, optimizer, and scheduler
        model = create_model(device, use_data_parallel=args.use_data_parallel)
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(optimizer)
        metrics_tracker = MockMetricsTracker()
        
        # Print initial model state
        print("\nInitial model state:")
        print_state_dict_info(model.state_dict(), "Initial Model State")
        
        # Create save directory
        save_dir = os.path.join(args.save_dir, "test_checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save checkpoint using our utility
        print("\nSaving checkpoint using checkpoint_utils.save_checkpoint...")
        epoch = 1
        global_step = 1000
        val_metrics = {'acc': 0.85, 'nce': 0.15}
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics_tracker=metrics_tracker,
            args=mock_args,
            exp_dir=save_dir,
            epoch=epoch,
            global_step=global_step,
            val_metrics=val_metrics,
            is_best=True
        )
        
        # Find the latest checkpoint
        print("\nFinding latest checkpoint using checkpoint_utils.find_latest_checkpoint...")
        last_epoch, checkpoint_path = find_latest_checkpoint(save_dir)
        print(f"Found checkpoint for epoch {last_epoch} at {checkpoint_path}")
        
        # Test the checkpoint we just created
        success = test_existing_checkpoint(checkpoint_path, device, args.use_data_parallel)
        if success:
            print("\nSuccessfully tested newly created checkpoint!")
        else:
            print("\nFailed to test newly created checkpoint.")
        
        print("\nAll tests completed!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 