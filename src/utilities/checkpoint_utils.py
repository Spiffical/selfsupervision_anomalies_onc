import os
import torch
import pickle
import numpy as np
from collections import defaultdict

def save_checkpoint(model, optimizer, scheduler, metrics_tracker, args, exp_dir, 
                   epoch, global_step, val_metrics, is_best=False):
    """
    Save model checkpoint with all necessary states for resuming training.
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        scheduler: The scheduler state to save
        metrics_tracker: The metrics tracker with best metrics
        args: Training arguments
        exp_dir: Experiment directory path
        epoch: Current epoch
        global_step: Current global step
        val_metrics: Validation metrics
        is_best: Whether this is the best model so far
    """
    # Extract only necessary args to avoid pickle issues
    args_to_save = {
        'task': args.task,
        'lr': args.lr,
        'n_epochs': args.n_epochs,
        'epoch_iter': args.epoch_iter,
        'mask_patch': args.mask_patch if hasattr(args, 'mask_patch') else None,
        'num_mel_bins': args.num_mel_bins,
        'fshape': args.fshape if hasattr(args, 'fshape') else None
    }
    
    # Get model state dict - handle DataParallel case
    if hasattr(model, 'module'):
        # If using DataParallel, save without module prefix for consistency
        model_state_dict = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
    else:
        model_state_dict = model.state_dict()
    
    # Convert any NumPy values in val_metrics to Python native types or torch tensors
    safe_val_metrics = {}
    for k, v in val_metrics.items():
        if hasattr(v, 'dtype') and hasattr(v, 'item') and not isinstance(v, torch.Tensor):
            # This is likely a numpy scalar, convert to Python native type
            safe_val_metrics[k] = v.item()
        else:
            safe_val_metrics[k] = v
    
    # Convert best_metrics to safe types if available
    safe_best_metrics = None
    if metrics_tracker and hasattr(metrics_tracker, 'best_metrics'):
        safe_best_metrics = {}
        for k, v in metrics_tracker.best_metrics.items():
            if hasattr(v, 'dtype') and hasattr(v, 'item') and not isinstance(v, torch.Tensor):
                # This is likely a numpy scalar, convert to Python native type
                safe_best_metrics[k] = v.item()
            else:
                safe_best_metrics[k] = v
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metrics': safe_best_metrics,
        'val_acc': safe_val_metrics.get('acc', 0.0),
        'args': args_to_save
    }
    
    # Save regular checkpoint
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    checkpoint_path = os.path.join(exp_dir, f'models/checkpoint.{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint for epoch {epoch} to {checkpoint_path}")
    
    # Save best checkpoint if needed
    if is_best:
        best_path = os.path.join(exp_dir, 'models/best_checkpoint.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved new best checkpoint with accuracy: {safe_val_metrics.get('acc', 0.0):.6f}")

def load_checkpoint(checkpoint_path, device=None):
    """
    Load checkpoint from file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to
        
    Returns:
        checkpoint: The loaded checkpoint dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        # First try with weights_only=False (PyTorch < 2.6 behavior)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"Checkpoint loaded with weights_only=False. Contents: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"Error loading checkpoint with weights_only=False: {str(e)}")
        # Try adding numpy.core.multiarray.scalar to safe globals
        try:
            from torch.serialization import add_safe_globals
            add_safe_globals(['numpy.core.multiarray.scalar'])
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print(f"Checkpoint loaded after adding scalar to safe globals. Contents: {list(checkpoint.keys())}")
        except Exception as e2:
            print(f"All loading attempts failed. Last error: {str(e2)}")
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}") from e2
    
    return checkpoint

def find_latest_checkpoint(exp_dir):
    """
    Find the latest checkpoint in the experiment directory.
    
    Args:
        exp_dir: Experiment directory path
        
    Returns:
        last_epoch: The epoch of the latest checkpoint
        checkpoint_path: Path to the latest checkpoint file
    """
    progress_path = os.path.join(exp_dir, 'progress.pkl')
    last_epoch = 0
    checkpoint_path = None
    
    # Try to get the last epoch from progress file
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "rb") as f:
                progress = pickle.load(f)
                if progress:
                    last_epoch = progress[-1][0]  # First element is epoch
                    print(f"Found previous training progress at epoch {last_epoch}")
                    # Look for the last saved checkpoint
                    checkpoint_path = os.path.join(exp_dir, f'models/checkpoint.{last_epoch}.pth')
        except Exception as e:
            print(f"Could not load progress file: {str(e)}")
            last_epoch = 0
    
    # If no progress file or couldn't load it, try to find checkpoints directly
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        models_dir = os.path.join(exp_dir, 'models')
        if os.path.exists(models_dir):
            checkpoint_files = [f for f in os.listdir(models_dir) if f.startswith('checkpoint.') and f.endswith('.pth')]
            if checkpoint_files:
                # Extract epoch numbers and find the latest
                epochs = [int(f.split('.')[1]) for f in checkpoint_files]
                last_epoch = max(epochs)
                checkpoint_path = os.path.join(models_dir, f'checkpoint.{last_epoch}.pth')
                print(f"Found checkpoint file for epoch {last_epoch}")
    
    return last_epoch, checkpoint_path

def setup_model_from_checkpoint(checkpoint, model, create_optimizer_fn, create_scheduler_fn):
    """
    Set up model, optimizer and scheduler from a checkpoint.
    
    Args:
        checkpoint: The loaded checkpoint dictionary
        model: The model to load state into
        create_optimizer_fn: Function to create optimizer (takes model as input)
        create_scheduler_fn: Function to create scheduler (takes optimizer as input)
        
    Returns:
        model: Model with loaded state
        optimizer: Optimizer with loaded state
        scheduler: Scheduler with loaded state
        start_epoch: Epoch to start from
    """
    # Load model state
    print("Loading model state...")
    
    # Handle DataParallel module prefix mismatch
    state_dict = checkpoint['model_state_dict']
    
    # Check if the model is wrapped in DataParallel but the state_dict is not
    if hasattr(model, 'module'):
        # Add 'module.' prefix to keys if they don't have it
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                new_state_dict['module.' + k] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    # Check if the model is not wrapped in DataParallel but the state_dict has 'module.' prefix
    elif not hasattr(model, 'module'):
        # Remove 'module.' prefix from keys if they have it
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix (7 characters)
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # Load the state dict
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model state loaded successfully")
    except Exception as e:
        print(f"Error loading model state: {str(e)}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print("Model state loaded with strict=False")
    
    # Create and load optimizer
    print("Creating optimizer and loading state...")
    optimizer = create_optimizer_fn(model)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Optimizer state loaded successfully")
    
    # Create and load scheduler
    print("Creating scheduler and loading state...")
    scheduler = create_scheduler_fn(optimizer)
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded successfully")
    
    # Get starting epoch
    start_epoch = checkpoint['epoch']
    print(f"Resuming from epoch {start_epoch}")
    
    return model, optimizer, scheduler, start_epoch 