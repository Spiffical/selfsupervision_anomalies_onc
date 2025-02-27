"""
Utilities for Weights & Biases (wandb) integration.
This module centralizes all wandb-related functionality for the project.
"""

import os
import wandb
import numpy as np
import torch
from collections import defaultdict

def init_wandb(args, project_name="ssamba", entity=None, group=None, run_id=None):
    """
    Initialize a wandb run.
    
    Args:
        args: Arguments object containing experiment configuration
        project_name: Name of the wandb project
        entity: wandb entity (username or team name)
        group: Group name for organizing runs
        run_id: Run ID for resuming a previous run
        
    Returns:
        wandb run object
    """
    # Check if wandb is already initialized
    if hasattr(args, 'wandb_initialized') and args.wandb_initialized:
        print("wandb already initialized, skipping initialization")
        return None
        
    if not args.use_wandb:
        return None
    
    # Extract relevant config from args
    config = {
        "architecture": args.model if hasattr(args, 'model') else None,
        "dataset": args.dataset if hasattr(args, 'dataset') else None,
        "batch_size": args.batch_size if hasattr(args, 'batch_size') else None,
        "learning_rate": args.lr if hasattr(args, 'lr') else None,
        "epochs": args.n_epochs if hasattr(args, 'n_epochs') else None,
    }
    
    # Add task-specific config
    if hasattr(args, 'task'):
        config["task"] = args.task
        
        if args.task.startswith('pretrain'):
            config["mask_patch"] = args.mask_patch if hasattr(args, 'mask_patch') else None
            config["fshape"] = args.fshape if hasattr(args, 'fshape') else None
            config["num_mel_bins"] = args.num_mel_bins if hasattr(args, 'num_mel_bins') else None
    
    # Initialize wandb
    run = wandb.init(
        project=project_name,
        entity=entity,
        config=config,
        name=os.path.basename(args.exp_dir),
        dir=args.exp_dir,
        resume="allow",
        group=group,
        id=run_id  # Use the same run ID if resuming
    )
    
    # Mark as initialized
    args.wandb_initialized = True
    
    return run

def create_hydrophone_table(task):
    """
    Create a wandb Table with appropriate columns based on the task.
    
    Args:
        task: The training task (e.g., 'pretrain_mpc', 'pretrain_mpg', etc.)
        
    Returns:
        wandb.Table object with appropriate columns
    """
    if task == 'pretrain_mpc':
        return wandb.Table(columns=["Epoch", "Hydrophone", "Sample Count", "Reconstruction Accuracy"])
    elif task == 'pretrain_mpg':
        return wandb.Table(columns=["Epoch", "Hydrophone", "Sample Count", "Patch Localization MSE"])
    elif task == 'pretrain_joint':
        return wandb.Table(columns=["Epoch", "Hydrophone", "Sample Count", "Reconstruction Accuracy", "Patch Localization MSE"])
    else:
        return wandb.Table(columns=["Epoch", "Hydrophone", "Sample Count", "Accuracy", "Precision", "Recall", "F2"])

def log_training_metrics(metrics_dict, step=None, use_wandb=True):
    """
    Log training metrics to wandb.
    
    Args:
        metrics_dict: Dictionary of metrics to log
        step: Optional step number for logging
        use_wandb: Whether to use wandb for logging
    """
    if not use_wandb:
        print("[DEBUG] wandb_utils.log_training_metrics: use_wandb is False, returning")
        return
    
    print("[DEBUG] wandb_utils.log_training_metrics: Starting")
    
    # Extract hydrophone metrics if present
    hydrophone_metrics = None
    if 'hydrophone_metrics' in metrics_dict:
        print(f"[DEBUG] wandb_utils.log_training_metrics: Found hydrophone_metrics in metrics_dict")
        hydrophone_metrics = metrics_dict.pop('hydrophone_metrics')
        print(f"[DEBUG] wandb_utils.log_training_metrics: Extracted hydrophone_metrics with {len(hydrophone_metrics)} hydrophones")
    
    # Get epoch from metrics_dict if available
    epoch = metrics_dict.get('pt_epoch', metrics_dict.get('ft_epoch', None))
    
    # Log regular metrics using epoch instead of step for x-axis
    print(f"[DEBUG] wandb_utils.log_training_metrics: Logging regular metrics: {list(metrics_dict.keys())}")
    wandb.log(metrics_dict, step=epoch)
    
    # Log hydrophone metrics separately if present
    if hydrophone_metrics:
        epoch = metrics_dict.get('pt_epoch', metrics_dict.get('ft_epoch', epoch))
        prefix = "pt_" if 'pt_epoch' in metrics_dict else "ft_"
        print(f"[DEBUG] wandb_utils.log_training_metrics: Calling log_hydrophone_metrics for epoch {epoch} with prefix {prefix}")
        
        # Log per-hydrophone metrics
        log_hydrophone_metrics(hydrophone_metrics, epoch=epoch, prefix=prefix)
    else:
        print("[DEBUG] wandb_utils.log_training_metrics: No hydrophone_metrics to log")
    
    print("[DEBUG] wandb_utils.log_training_metrics: Finished")

def log_hydrophone_metrics(hydrophone_metrics, epoch=None, prefix=""):
    """
    Log hydrophone-specific metrics to wandb.
    
    Args:
        hydrophone_metrics: Dictionary of hydrophone metrics
        epoch: Current epoch number
        prefix: Prefix for metric names (e.g., 'pt_' for pretraining)
    """
    print(f"[DEBUG] wandb_utils.log_hydrophone_metrics: Starting with {len(hydrophone_metrics)} hydrophones, epoch {epoch}, prefix {prefix}")
    
    # Log per-hydrophone metrics
    for hydrophone, metrics in hydrophone_metrics.items():
        metric_dict = {}
        
        for metric_name, value in metrics.items():
            if metric_name != 'count':  # Handle count separately
                metric_dict[f"{prefix}{metric_name.capitalize()}/{hydrophone}"] = value
        
        # Always log sample count
        if 'count' in metrics:
            metric_dict[f"{prefix}Sample_Count/{hydrophone}"] = metrics['count']
        
        # Add epoch to ensure metrics are tracked as a function of epoch
        if epoch is not None:
            metric_dict[f"{prefix}epoch"] = epoch
        
        print(f"[DEBUG] wandb_utils.log_hydrophone_metrics: Logging metrics for {hydrophone}: {metric_dict}")
        wandb.log(metric_dict, step=epoch)  # Use epoch for x-axis
    
    # Create custom wandb.Table for sample distribution periodically
    if isinstance(epoch, int) and (epoch == 1 or epoch % 10 == 0):
        print(f"[DEBUG] wandb_utils.log_hydrophone_metrics: Creating sample distribution table for epoch {epoch}")
        # Create table for sample distribution
        table_data = [[hydrophone, metrics['count']] for hydrophone, metrics in hydrophone_metrics.items()]
        wandb.log({
            f"{prefix}Sample_Distribution": wandb.Table(
                data=table_data,
                columns=["Hydrophone", "Sample Count"]
            )
        }, step=epoch)  # Use epoch for x-axis
        
        # Create tables for each metric type
        metric_types = set()
        for metrics in hydrophone_metrics.values():
            metric_types.update([k for k in metrics.keys() if k != 'count'])
            
        for metric_type in metric_types:
            table_data = [[hydrophone, metrics.get(metric_type, 0), metrics['count']] 
                          for hydrophone, metrics in hydrophone_metrics.items() 
                          if metric_type in metrics]
            
            if table_data:  # Only create table if we have data
                wandb.log({
                    f"{prefix}{metric_type.capitalize()}_by_Hydrophone": wandb.Table(
                        data=table_data,
                        columns=["Hydrophone", metric_type.capitalize(), "Sample Count"]
                    )
                }, step=epoch)  # Use epoch for x-axis
        
        # Create interactive plots for metrics over time
        create_hydrophone_plots(hydrophone_metrics, epoch, prefix, metric_types)

def create_hydrophone_plots(hydrophone_metrics, epoch, prefix, metric_types):
    """
    Create interactive plots for hydrophone metrics over time.
    
    Args:
        hydrophone_metrics: Dictionary of hydrophone metrics
        epoch: Current epoch number
        prefix: Prefix for metric names
        metric_types: Set of metric types to plot
    """
    import wandb
    
    # Only create plots periodically to avoid cluttering the UI
    if not isinstance(epoch, int) or (epoch != 1 and epoch % 10 != 0):
        return
        
    # Get list of hydrophones
    hydrophones = list(hydrophone_metrics.keys())
    
    # Create a plot for each metric type
    for metric_type in metric_types:
        # Skip if this metric type isn't present
        if not any(metric_type in metrics for metrics in hydrophone_metrics.values()):
            continue
            
        # Create data for the plot
        data = []
        for hydrophone in hydrophones:
            if metric_type in hydrophone_metrics[hydrophone]:
                data.append([hydrophone, hydrophone_metrics[hydrophone][metric_type]])
        
        if not data:
            continue
            
        # Sort by metric value for better visualization
        data.sort(key=lambda x: x[1], reverse=True)
        
        # Create a bar chart
        hydrophone_names = [item[0] for item in data]
        metric_values = [item[1] for item in data]
        
        wandb.log({
            f"{prefix}{metric_type.capitalize()}_Plot": wandb.plot.bar(
                wandb.Table(data=[[h, v] for h, v in zip(hydrophone_names, metric_values)],
                           columns=["Hydrophone", metric_type.capitalize()]),
                "Hydrophone", 
                metric_type.capitalize(),
                title=f"{metric_type.capitalize()} by Hydrophone (Epoch {epoch})"
            )
        }, step=epoch)  # Use epoch for x-axis

def log_validation_metrics(metrics, task, epoch=None, prefix="", use_wandb=True):
    """
    Log validation metrics to wandb.
    
    Args:
        metrics: Dictionary of validation metrics
        task: The training task
        epoch: Current epoch number
        prefix: Prefix for metric names
        use_wandb: Whether to use wandb for logging
    """
    if not use_wandb:
        return
    
    wandb_metrics = {}
    
    if task.startswith('pretrain_'):
        # Pretraining metrics
        wandb_metrics.update({
            f"{prefix}val_accuracy": metrics['acc'],
            f"{prefix}val_loss": metrics['nce'],
            f"{prefix}epoch": epoch if epoch is not None else 0
        })
        
        # Add per-hydrophone metrics for pretraining if available
        if 'hydrophone_metrics' in metrics and metrics['hydrophone_metrics']:
            # Log hydrophone metrics separately to ensure they're tracked by epoch
            log_hydrophone_metrics(metrics['hydrophone_metrics'], epoch=epoch, prefix=f"{prefix}val_")
            
            # Also include in the main metrics dict for completeness
            for hydrophone, hyd_metrics in metrics['hydrophone_metrics'].items():
                for metric_name, value in hyd_metrics.items():
                    if task == 'pretrain_mpc':
                        if metric_name in ['accuracy']:
                            wandb_metrics[f"{prefix}hydrophone/{hydrophone}/val_{metric_name}"] = value
                    elif task == 'pretrain_mpg':
                        if metric_name in ['mse']:
                            wandb_metrics[f"{prefix}hydrophone/{hydrophone}/val_{metric_name}"] = value
                    elif task == 'pretrain_joint':
                        if metric_name in ['accuracy', 'mpc_accuracy', 'mpg_mse']:
                            wandb_metrics[f"{prefix}hydrophone/{hydrophone}/val_{metric_name}"] = value
                    
                    if metric_name == 'count':
                        wandb_metrics[f"{prefix}hydrophone/{hydrophone}/sample_count"] = value
    else:
        # Fine-tuning metrics
        wandb_metrics.update({
            f"{prefix}val_loss": metrics.get('loss', 0),
            f"{prefix}val_accuracy": metrics.get('acc', 0),
            f"{prefix}val_auc": metrics.get('auc', 0),
            f"{prefix}val_precision": metrics.get('global_precision', 0),
            f"{prefix}val_recall": metrics.get('global_recall', 0),
            f"{prefix}val_f2": metrics.get('global_f2', 0),
            f"{prefix}epoch": epoch if epoch is not None else 0
        })
        
        # Add per-hydrophone metrics
        if 'hydrophone_metrics' in metrics and metrics['hydrophone_metrics']:
            # Log hydrophone metrics separately to ensure they're tracked by epoch
            log_hydrophone_metrics(metrics['hydrophone_metrics'], epoch=epoch, prefix=f"{prefix}val_")
            
            # Also include in the main metrics dict for completeness
            for hydrophone, hyd_metrics in metrics['hydrophone_metrics'].items():
                for metric_name, value in hyd_metrics.items():
                    if metric_name in ['accuracy', 'precision', 'recall', 'f2']:
                        wandb_metrics[f"{prefix}hydrophone/{hydrophone}/val_{metric_name}"] = value
                    elif metric_name == 'count':
                        wandb_metrics[f"{prefix}hydrophone/{hydrophone}/sample_count"] = value
    
    # Always use epoch for x-axis
    wandb.log(wandb_metrics, step=epoch)

def log_model_artifact(model, model_path, name, type="model", metadata=None):
    """
    Log a model as a wandb artifact.
    
    Args:
        model: The model to log
        model_path: Path where the model is saved
        name: Name for the artifact
        type: Type of artifact
        metadata: Optional metadata to attach to the artifact
    """
    # Save model if it's not already saved
    if not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)
    
    # Create artifact
    artifact = wandb.Artifact(name=name, type=type, metadata=metadata)
    artifact.add_file(model_path)
    
    # Log artifact
    wandb.log_artifact(artifact)

def finish_run():
    """
    Finish the current wandb run.
    """
    if wandb.run is not None:
        wandb.finish() 