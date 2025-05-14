"""
Utilities for Weights & Biases (wandb) integration.
This module centralizes all wandb-related functionality for the project.
"""

import os
import wandb
import numpy as np
import torch
from collections import defaultdict
import pandas as pd
import json

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
    
    # Setup custom report sections
    if run is not None:
        # Create custom sections in the dashboard
        setup_custom_sections(run, args.task if hasattr(args, 'task') else None)
    
    # Mark as initialized
    args.wandb_initialized = True
    
    return run

def setup_custom_sections(run, task=None):
    """
    Set up custom sections in the wandb dashboard
    
    Args:
        run: wandb run object
        task: Current task (e.g. 'pretrain_mpc', 'finetune_classification')
    """
    # Define the sections we want to create
    sections = [
        "Global Metrics",
    ]
    
    # Add per-hydrophone sections for each metric type
    per_hydrophone_sections = []
    
    if task and task.startswith('pretrain'):
        sections.extend([
            "Pretraining Accuracy",
            "Pretraining Loss",
            "Pretraining Other Metrics"
        ])
        per_hydrophone_sections.extend([
            "Per_Hydrophone_PT_Accuracy",
            "Per_Hydrophone_PT_Loss",
            "Per_Hydrophone_PT_NCE"
        ])
    else:
        sections.extend([
            "Finetuning Accuracy",
            "Finetuning Loss",
            "Finetuning Precision",
            "Finetuning Recall",
            "Finetuning F2",
            "Finetuning AUC"
        ])
        per_hydrophone_sections.extend([
            "Per_Hydrophone_FT_Accuracy",
            "Per_Hydrophone_FT_Loss",
            "Per_Hydrophone_FT_Precision",
            "Per_Hydrophone_FT_Recall",
            "Per_Hydrophone_FT_F2",
            "Per_Hydrophone_FT_AUC"
        ])
    
    # Add per-hydrophone sections to the main sections list
    sections.extend(per_hydrophone_sections)
        
    # Create an empty dashboard visualization for each section
    for section in sections:
        # This creates a custom section if it doesn't exist yet
        run.log({f"{section}/initialized": True})

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
        return wandb.Table(columns=["Epoch", "Hydrophone", "Sample Count", "Accuracy", "Precision", "Recall", "F2", "AUC"])

def log_training_metrics(metrics_dict, use_wandb=True):
    """Log training metrics to wandb with organized categories."""
    if not use_wandb:
        return
        
    import wandb
    if wandb.run is None:
        print("[DEBUG] Warning: wandb.run is None, skipping metric logging")
        return
        
    print("[DEBUG] wandb_utils.log_training_metrics: Starting")
    
    # Organize metrics by category
    pt_metrics = {}  # Pretraining metrics
    ft_metrics = {}  # Finetuning metrics
    global_metrics = {}  # Global metrics
    
    # Group metrics by type
    for key, value in metrics_dict.items():
        if key.startswith('pt_'):
            clean_key = key[3:]  # Remove 'pt_' prefix
            pt_metrics[clean_key] = value
        elif key.startswith('ft_'):
            clean_key = key[3:]  # Remove 'ft_' prefix
            ft_metrics[clean_key] = value
        else:
            # Global metrics
            global_metrics[key] = value
    
    try:
        # Log global metrics
        if global_metrics:
            wandb.log({"Global Metrics/training": global_metrics})
        
        # Log pretraining metrics to appropriate sections
        for key, value in pt_metrics.items():
            if 'acc' in key or 'accuracy' in key:
                wandb.log({"Pretraining Accuracy/training": {key: value}})
            elif 'loss' in key:
                wandb.log({"Pretraining Loss/training": {key: value}})
            else:
                wandb.log({"Pretraining Other Metrics/training": {key: value}})
        
        # Log finetuning metrics to appropriate sections
        for key, value in ft_metrics.items():
            if 'acc' in key or 'accuracy' in key:
                wandb.log({"Finetuning Accuracy/training": {key: value}})
            elif 'loss' in key:
                wandb.log({"Finetuning Loss/training": {key: value}})
            elif 'precision' in key:
                wandb.log({"Finetuning Precision/training": {key: value}})
            elif 'recall' in key:
                wandb.log({"Finetuning Recall/training": {key: value}})
            elif 'f2' in key or 'f1' in key:
                wandb.log({"Finetuning F2/training": {key: value}})
            elif 'auc' in key:
                wandb.log({"Finetuning AUC/training": {key: value}})
            else:
                wandb.log({"Global Metrics/training": {key: value}})
    
    except Exception as e:
        print(f"[DEBUG] Error logging training metrics: {str(e)}")
    
    print("[DEBUG] wandb_utils.log_training_metrics: Finished")

def log_validation_metrics(metrics, task, epoch, prefix="", use_wandb=True):
    """Log validation metrics to wandb with organized categories."""
    if not use_wandb:
        return
        
    import wandb
    if wandb.run is None:
        print("[DEBUG] Warning: wandb.run is None, skipping validation metric logging")
        return
    
    print("[DEBUG] wandb_utils.log_validation_metrics: Starting")
    
    # Determine if we're in pretraining or finetuning phase
    is_pretraining = task.startswith('pretrain')
    section_prefix = "Pretraining" if is_pretraining else "Finetuning"
    
    # Process each metric
    for k, v in metrics.items():
        if k == 'hydrophone_metrics':
            continue
            
        # Clean the key name
        clean_key = k[len(prefix):] if prefix and k.startswith(prefix) else k
        
        # Determine which section to log to
        if 'acc' in clean_key or 'accuracy' in clean_key:
            section = f"{section_prefix} Accuracy"
        elif 'loss' in clean_key:
            section = f"{section_prefix} Loss"
        elif 'precision' in clean_key:
            section = f"{section_prefix} Precision"
        elif 'recall' in clean_key:
            section = f"{section_prefix} Recall"
        elif 'f2' in clean_key or 'f1' in clean_key:
            section = f"{section_prefix} F2"
        elif 'auc' in clean_key:
            section = f"{section_prefix} AUC"
        else:
            section = "Global Metrics"
        
        # Log to the appropriate section
        wandb.log({f"{section}/validation": {clean_key: v, "epoch": epoch}})
    
    # Log per-hydrophone metrics if available
    if 'hydrophone_metrics' in metrics:
        print("[DEBUG] Found hydrophone metrics, logging them separately")
        log_hydrophone_metrics(metrics, epoch, prefix, use_wandb)
    
    print("[DEBUG] wandb_utils.log_validation_metrics: Finished")

def log_hydrophone_metrics(metrics, epoch, prefix="", use_wandb=True):
    """Log per-hydrophone metrics to wandb with organized categories."""
    if not use_wandb or 'hydrophone_metrics' not in metrics:
        return
        
    import wandb
    if wandb.run is None:
        print("[DEBUG] Warning: wandb.run is None, skipping hydrophone metric logging")
        return
    
    print("[DEBUG] wandb_utils.log_hydrophone_metrics: Starting")
    
    hydrophone_metrics = metrics['hydrophone_metrics']
    
    # Determine if we're in pretraining or finetuning phase
    is_pretraining = prefix.startswith('pt')
    phase_prefix = "PT" if is_pretraining else "FT"
    
    # Group metrics by type
    metric_groups = defaultdict(dict)
    
    # Process each hydrophone's metrics
    for hydrophone, hyd_metrics in hydrophone_metrics.items():
        # Process pretraining metrics
        if 'mpc_accuracy' in hyd_metrics:
            metric_groups["accuracy"][hydrophone] = hyd_metrics['mpc_accuracy']
        
        if 'mpg_mse' in hyd_metrics:
            metric_groups["mse"][hydrophone] = hyd_metrics['mpg_mse']
            
        if 'nce' in hyd_metrics:
            metric_groups["nce"][hydrophone] = hyd_metrics['nce']
        
        # Process finetuning metrics
        if 'accuracy' in hyd_metrics:
            metric_groups["accuracy"][hydrophone] = hyd_metrics['accuracy']
            
        if 'precision' in hyd_metrics:
            metric_groups["precision"][hydrophone] = hyd_metrics['precision']
            
        if 'recall' in hyd_metrics:
            metric_groups["recall"][hydrophone] = hyd_metrics['recall']
            
        if 'f2' in hyd_metrics:
            metric_groups["f2"][hydrophone] = hyd_metrics['f2']
            
        if 'auc' in hyd_metrics:
            metric_groups["auc"][hydrophone] = hyd_metrics['auc']
    
    try:
        # We no longer log global averages to Global Metrics
        # This was previously causing per-hydrophone metrics to appear in Global Metrics
        
        # Log per-hydrophone metrics to their dedicated sections
        for metric_type, values in metric_groups.items():
            if not values:
                continue
                
            # Determine the appropriate section name for per-hydrophone metrics
            if metric_type == "accuracy":
                section = f"Per_Hydrophone_{phase_prefix}_Accuracy"
            elif metric_type in ["loss", "mse"]:
                section = f"Per_Hydrophone_{phase_prefix}_Loss"
            elif metric_type == "precision":
                section = f"Per_Hydrophone_{phase_prefix}_Precision"
            elif metric_type == "recall":
                section = f"Per_Hydrophone_{phase_prefix}_Recall"
            elif metric_type in ["f2", "f1"]:
                section = f"Per_Hydrophone_{phase_prefix}_F2"
            elif metric_type == "nce":
                section = f"Per_Hydrophone_{phase_prefix}_NCE"
            else:
                section = f"Per_Hydrophone_{phase_prefix}_Other"
            
            # Create the data dictionary with all hydrophone values
            hydrophone_data = {**values, "epoch": epoch}
            
            # Log the raw data to the section
            wandb.log({f"{section}/data": hydrophone_data})
            
            # Create and log a bar chart
            data = [[h, v] for h, v in values.items()]
            table = wandb.Table(data=data, columns=["Hydrophone", metric_type.capitalize()])
            chart = wandb.plot.bar(table, "Hydrophone", metric_type.capitalize(), 
                                  title=f"{metric_type.capitalize()} by Hydrophone")
            
            # Log the chart to the appropriate section
            wandb.log({f"{section}/bar_chart": chart})
            
    except Exception as e:
        print(f"[DEBUG] Error logging hydrophone metrics: {str(e)}")
        print(f"Exception details: {str(e)}")
    
    print("[DEBUG] wandb_utils.log_hydrophone_metrics: Finished")

def create_hydrophone_plots(hydrophone_metrics, epoch, prefix, metric_types):
    """This function is now handled by the reorganized log_hydrophone_metrics"""
    pass

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