"""
Utilities for tracking training metrics and progress.
"""

import os
import torch
import pickle
import numpy as np
import time
from collections import defaultdict
import torch.nn as nn
from ..wandb_utils import log_training_metrics, create_hydrophone_table

class MetricsTracker:
    """Class to track training metrics and handle model checkpointing."""
    def __init__(self, exp_dir, args, use_wandb=False):
        self.exp_dir = exp_dir
        self.args = args
        self.use_wandb = use_wandb
        self.progress = []
        self.start_time = None
        
        # Initialize best metrics with worst possible values
        self.best_metrics = {
            'acc': 0.0,
            'auc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f2': 0.0,
            'loss': float('inf')
        }
        
        # Load previous progress if resuming
        self._load_previous_progress()
        
        # Create predictions directory
        os.makedirs(os.path.join(exp_dir, 'predictions'), exist_ok=True)
        
        # Initialize wandb table for hydrophone metrics
        if self.use_wandb:
            if hasattr(args, 'task'):
                self.hydrophone_table = create_hydrophone_table(args.task)
            else:
                self.hydrophone_table = create_hydrophone_table('pretrain_mpc')
        
        # Create necessary directories
        os.makedirs(f"{exp_dir}/models", exist_ok=True)
        
        # Load previous progress if exists
        self._load_previous_progress()
    
    def _load_previous_progress(self):
        """Load previous training progress and best metrics."""
        # Check for task-specific progress file first
        task_progress_path = None
        if hasattr(self.args, 'task'):
            task_prefix = self.args.task.replace('_', '-')
            task_progress_path = f"{self.exp_dir}/{task_prefix}_progress.pkl"
        
        # Default progress path as fallback
        default_progress_path = f"{self.exp_dir}/progress.pkl"
        
        # Check for task-specific best metric file
        task_best_metric_path = None
        if hasattr(self.args, 'task'):
            task_prefix = self.args.task.replace('_', '-')
            task_best_metric_path = f"{self.exp_dir}/models/{task_prefix}_best_metric.pth"
        
        # Default best metric path as fallback
        default_best_metric_path = f"{self.exp_dir}/models/best_metric.pth"
        
        # Try to load task-specific best metric first, then fall back to default
        best_metric_path = task_best_metric_path if task_best_metric_path and os.path.exists(task_best_metric_path) else default_best_metric_path
        
        # Load best metric
        if os.path.exists(best_metric_path):
            try:
                # First try loading with weights_only=False since we know this is our metric file
                try:
                    # Add numpy scalar to safe globals
                    torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
                    self.best_metrics['acc'] = torch.load(
                        best_metric_path,
                        map_location='cuda' if torch.cuda.is_available() else 'cpu',
                        weights_only=False  # Explicitly set to False for metrics
                    )
                except Exception as e1:
                    # If that fails, try loading without weights_only parameter
                    try:
                        self.best_metrics['acc'] = torch.load(
                            best_metric_path,
                            map_location='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                    except Exception as e2:
                        # If both attempts fail, use default value
                        print(f"Could not load best metric: {str(e2)}")
                        self.best_metrics['acc'] = -np.inf
                        print("Using default best accuracy value")
                        return

                print(f"Restored previous best accuracy: {self.best_metrics['acc']:.6f}")
            except Exception as e:
                print(f"Could not load best metric: {str(e)}")
                self.best_metrics['acc'] = -np.inf
                print("Using default best accuracy value")
        
        # Try to load task-specific progress first, then fall back to default
        progress_path = task_progress_path if task_progress_path and os.path.exists(task_progress_path) else default_progress_path
        
        # Load progress
        if os.path.exists(progress_path):
            try:
                with open(progress_path, "rb") as f:
                    self.progress = pickle.load(f)
                if self.progress:
                    print(f"Restored previous progress with {len(self.progress)} entries from {progress_path}")
            except Exception as e:
                print(f"Could not load previous progress: {str(e)}")
                self.progress = []
    
    def save_progress(self, epoch, global_step, best_epoch):
        """Save training progress."""
        self.progress.append([
            epoch, 
            global_step, 
            best_epoch, 
            self.best_metrics['acc'],
            self._get_elapsed_time()
        ])
        
        # Save to the main progress file
        with open(f"{self.exp_dir}/progress.pkl", "wb") as f:
            pickle.dump(self.progress, f)
            
        # If task is available, also save to a task-specific progress file
        if hasattr(self.args, 'task'):
            task_prefix = self.args.task.replace('_', '-')
            with open(f"{self.exp_dir}/{task_prefix}_progress.pkl", "wb") as f:
                pickle.dump(self.progress, f)
    
    def _get_elapsed_time(self):
        """Get elapsed time since training started."""
        if self.start_time is None:
            self.start_time = time.time()
        return time.time() - self.start_time
    
    def save_model(self, model, optimizer, metric_value, metric_name='acc', is_best=False):
        """Save model checkpoint."""
        if is_best:
            # Get task prefix if available
            task_prefix = ""
            if hasattr(self.args, 'task'):
                task_prefix = f"{self.args.task.replace('_', '-')}_"
            
            # Save best model state
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(model_state, f"{self.exp_dir}/models/{task_prefix}best_audio_model.pth")
            
            # Save complete checkpoint
            checkpoint = {
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': metric_value,
                'metric_name': metric_name,
                # Save only necessary args as a dict
                'args': {
                    'task': self.args.task if hasattr(self.args, 'task') else None,
                    'lr': self.args.lr if hasattr(self.args, 'lr') else None,
                    'head_lr': self.args.head_lr if hasattr(self.args, 'head_lr') else None,
                    'n_epochs': self.args.n_epochs if hasattr(self.args, 'n_epochs') else None,
                    'adaptschedule': self.args.adaptschedule if hasattr(self.args, 'adaptschedule') else None,
                    'main_metric': self.args.main_metric if hasattr(self.args, 'main_metric') else None,
                    'loss': self.args.loss if hasattr(self.args, 'loss') else None
                }
            }
            torch.save(checkpoint, f"{self.exp_dir}/models/{task_prefix}best_checkpoint.pth")
            
            # Save optimizer state for backward compatibility
            torch.save(optimizer.state_dict(), f"{self.exp_dir}/models/{task_prefix}best_optim_state.pth")
            
            # Save metric value separately for backward compatibility
            torch.save(
                metric_value,
                f"{self.exp_dir}/models/{task_prefix}best_metric.pth",
                _use_new_zipfile_serialization=False,  # Use old format for better compatibility
            )
            self.best_metrics[metric_name] = metric_value
            print(f"Saved new best {task_prefix.rstrip('_')} model with {metric_name}: {metric_value:.6f}")
    
    def save_predictions(self, predictions, epoch, is_cumulative=False):
        """Save model predictions."""
        # Get task prefix if available
        task_prefix = ""
        if hasattr(self.args, 'task'):
            task_prefix = f"{self.args.task.replace('_', '-')}_"
            
        filename = f'{task_prefix}cum_predictions.csv' if is_cumulative else f'{task_prefix}predictions_{epoch}.csv'
        np.savetxt(f"{self.exp_dir}/predictions/{filename}", predictions, delimiter=',')
    
    def log_training_metrics(self, metrics_dict, step=None):
        """Log training metrics to wandb."""
        if self.use_wandb:
            print("[DEBUG] Inside MetricsTracker.log_training_metrics")
            # Don't pass step parameter, we'll use epoch from metrics_dict
            log_training_metrics(metrics_dict, use_wandb=self.use_wandb)
            
            # If hydrophone metrics are present, update the table
            print(f"[DEBUG] Checking for hydrophone_metrics in metrics_dict: {'hydrophone_metrics' in metrics_dict}")
            if 'hydrophone_metrics' in metrics_dict:
                epoch = metrics_dict.get('pt_epoch', metrics_dict.get('ft_epoch', 0))
                print(f"[DEBUG] Found hydrophone_metrics for epoch {epoch}")
                hydrophone_metrics = metrics_dict['hydrophone_metrics']
                print(f"[DEBUG] Number of hydrophones in metrics: {len(hydrophone_metrics)}")
                
                for hydrophone, metrics in hydrophone_metrics.items():
                    print(f"[DEBUG] Processing hydrophone {hydrophone} with metrics: {metrics}")
                    print(f"[DEBUG] Task: {self.args.task}")
                    
                    if self.args.task == 'pretrain_mpc':
                        print(f"[DEBUG] Adding data to hydrophone_table for {hydrophone} in pretrain_mpc task")
                        self.hydrophone_table.add_data(
                            epoch,
                            hydrophone,
                            metrics.get('count', 0),
                            metrics.get('accuracy', 0.0)
                        )
                    elif self.args.task == 'pretrain_mpg':
                        self.hydrophone_table.add_data(
                            epoch,
                            hydrophone,
                            metrics.get('count', 0),
                            metrics.get('mse', 0.0)
                        )
                    elif self.args.task == 'pretrain_joint':
                        # For joint training, we'll use the accuracy metric we calculated
                        # We don't have separate mpc_accuracy and mpg_mse for each hydrophone
                        # since we're only storing MPC predictions
                        print(f"[DEBUG] Adding data to hydrophone_table for {hydrophone} in pretrain_joint task")
                        
                        # Check which metrics are available
                        if 'accuracy' in metrics:
                            mpc_accuracy = metrics.get('accuracy', 0.0)
                            print(f"[DEBUG] Using accuracy: {mpc_accuracy}")
                        elif 'mpc_accuracy' in metrics:
                            mpc_accuracy = metrics.get('mpc_accuracy', 0.0)
                            print(f"[DEBUG] Using mpc_accuracy: {mpc_accuracy}")
                        else:
                            mpc_accuracy = 0.0
                            print(f"[DEBUG] No accuracy metrics found, using 0.0")
                        
                        # Use a placeholder for MPG MSE since we don't have it
                        mpg_mse = metrics.get('mpg_mse', 0.0)
                        
                        self.hydrophone_table.add_data(
                            epoch,
                            hydrophone,
                            metrics.get('count', 0),
                            mpc_accuracy,
                            mpg_mse
                        )
                    else:
                        self.hydrophone_table.add_data(
                            epoch,
                            hydrophone,
                            metrics.get('count', 0),
                            metrics.get('accuracy', 0.0),
                            metrics.get('precision', 0.0),
                            metrics.get('recall', 0.0),
                            metrics.get('f2', 0.0)
                        )
                
                # Log the updated table
                print("[DEBUG] Logging hydrophone_table to wandb")
                log_training_metrics({
                    "pt_Sample_Distribution": self.hydrophone_table,
                    f"pt_epoch": epoch
                }, use_wandb=self.use_wandb)
                print("[DEBUG] Finished logging hydrophone_table to wandb")
    
    def should_save_best(self, current_value, metric_name='acc'):
        """Check if current metric value is better than best so far."""
        return current_value > self.best_metrics[metric_name]
    
    def get_best_metric(self, metric_name='acc'):
        """Get best metric value."""
        return self.best_metrics[metric_name]

    @staticmethod
    def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
        """Load a checkpoint and restore model and optimizer states.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model: Model to load the state into
            optimizer: Optional optimizer to load the state into
            scheduler: Optional scheduler to load the state into
            
        Returns:
            tuple: (epoch, best_metrics) from the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model state
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint.get('epoch', 0), checkpoint.get('best_metrics', {})

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeterSet:
    """Set of average meters for tracking multiple metrics."""
    def __init__(self):
        self.meters = {}
    
    def update(self, name, value, n=1):
        """Update the meter with new value."""
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)
    
    def reset(self):
        """Reset all meters."""
        for meter in self.meters.values():
            meter.reset()
    
    def get_value(self, name):
        """Get current value of a meter."""
        return self.meters[name].avg if name in self.meters else 0.0
    
    def get_all_values(self):
        """Get all current values as a dictionary."""
        return {name: meter.avg for name, meter in self.meters.items()} 