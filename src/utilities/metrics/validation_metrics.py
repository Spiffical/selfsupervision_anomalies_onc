"""
Utilities for handling validation metrics and predictions.
"""

import torch
import numpy as np
import os
from .hydrophone_metrics import calculate_hydrophone_metrics, print_hydrophone_metrics, log_hydrophone_metrics_to_wandb
import wandb

class ValidationMetricsCollector:
    """Class to collect and process validation metrics."""
    def __init__(self, task='pretrain_mpc'):
        self.task = task
        self.reset()
    
    def reset(self):
        """Reset all collected metrics."""
        self.acc_list = []
        self.nce_list = []
        self.predictions = []
        self.targets = []
        self.sources = []
    
    def update(self, model_output, audio_input=None, sources=None):
        """Update metrics with new batch results."""
        if self.task == 'pretrain_mpc':
            acc, nce = model_output
            self.acc_list.append(torch.mean(acc).cpu())
            self.nce_list.append(torch.mean(nce).cpu())
            
            # Store predictions and targets for hydrophone metrics
            predictions = torch.sigmoid(acc).cpu().detach()
            targets = torch.ones_like(predictions)  # In MPC, we're predicting masked tokens
            self.predictions.append(predictions)
            self.targets.append(targets)
            
            if sources is not None:
                self.sources.extend(sources)
                
        elif self.task == 'pretrain_mpg':
            mse = model_output
            self.acc_list.append(torch.mean(mse).cpu())
            self.nce_list.append(torch.mean(mse).cpu())
            
        elif self.task == 'pretrain_joint':
            acc, _ = model_output[0]  # MPC output
            mse = model_output[1]     # MPG output
            self.acc_list.append(torch.mean(acc).cpu())
            self.nce_list.append(torch.mean(mse).cpu())
    
    def compute_metrics(self):
        """Compute final metrics from collected data."""
        acc = np.mean(self.acc_list)
        nce = np.mean(self.nce_list)
        
        # Initialize metrics
        metrics = {
            'acc': acc,
            'nce': nce,
            'hydrophone_metrics': {}
        }
        
        # For pre-training tasks, track appropriate metrics
        if self.task in ['pretrain_mpc', 'pretrain_mpg', 'pretrain_joint']:
            if self.sources and self.predictions:
                predictions = torch.cat(self.predictions).numpy()
                targets = torch.cat(self.targets).numpy()
                
                # Group by hydrophone
                hydrophone_metrics = {}
                unique_sources = list(set(self.sources))
                
                for source in unique_sources:
                    source_mask = [s == source for s in self.sources]
                    source_preds = predictions[source_mask]
                    source_targets = targets[source_mask]
                    
                    metrics_dict = {'count': len(source_preds)}
                    
                    if self.task == 'pretrain_mpc':
                        # For MPC, track patch prediction accuracy
                        accuracy = np.mean((source_preds > 0.5) == source_targets)
                        metrics_dict['accuracy'] = accuracy
                    elif self.task == 'pretrain_mpg':
                        # For MPG, track patch localization MSE
                        mse = np.mean((source_preds - source_targets) ** 2)
                        metrics_dict['mse'] = mse
                    elif self.task == 'pretrain_joint':
                        # For joint training, track both metrics
                        mpc_accuracy = np.mean((source_preds > 0.5) == source_targets)
                        mpg_mse = np.mean((source_preds - source_targets) ** 2)
                        metrics_dict.update({
                            'mpc_accuracy': mpc_accuracy,
                            'mpg_mse': mpg_mse
                        })
                    
                    hydrophone_metrics[source] = metrics_dict
                
                metrics['hydrophone_metrics'] = hydrophone_metrics
        
        # For fine-tuning, include classification metrics
        else:
            if self.sources and self.predictions:
                predictions = torch.cat(self.predictions).numpy()
                targets = torch.cat(self.targets).numpy()
                global_precision, global_recall, global_f2, hydrophone_metrics = calculate_hydrophone_metrics(
                    predictions, targets, self.sources
                )
                metrics.update({
                    'global_precision': global_precision,
                    'global_recall': global_recall,
                    'global_f2': global_f2,
                    'hydrophone_metrics': hydrophone_metrics
                })
        
        return metrics
    
    def log_metrics(self, metrics, epoch=None, prefix="", use_wandb=False):
        """Log validation metrics."""
        # Log global metrics based on task type
        print(f"\nValidation metrics for epoch {epoch}:")
        
        if self.task.startswith('pretrain_'):
            # Pretraining metrics
            print(f"Accuracy: {metrics['acc']:.6f}")
            print(f"NCE/Loss: {metrics['nce']:.6f}")
            
            # Log to wandb if enabled
            if use_wandb:
                wandb_metrics = {
                    f"{prefix}val_accuracy": metrics['acc'],
                    f"{prefix}val_loss": metrics['nce'],
                    f"{prefix}epoch": epoch if epoch is not None else 0
                }
                
                # Add per-hydrophone metrics for pretraining if available
                if metrics['hydrophone_metrics']:
                    for hydrophone, hyd_metrics in metrics['hydrophone_metrics'].items():
                        for metric_name, value in hyd_metrics.items():
                            if self.task == 'pretrain_mpc':
                                if metric_name in ['accuracy']:
                                    wandb_metrics[f"{prefix}hydrophone/{hydrophone}/val_{metric_name}"] = value
                            elif self.task == 'pretrain_mpg':
                                if metric_name in ['mse']:
                                    wandb_metrics[f"{prefix}hydrophone/{hydrophone}/val_{metric_name}"] = value
                            elif self.task == 'pretrain_joint':
                                if metric_name in ['mpc_accuracy', 'mpg_mse']:
                                    wandb_metrics[f"{prefix}hydrophone/{hydrophone}/val_{metric_name}"] = value
                            
                            if metric_name == 'count':
                                wandb_metrics[f"{prefix}hydrophone/{hydrophone}/sample_count"] = value
                
                wandb.log(wandb_metrics, step=epoch if epoch is not None else None)
        else:
            # Fine-tuning metrics
            print(f"Global accuracy: {metrics['acc']:.6f}")
            print(f"Global AUC: {metrics['auc']:.6f}")
            print(f"Global precision: {metrics['global_precision']:.6f}")
            print(f"Global recall: {metrics['global_recall']:.6f}")
            print(f"Global F2: {metrics['global_f2']:.6f}")
            print(f"Loss: {metrics['loss']:.6f}")
            
            # Log per-hydrophone metrics for fine-tuning
            if metrics['hydrophone_metrics']:
                print("\nPer-hydrophone metrics:")
                for hydrophone, hyd_metrics in metrics['hydrophone_metrics'].items():
                    print(f"\n{hydrophone}:")
                    print(f"  Sample count: {hyd_metrics['count']}")
                    print(f"  Accuracy: {hyd_metrics['accuracy']:.6f}")
                    print(f"  Precision: {hyd_metrics['precision']:.6f}")
                    print(f"  Recall: {hyd_metrics['recall']:.6f}")
                    print(f"  F2: {hyd_metrics['f2']:.6f}")
            
            # Log to wandb if enabled
            if use_wandb:
                wandb_metrics = {
                    f"{prefix}val_loss": metrics['loss'],
                    f"{prefix}val_accuracy": metrics['acc'],
                    f"{prefix}val_auc": metrics['auc'],
                    f"{prefix}val_precision": metrics['global_precision'],
                    f"{prefix}val_recall": metrics['global_recall'],
                    f"{prefix}val_f2": metrics['global_f2']
                }
                
                # Add per-hydrophone metrics
                if metrics['hydrophone_metrics']:
                    for hydrophone, hyd_metrics in metrics['hydrophone_metrics'].items():
                        for metric_name, value in hyd_metrics.items():
                            if metric_name in ['accuracy', 'precision', 'recall', 'f2']:
                                wandb_metrics[f"{prefix}hydrophone/{hydrophone}/val_{metric_name}"] = value
                            elif metric_name == 'count':
                                wandb_metrics[f"{prefix}hydrophone/{hydrophone}/sample_count"] = value
                
                wandb.log(wandb_metrics, step=epoch if epoch is not None else None)

def validate_ensemble(metrics_tracker, predictions, targets, epoch):
    """Calculate ensemble validation metrics."""
    if epoch == 1:
        cum_predictions = predictions
    else:
        cum_predictions = np.loadtxt(f"{metrics_tracker.exp_dir}/predictions/cum_predictions.csv", delimiter=',') * (epoch - 1)
        cum_predictions = (cum_predictions + predictions) / epoch
    
    # Save cumulative predictions
    metrics_tracker.save_predictions(cum_predictions, epoch, is_cumulative=True)
    
    # Calculate stats
    stats = calculate_stats(cum_predictions, targets)
    return stats 

def validate_wa(audio_model, val_loader, args, start_epoch, end_epoch):
    """Run validation with weighted average model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = args.exp_dir

    # Load and average model states
    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location=device)
    model_cnt = 1
    
    for epoch in range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location=device)
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1

        # Remove model file if not saving
        if not args.save_model:
            os.remove(exp_dir + '/models/audio_model.' + str(epoch) + '.pth')

    # Average weights
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)

    # Load averaged weights and validate
    audio_model.load_state_dict(sdA)
    torch.save(audio_model.state_dict(), exp_dir + '/models/audio_model_wa.pth')
    
    metrics = validate(audio_model, val_loader, args, 'wa')
    return metrics 