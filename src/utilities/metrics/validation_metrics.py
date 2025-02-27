"""
Utilities for handling validation metrics and predictions.
"""

import torch
import numpy as np
import os
from torch import nn
from .hydrophone_metrics import calculate_hydrophone_metrics, print_hydrophone_metrics
from ..wandb_utils import log_validation_metrics

# Import calculate_stats from stats module
from ..stats import calculate_stats

# Add binary metrics calculation function
def calculate_binary_metrics(predictions, targets, threshold=0.5):
    """Calculate precision, recall, and F2 score for binary predictions."""
    # Convert to numpy if they are torch tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    
    # Apply threshold to get binary predictions
    binary_preds = (predictions >= threshold).astype(np.int32)
    binary_targets = (targets >= threshold).astype(np.int32)
    
    # Calculate true positives, false positives, false negatives
    tp = np.sum((binary_preds == 1) & (binary_targets == 1))
    fp = np.sum((binary_preds == 1) & (binary_targets == 0))
    fn = np.sum((binary_preds == 0) & (binary_targets == 1))
    
    # Calculate precision, recall, F2
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F2 score gives more weight to recall than precision
    beta = 2
    f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f2

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
        if sources is not None and len(sources) > 0:
            print(f"[DEBUG] ValidationMetricsCollector.update: Received {len(sources)} sources")
            if len(sources) > 0:
                print(f"[DEBUG] First source: {sources[0]}")
                print(f"[DEBUG] Source types: {type(sources[0])}")
                
                # Process sources to extract hydrophone names
                string_sources = []
                for s in sources:
                    if s is None:
                        continue
                        
                    if isinstance(s, bytes):
                        s = s.decode('utf-8')
                    
                    # Extract just the hydrophone name if not already processed
                    if '_' in s:
                        hydrophone_name = s.split('_')[0]
                        string_sources.append(hydrophone_name)
                    else:
                        string_sources.append(s)
                
                self.sources.extend(string_sources)
                print(f"[DEBUG] Total sources collected so far: {len(self.sources)}")
        else:
            print("[DEBUG] ValidationMetricsCollector.update: No sources received")
        
        # Debug the model_output structure
        print(f"[DEBUG] model_output type: {type(model_output)}")
        if isinstance(model_output, tuple):
            print(f"[DEBUG] model_output is a tuple of length {len(model_output)}")
            for i, item in enumerate(model_output):
                print(f"[DEBUG] model_output[{i}] type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'no shape'}")
                if isinstance(item, tuple):
                    print(f"[DEBUG] model_output[{i}] is a nested tuple of length {len(item)}")
                    for j, subitem in enumerate(item):
                        print(f"[DEBUG] model_output[{i}][{j}] type: {type(subitem)}, shape: {subitem.shape if hasattr(subitem, 'shape') else 'no shape'}")
        elif isinstance(model_output, torch.Tensor):
            print(f"[DEBUG] model_output is a tensor with shape: {model_output.shape}, dim: {model_output.dim()}")
        
        if self.task == 'pretrain_mpc':
            acc, nce = model_output
            self.acc_list.append(torch.mean(acc).cpu())
            self.nce_list.append(torch.mean(nce).cpu())
            
            # Store predictions and targets for hydrophone metrics
            predictions = torch.sigmoid(acc).cpu().detach()
            targets = torch.ones_like(predictions)  # In MPC, we're predicting masked tokens
            
            # Ensure predictions is not a scalar tensor
            if predictions.dim() > 0:
                self.predictions.append(predictions)
                self.targets.append(targets)
            else:
                print(f"[DEBUG] Skipping scalar prediction tensor with shape: {predictions.shape}")
            
        elif self.task == 'pretrain_mpg':
            mse = model_output
            self.acc_list.append(torch.mean(mse).cpu())
            self.nce_list.append(torch.mean(mse).cpu())
            
        elif self.task == 'pretrain_joint':
            # For joint training, we need to handle both MPC and MPG outputs
            try:
                # First try to unpack as (mpc_output, mpg_output)
                print(f"[DEBUG] Trying to unpack model_output for pretrain_joint task")
                mpc_output, mpg_output = model_output
                
                print(f"[DEBUG] mpc_output type: {type(mpc_output)}")
                if isinstance(mpc_output, tuple):
                    print(f"[DEBUG] mpc_output is a tuple of length {len(mpc_output)}")
                    # mpc_output is already (acc, nce)
                    acc, nce = mpc_output
                    print(f"[DEBUG] Unpacked mpc_output as (acc, nce)")
                    print(f"[DEBUG] acc shape: {acc.shape if hasattr(acc, 'shape') else 'no shape'}, type: {type(acc)}")
                    print(f"[DEBUG] nce shape: {nce.shape if hasattr(nce, 'shape') else 'no shape'}, type: {type(nce)}")
                else:
                    # If mpc_output is not a tuple, it might be the acc tensor directly
                    print(f"[DEBUG] mpc_output is not a tuple, treating as acc directly")
                    acc = mpc_output
                    print(f"[DEBUG] acc shape: {acc.shape if hasattr(acc, 'shape') else 'no shape'}, type: {type(acc)}")
                
                print(f"[DEBUG] mpg_output type: {type(mpg_output)}")
                if isinstance(mpg_output, torch.Tensor):
                    print(f"[DEBUG] mpg_output is a tensor with shape: {mpg_output.shape if hasattr(mpg_output, 'shape') else 'no shape'}")
                
                # Add to metrics
                self.acc_list.append(torch.mean(acc).cpu())
                self.nce_list.append(torch.mean(mpg_output).cpu())
                
                # Store predictions and targets for hydrophone metrics (using MPC part)
                if hasattr(acc, 'dim') and acc.dim() > 0:
                    print(f"[DEBUG] Creating predictions from acc with shape: {acc.shape}")
                    predictions = torch.sigmoid(acc).cpu().detach()
                    
                    # Debug the shape of predictions
                    print(f"[DEBUG] Predictions shape: {predictions.shape}, dim: {predictions.dim()}")
                    
                    # Ensure predictions is not a scalar tensor
                    if predictions.dim() > 0:
                        targets = torch.ones_like(predictions)  # In MPC, we're predicting masked tokens
                        self.predictions.append(predictions)
                        self.targets.append(targets)
                        print(f"[DEBUG] Successfully added predictions with shape {predictions.shape}")
                    else:
                        print(f"[DEBUG] Skipping scalar prediction tensor with shape: {predictions.shape}")
                else:
                    print(f"[DEBUG] acc is a scalar or doesn't have a dim attribute, cannot create predictions")
                    
                    # Instead of creating dummy predictions, track actual metrics per hydrophone
                    if sources is not None and len(sources) > 0:
                        print(f"[DEBUG] Tracking actual metrics for {len(sources)} sources")
                        
                        # Store the scalar acc and nce values for each source
                        for source in sources:
                            if isinstance(source, bytes):
                                source = source.decode('utf-8')
                            
                            # Extract just the hydrophone name if not already processed
                            if '_' in source:
                                hydrophone_name = source.split('_')[0]
                            else:
                                hydrophone_name = source
                            
                            # Store the source along with the current acc and nce values
                            # We'll use these to calculate per-hydrophone metrics later
                            self.sources.append(hydrophone_name)
                            self.predictions.append((hydrophone_name, torch.mean(acc).cpu().item()))
                            self.targets.append((hydrophone_name, torch.mean(mpg_output).cpu().item()))
                        
                        print(f"[DEBUG] Added metrics for {len(sources)} sources")
            except (ValueError, TypeError) as e:
                # If unpacking fails, model_output might be structured differently
                print(f"[DEBUG] Error unpacking model_output for pretrain_joint: {e}")
                print(f"[DEBUG] model_output type: {type(model_output)}")
                
                # Try to handle it as a scalar or other format
                if isinstance(model_output, torch.Tensor):
                    self.acc_list.append(torch.mean(model_output).cpu())
                    self.nce_list.append(torch.mean(model_output).cpu())
                    print(f"[DEBUG] Handled model_output as a single tensor with shape: {model_output.shape}")
                else:
                    print(f"[DEBUG] Unable to process model_output for pretrain_joint task: {model_output}")
    
    def compute_metrics(self):
        """Compute metrics from collected data."""
        print(f"[DEBUG] ValidationMetricsCollector.compute_metrics: Starting with acc={np.mean(self.acc_list):.6f}, nce={np.mean(self.nce_list):.6f}")
        
        # Basic metrics
        metrics = {
            'acc': np.mean(self.acc_list),
            'nce': np.mean(self.nce_list)
        }
        
        print(f"[DEBUG] Task: {self.task}")
        print(f"[DEBUG] Have sources: {bool(self.sources)}, Have predictions: {bool(self.predictions)}")
        
        # Calculate hydrophone metrics if we have sources and predictions
        if self.sources and self.predictions and self.task in ['pretrain_mpc', 'pretrain_joint', 'pretrain_mpg']:
            try:
                # For pretrain_joint task with scalar outputs, we've stored (hydrophone, value) tuples
                if self.task == 'pretrain_joint' and isinstance(self.predictions[0], tuple):
                    print("[DEBUG] Processing per-hydrophone metrics from scalar values")
                    
                    # Group metrics by hydrophone
                    hydrophone_metrics = {}
                    hydrophone_acc_values = {}
                    hydrophone_nce_values = {}
                    hydrophone_counts = {}
                    
                    # Collect all values per hydrophone
                    for i in range(len(self.predictions)):
                        hydrophone, acc_value = self.predictions[i]
                        _, nce_value = self.targets[i]  # We stored nce in targets
                        
                        if hydrophone not in hydrophone_acc_values:
                            hydrophone_acc_values[hydrophone] = []
                            hydrophone_nce_values[hydrophone] = []
                            hydrophone_counts[hydrophone] = 0
                        
                        hydrophone_acc_values[hydrophone].append(acc_value)
                        hydrophone_nce_values[hydrophone].append(nce_value)
                        hydrophone_counts[hydrophone] += 1
                    
                    # Calculate mean values for each hydrophone
                    for hydrophone in hydrophone_acc_values:
                        metrics_dict = {
                            'count': hydrophone_counts[hydrophone],
                            'accuracy': np.mean(hydrophone_acc_values[hydrophone]),
                            'mpc_accuracy': np.mean(hydrophone_acc_values[hydrophone]),
                            'mpg_mse': np.mean(hydrophone_nce_values[hydrophone])
                        }
                        hydrophone_metrics[hydrophone] = metrics_dict
                    
                    # Add hydrophone metrics to the metrics dictionary
                    metrics.update({
                        'hydrophone_metrics': hydrophone_metrics
                    })
                    
                    print(f"[DEBUG] Computed hydrophone metrics for {len(hydrophone_metrics)} unique hydrophones")
                    return metrics
                
                # For tensor predictions (non-scalar case), use the original approach
                # Check if we have valid predictions to concatenate
                if len(self.predictions) == 0:
                    print("[DEBUG] No predictions collected, skipping hydrophone metrics")
                    return metrics
                
                # Debug the shapes of predictions
                print(f"[DEBUG] Number of prediction tensors: {len(self.predictions)}")
                for i, p in enumerate(self.predictions[:3]):  # Show first 3 for debugging
                    if not isinstance(p, tuple):  # Skip tuples which we handle separately
                        print(f"[DEBUG] Prediction tensor {i} shape: {p.shape}, dim: {p.dim()}")
                
                # Filter out any scalar tensors and tuples
                valid_predictions = [p for p in self.predictions if not isinstance(p, tuple) and p.dim() > 0]
                valid_targets = [t for t in self.targets if not isinstance(t, tuple) and t.dim() > 0]
                valid_sources = [s for i, s in enumerate(self.sources) if not isinstance(self.predictions[i], tuple)]
                
                if not valid_predictions:
                    print("[DEBUG] No valid prediction tensors found, skipping hydrophone metrics")
                    return metrics
                
                print(f"[DEBUG] Valid predictions: {len(valid_predictions)}/{len(self.predictions)}")
                
                # Concatenate valid tensors
                predictions = torch.cat(valid_predictions).numpy()
                targets = torch.cat(valid_targets).numpy()
                
                print(f"[DEBUG] Concatenated predictions shape: {predictions.shape}")
                print(f"[DEBUG] Concatenated targets shape: {targets.shape}")
                print(f"[DEBUG] Number of valid sources: {len(valid_sources)}")
                
                # Ensure we have the same number of predictions and sources
                if len(predictions) != len(valid_sources):
                    print(f"[WARNING] Number of predictions ({len(predictions)}) doesn't match number of sources ({len(valid_sources)})")
                    # Use the minimum length
                    min_len = min(len(predictions), len(valid_sources))
                    predictions = predictions[:min_len]
                    targets = targets[:min_len]
                    sources = valid_sources[:min_len]
                else:
                    sources = valid_sources
                
                # For pretraining tasks, we only want to track accuracy and loss per hydrophone
                if self.task.startswith('pretrain_'):
                    # Group by hydrophone
                    hydrophone_metrics = {}
                    unique_sources = list(set(sources))
                    print(f"[DEBUG] Unique sources: {unique_sources}")
                    
                    for source in unique_sources:
                        source_mask = [s == source for s in sources]
                        source_preds = predictions[source_mask]
                        source_targets = targets[source_mask]
                        
                        metrics_dict = {'count': len(source_preds)}
                        
                        if len(source_preds) > 0:
                            # For pretraining, calculate mean accuracy (how close predictions are to targets)
                            if self.task in ['pretrain_mpc', 'pretrain_joint']:
                                # For MPC, accuracy is how close predictions are to 1.0
                                accuracy = np.mean(source_preds)
                                metrics_dict['accuracy'] = accuracy
                                
                                if self.task == 'pretrain_joint':
                                    # For joint training, also store as mpc_accuracy for clarity
                                    metrics_dict['mpc_accuracy'] = accuracy
                            
                            if self.task in ['pretrain_mpg', 'pretrain_joint']:
                                # For MPG, we're tracking MSE
                                # Since we don't have actual MSE values per sample, use a placeholder
                                # This would ideally be replaced with actual per-hydrophone MSE
                                if self.task == 'pretrain_joint':
                                    metrics_dict['mpg_mse'] = metrics['nce']  # Use global MSE as placeholder
                                else:
                                    metrics_dict['mse'] = metrics['nce']  # Use global MSE as placeholder
                        else:
                            if self.task in ['pretrain_mpc', 'pretrain_joint']:
                                metrics_dict['accuracy'] = 0.0
                                if self.task == 'pretrain_joint':
                                    metrics_dict['mpc_accuracy'] = 0.0
                            
                            if self.task in ['pretrain_mpg', 'pretrain_joint']:
                                if self.task == 'pretrain_joint':
                                    metrics_dict['mpg_mse'] = 0.0
                                else:
                                    metrics_dict['mse'] = 0.0
                        
                        hydrophone_metrics[source] = metrics_dict
                    
                    # Add hydrophone metrics to the metrics dictionary
                    metrics.update({
                        'hydrophone_metrics': hydrophone_metrics
                    })
                    
                    print(f"[DEBUG] Computed hydrophone metrics for {len(hydrophone_metrics)} unique hydrophones")
                else:
                    # For non-pretraining tasks, keep the original binary metrics
                    global_precision, global_recall, global_f2 = calculate_binary_metrics(predictions, targets)
                    
                    # Group by hydrophone
                    hydrophone_metrics = {}
                    unique_sources = list(set(sources))
                    print(f"[DEBUG] Unique sources: {unique_sources}")
                    
                    for source in unique_sources:
                        source_mask = [s == source for s in sources]
                        source_preds = predictions[source_mask]
                        source_targets = targets[source_mask]
                        
                        metrics_dict = {'count': len(source_preds)}
                        
                        if len(source_preds) > 0:
                            precision, recall, f2 = calculate_binary_metrics(source_preds, source_targets)
                            metrics_dict.update({
                                'precision': precision,
                                'recall': recall,
                                'f2': f2
                            })
                        else:
                            metrics_dict.update({
                                'precision': 0.0,
                                'recall': 0.0,
                                'f2': 0.0
                            })
                        
                        hydrophone_metrics[source] = metrics_dict
                    
                    # Add hydrophone metrics to the metrics dictionary
                    metrics.update({
                        'global_precision': global_precision,
                        'global_recall': global_recall,
                        'global_f2': global_f2,
                        'hydrophone_metrics': hydrophone_metrics
                    })
                    
                    print(f"[DEBUG] Computed hydrophone metrics for {len(hydrophone_metrics)} unique hydrophones")
                
            except Exception as e:
                print(f"[ERROR] Failed to compute hydrophone metrics: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return metrics
    
    def log_metrics(self, metrics, epoch=None, prefix="", use_wandb=False):
        """Log validation metrics."""
        # Log global metrics based on task type
        print(f"\nValidation metrics for epoch {epoch}:")
        
        if self.task.startswith('pretrain_'):
            # Pretraining metrics
            print(f"Accuracy: {metrics['acc']:.6f}")
            print(f"NCE/Loss: {metrics['nce']:.6f}")
            
            # Log per-hydrophone metrics for pretraining
            if 'hydrophone_metrics' in metrics and metrics['hydrophone_metrics']:
                print("\nPer-hydrophone metrics:")
                for hydrophone, hyd_metrics in metrics['hydrophone_metrics'].items():
                    print(f"\n{hydrophone}:")
                    print(f"  Sample count: {hyd_metrics['count']}")
                    
                    if self.task in ['pretrain_mpc', 'pretrain_joint']:
                        if 'accuracy' in hyd_metrics:
                            print(f"  Accuracy: {hyd_metrics['accuracy']:.6f}")
                        if 'mpc_accuracy' in hyd_metrics:
                            print(f"  MPC Accuracy: {hyd_metrics['mpc_accuracy']:.6f}")
                    
                    if self.task in ['pretrain_mpg', 'pretrain_joint']:
                        if 'mse' in hyd_metrics:
                            print(f"  MSE: {hyd_metrics['mse']:.6f}")
                        if 'mpg_mse' in hyd_metrics:
                            print(f"  MPG MSE: {hyd_metrics['mpg_mse']:.6f}")
            
            # Log to wandb if enabled
            if use_wandb:
                log_validation_metrics(metrics, self.task, epoch, prefix, use_wandb)
        else:
            # Fine-tuning metrics
            print(f"Global accuracy: {metrics['acc']:.6f}")
            print(f"Global AUC: {metrics['auc']:.6f}")
            print(f"Global precision: {metrics['global_precision']:.6f}")
            print(f"Global recall: {metrics['global_recall']:.6f}")
            print(f"Global F2: {metrics['global_f2']:.6f}")
            print(f"Loss: {metrics['loss']:.6f}")
            
            # Log per-hydrophone metrics for fine-tuning
            if 'hydrophone_metrics' in metrics and metrics['hydrophone_metrics']:
                print("\nPer-hydrophone metrics:")
                for hydrophone, hyd_metrics in metrics['hydrophone_metrics'].items():
                    print(f"\n{hydrophone}:")
                    print(f"  Sample count: {hyd_metrics['count']}")
                    print(f"  Accuracy: {hyd_metrics.get('accuracy', 0.0):.6f}")
                    print(f"  Precision: {hyd_metrics.get('precision', 0.0):.6f}")
                    print(f"  Recall: {hyd_metrics.get('recall', 0.0):.6f}")
                    print(f"  F2: {hyd_metrics.get('f2', 0.0):.6f}")
            
            # Log to wandb if enabled
            if use_wandb:
                log_validation_metrics(metrics, self.task, epoch, prefix, use_wandb)

def validate(audio_model, val_loader, args, epoch):
    """Run validation and compute metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    val_collector = ValidationMetricsCollector(task=args.task)
    
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            labels = labels.to(device)
            
            # Get sources if available
            sources = None
            if hasattr(val_loader.dataset, 'sources'):
                sources = val_loader.dataset.sources[val_loader.dataset.indices[i*val_loader.batch_size:(i+1)*val_loader.batch_size]]
            
            # Forward pass
            audio_output = audio_model(audio_input, args.task)
            
            # Calculate loss
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            
            val_collector.update((audio_output, loss), labels, sources)
    
    # Compute metrics
    metrics = val_collector.compute_metrics()
    
    # Log metrics
    val_collector.log_metrics(metrics, epoch=epoch, prefix="ft_", use_wandb=args.use_wandb)
    
    return metrics

def validate_ensemble(metrics_tracker, predictions, targets, epoch):
    """Calculate ensemble validation metrics."""
    # Get task prefix if available
    task_prefix = ""
    if hasattr(metrics_tracker.args, 'task'):
        task_prefix = f"{metrics_tracker.args.task.replace('_', '-')}_"
    
    if epoch == 1:
        cum_predictions = predictions
    else:
        cum_predictions = np.loadtxt(f"{metrics_tracker.exp_dir}/predictions/{task_prefix}cum_predictions.csv", delimiter=',') * (epoch - 1)
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