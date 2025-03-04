"""
Utilities for handling validation metrics and predictions.
"""

import torch
import numpy as np
import os
from torch import nn
import sklearn.metrics
from collections import defaultdict
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
    
    # Calculate metrics using sklearn
    precision = sklearn.metrics.precision_score(binary_targets, binary_preds)
    recall = sklearn.metrics.recall_score(binary_targets, binary_preds)
    f2 = sklearn.metrics.fbeta_score(binary_targets, binary_preds, beta=2)
    
    return precision, recall, f2

class ValidationMetricsCollector:
    """Class to collect and process validation metrics."""
    
    def __init__(self, task=None):
        """Initialize metrics collector."""
        self.reset()
        self.task = task
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.sources = []
        self.acc_list = []
        self.nce_list = []
        self.loss_list = []  # Add loss list for finetuning
        
    def update(self, output, labels, sources=None):
        """Update metrics with a new batch of outputs."""
        print("[DEBUG] Processing batch in ValidationMetricsCollector.update")
        print(f"[DEBUG] Task: {self.task}")
        print(f"[DEBUG] Output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"[DEBUG] Output tuple length: {len(output)}")
            
        if self.task == 'pretrain_joint':
            print("[DEBUG] Processing pretrain_joint task")
            mpc_output, mpg_output = output
            
            # Handle MPC output (acc, nce)
            if isinstance(mpc_output, tuple):
                mpc_acc, mpc_nce = mpc_output
                print(f"[DEBUG] MPC accuracy: {mpc_acc}")
                print(f"[DEBUG] MPC NCE loss: {mpc_nce}")
                
                # Store accuracy
                if isinstance(mpc_acc, torch.Tensor):
                    acc_val = mpc_acc.mean().item()
                    self.acc_list.append(acc_val)
                    # Store raw accuracy tensor for per-hydrophone metrics
                    self.predictions.append(mpc_acc.detach().cpu())
                else:
                    acc_val = float(mpc_acc)
                    self.acc_list.append(acc_val)
                    self.predictions.append(torch.tensor([acc_val]))
                
                # Store NCE loss
                if isinstance(mpc_nce, torch.Tensor):
                    nce_val = mpc_nce.mean().item()
                    self.nce_list.append(nce_val)
                    # Store raw NCE tensor for per-hydrophone metrics
                    self.targets.append(mpc_nce.detach().cpu())
                else:
                    nce_val = float(mpc_nce)
                    self.nce_list.append(nce_val)
                    self.targets.append(torch.tensor([nce_val]))
            
            # Handle MPG output (mse)
            if isinstance(mpg_output, torch.Tensor):
                mpg_mse = mpg_output.mean().item()
            else:
                mpg_mse = float(mpg_output)
            print(f"[DEBUG] MPG MSE loss: {mpg_mse}")
                
        elif self.task == 'pretrain_mpc':
            if isinstance(output, tuple):
                acc, nce = output
                print(f"[DEBUG] MPC accuracy: {acc}")
                print(f"[DEBUG] MPC NCE loss: {nce}")
                
                # Store accuracy
                if isinstance(acc, torch.Tensor):
                    acc_val = acc.mean().item()
                    self.acc_list.append(acc_val)
                    # Store raw accuracy tensor for per-hydrophone metrics
                    self.predictions.append(acc.detach().cpu())
                else:
                    acc_val = float(acc)
                    self.acc_list.append(acc_val)
                    self.predictions.append(torch.tensor([acc_val]))
                
                # Store NCE loss
                if isinstance(nce, torch.Tensor):
                    nce_val = nce.mean().item()
                    self.nce_list.append(nce_val)
                    # Store raw NCE tensor for per-hydrophone metrics
                    self.targets.append(nce.detach().cpu())
                else:
                    nce_val = float(nce)
                    self.nce_list.append(nce_val)
                    self.targets.append(torch.tensor([nce_val]))
                    
        elif self.task == 'pretrain_mpg':
            mse = output
            print(f"[DEBUG] MPG MSE loss: {mse}")
            
            # Store MSE loss
            if isinstance(mse, torch.Tensor):
                mse_val = mse.mean().item()
                self.nce_list.append(mse_val)
                # Store raw MSE tensor for per-hydrophone metrics
                self.predictions.append(mse.detach().cpu())
                self.targets.append(torch.ones_like(mse.detach().cpu()))  # For consistency
            else:
                mse_val = float(mse)
                self.nce_list.append(mse_val)
                self.predictions.append(torch.tensor([mse_val]))
                self.targets.append(torch.tensor([1.0]))  # For consistency
                
        else:
            # For finetuning tasks
            if isinstance(output, tuple):
                predictions, loss = output  # Unpack predictions and loss
                if isinstance(loss, torch.Tensor):
                    self.loss_list.append(loss.item())
                else:
                    self.loss_list.append(float(loss))
            else:
                predictions = output  # Raw logits
            
            print(f"\n[DEBUG] Model output shape: {predictions.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Model output stats (raw logits):")
            print(f"min/max/mean: {predictions.min():.4f}/{predictions.max():.4f}/{predictions.mean():.4f}")
            
            # Convert logits to probabilities for accuracy calculation
            probabilities = torch.sigmoid(predictions)
            print(f"Probabilities stats:")
            print(f"min/max/mean: {probabilities.min():.4f}/{probabilities.max():.4f}/{probabilities.mean():.4f}")
            
            # Calculate accuracy using thresholded probabilities
            predicted_labels = (probabilities >= 0.5).float()
            accuracy = (predicted_labels == labels).float().mean()
            self.acc_list.append(accuracy.item())
            
            # Store raw predictions and labels for later metric computation
            self.predictions.append(predictions.detach().cpu())
            self.targets.append(labels.detach().cpu())
            
            print(f"Batch accuracy: {accuracy.item():.4f}")
            print(f"Predicted anomalies: {predicted_labels.sum().item()}/{len(predicted_labels)}")
            print(f"Actual anomalies: {labels.sum().item()}/{len(predicted_labels)}")
        
        # Store sources if provided
        if sources is not None:
            print(f"[DEBUG] Adding sources to list: {sources[:5]}")
            self.sources.extend(sources)
    
    def compute_metrics(self):
        """Compute metrics from collected predictions and targets."""
        print("\n[DEBUG] Computing validation metrics:")
        print(f"[DEBUG] Task: {self.task}")
        print(f"[DEBUG] Number of predictions: {len(self.predictions)}")
        print(f"[DEBUG] Number of targets: {len(self.targets)}")
        print(f"[DEBUG] Number of sources: {len(self.sources)}")
        print(f"[DEBUG] Number of accuracy values: {len(self.acc_list)}")
        print(f"[DEBUG] Number of NCE values: {len(self.nce_list)}")
        print(f"[DEBUG] Number of loss values: {len(self.loss_list)}")
        
        metrics = {}
        
        # Calculate mean accuracy if available
        if len(self.acc_list) > 0:
            metrics['acc'] = float(np.mean(self.acc_list))
            print(f"[DEBUG] Mean accuracy: {metrics['acc']:.4f}")
        
        # Calculate mean loss/nce if available
        if len(self.nce_list) > 0:
            metrics['loss'] = float(np.mean(self.nce_list))
            if self.task in ['pretrain_mpc', 'pretrain_joint']:
                metrics['nce'] = metrics['loss']  # For MPC tasks, loss is NCE
            print(f"[DEBUG] Mean loss: {metrics['loss']:.4f}")
        elif len(self.loss_list) > 0:  # For finetuning tasks
            metrics['loss'] = float(np.mean(self.loss_list))
            print(f"[DEBUG] Mean loss: {metrics['loss']:.4f}")
        
        # Calculate per-hydrophone metrics if sources are available
        if len(self.sources) > 0 and len(self.predictions) > 0:
            print("\n[DEBUG] Computing per-hydrophone metrics")
            hydrophone_metrics = defaultdict(lambda: {'count': 0, 'values': []})
            
            # Group predictions by hydrophone
            for pred, target, source in zip(self.predictions, self.targets, self.sources):
                # Extract hydrophone name from source
                hydrophone = source.split('_')[0] if isinstance(source, str) else source.decode('utf-8').split('_')[0]
                
                # Store values based on task type
                if self.task == 'pretrain_mpc':
                    hydrophone_metrics[hydrophone]['values'].append({
                        'accuracy': pred.mean().item(),
                        'nce': target.mean().item()
                    })
                elif self.task == 'pretrain_mpg':
                    hydrophone_metrics[hydrophone]['values'].append({
                        'mse': pred.mean().item()
                    })
                elif self.task == 'pretrain_joint':
                    hydrophone_metrics[hydrophone]['values'].append({
                        'mpc_accuracy': pred.mean().item(),
                        'nce': target.mean().item()
                    })
                hydrophone_metrics[hydrophone]['count'] += 1
            
            # Calculate average metrics for each hydrophone
            for hydrophone, hyd_metrics in hydrophone_metrics.items():
                values = hyd_metrics['values']
                if self.task == 'pretrain_mpc':
                    hyd_metrics['accuracy'] = np.mean([v['accuracy'] for v in values])
                    hyd_metrics['nce'] = np.mean([v['nce'] for v in values])
                    print(f"\n[DEBUG] {hydrophone}:")
                    print(f"[DEBUG]   Samples: {hyd_metrics['count']}")
                    print(f"[DEBUG]   Accuracy: {hyd_metrics['accuracy']:.4f}")
                    print(f"[DEBUG]   NCE: {hyd_metrics['nce']:.4f}")
                elif self.task == 'pretrain_mpg':
                    hyd_metrics['mse'] = np.mean([v['mse'] for v in values])
                    print(f"\n[DEBUG] {hydrophone}:")
                    print(f"[DEBUG]   Samples: {hyd_metrics['count']}")
                    print(f"[DEBUG]   MSE: {hyd_metrics['mse']:.4f}")
                elif self.task == 'pretrain_joint':
                    hyd_metrics['mpc_accuracy'] = np.mean([v['mpc_accuracy'] for v in values])
                    hyd_metrics['nce'] = np.mean([v['nce'] for v in values])
                    print(f"\n[DEBUG] {hydrophone}:")
                    print(f"[DEBUG]   Samples: {hyd_metrics['count']}")
                    print(f"[DEBUG]   MPC Accuracy: {hyd_metrics['mpc_accuracy']:.4f}")
                    print(f"[DEBUG]   NCE: {hyd_metrics['nce']:.4f}")
                
                # Remove the values list to save memory
                del hyd_metrics['values']
            
            metrics['hydrophone_metrics'] = dict(hydrophone_metrics)
        
        # For pretraining tasks, we're done here
        if self.task.startswith('pretrain_'):
            return metrics
            
        # For finetuning tasks, compute binary classification metrics
        if len(self.predictions) > 0 and len(self.targets) > 0:
            try:
                # Concatenate all predictions and targets
                all_predictions = torch.cat(self.predictions, dim=0).numpy()  # Raw logits
                all_targets = torch.cat(self.targets, dim=0).numpy()
                print(f"[DEBUG] Total samples: {len(all_predictions)}")
                print(f"[DEBUG] Predictions shape: {all_predictions.shape}")
                print(f"[DEBUG] Targets shape: {all_targets.shape}")
                
                # Convert logits to probabilities
                probabilities = 1 / (1 + np.exp(-all_predictions))  # sigmoid
                print(f"\n[DEBUG] Probabilities stats:")
                print(f"[DEBUG] min/max/mean: {probabilities.min():.4f}/{probabilities.max():.4f}/{probabilities.mean():.4f}")
                
                # Calculate AUC using probabilities
                try:
                    metrics['auc'] = float(sklearn.metrics.roc_auc_score(all_targets, probabilities))
                    print(f"[DEBUG] AUC score: {metrics['auc']:.4f}")
                except Exception as e:
                    print(f"[DEBUG] Warning: Failed to calculate AUC: {str(e)}")
                    metrics['auc'] = 0.0
                
                # Calculate precision, recall, and F2 using thresholded probabilities
                predicted_labels = (probabilities >= 0.5).astype(int)
                metrics['precision'] = float(sklearn.metrics.precision_score(all_targets, predicted_labels))
                metrics['recall'] = float(sklearn.metrics.recall_score(all_targets, predicted_labels))
                metrics['f2'] = float(sklearn.metrics.fbeta_score(all_targets, predicted_labels, beta=2))
                
                print(f"[DEBUG] Precision: {metrics['precision']:.4f}")
                print(f"[DEBUG] Recall: {metrics['recall']:.4f}")
                print(f"[DEBUG] F2 score: {metrics['f2']:.4f}")
                
                # Print confusion matrix
                cm = sklearn.metrics.confusion_matrix(all_targets, predicted_labels)
                print("\n[DEBUG] Confusion Matrix:")
                print("[DEBUG] TN FP")
                print("[DEBUG] FN TP")
                print(f"[DEBUG] {cm}")
                
                # Calculate per-hydrophone metrics if sources are available
                if len(self.sources) > 0:
                    print("\n[DEBUG] Computing per-hydrophone metrics")
                    hydrophone_metrics = defaultdict(lambda: {'count': 0, 'predictions': [], 'targets': []})
                    
                    # Group predictions by hydrophone
                    for pred, target, source in zip(probabilities, all_targets, self.sources):
                        hydrophone_metrics[source]['predictions'].append(pred)
                        hydrophone_metrics[source]['targets'].append(target)
                        hydrophone_metrics[source]['count'] += 1
                    
                    # Calculate metrics for each hydrophone
                    for hydrophone, hyd_metrics in hydrophone_metrics.items():
                        hyd_preds = np.array(hyd_metrics['predictions'])
                        hyd_targets = np.array(hyd_metrics['targets'])
                        hyd_pred_labels = (hyd_preds >= 0.5).astype(int)
                        
                        try:
                            hyd_metrics['precision'] = float(sklearn.metrics.precision_score(hyd_targets, hyd_pred_labels))
                            hyd_metrics['recall'] = float(sklearn.metrics.recall_score(hyd_targets, hyd_pred_labels))
                            hyd_metrics['f2'] = float(sklearn.metrics.fbeta_score(hyd_targets, hyd_pred_labels, beta=2))
                            hyd_metrics['auc'] = float(sklearn.metrics.roc_auc_score(hyd_targets, hyd_preds))
                        except Exception as e:
                            print(f"[DEBUG] Warning: Failed to calculate metrics for hydrophone {hydrophone}: {str(e)}")
                            hyd_metrics.update({'precision': 0.0, 'recall': 0.0, 'f2': 0.0, 'auc': 0.0})
                        
                        # Convert predictions and targets lists to counts for memory efficiency
                        hyd_metrics['predictions'] = len(hyd_metrics['predictions'])
                        hyd_metrics['targets'] = sum(hyd_metrics['targets'])
                    
                    metrics['hydrophone_metrics'] = dict(hydrophone_metrics)
                    
            except Exception as e:
                print(f"[DEBUG] Error computing metrics: {str(e)}")
                print("[DEBUG] Exception details:", e.__class__.__name__)
                import traceback
                print("[DEBUG] Traceback:", traceback.format_exc())
                metrics.update({
                    'auc': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f2': 0.0
                })
        
        return metrics
    
    def log_metrics(self, metrics, epoch=None, prefix="", use_wandb=False):
        """Log validation metrics."""
        # Log global metrics based on task type
        print(f"\nValidation metrics for epoch {epoch}:")
        
        if len(self.acc_list) > 0:
            print(f"Accuracy: {metrics.get('acc', 0.0):.6f}")
            print(f"Loss: {metrics.get('loss', 0.0):.6f}")
            
            # Log per-hydrophone metrics for fine-tuning
            if len(self.sources) > 0 and 'hydrophone_metrics' in metrics:
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
        else:
            print("Warning: No accuracy values collected")

def validate(audio_model, val_loader, args, epoch):
    """Run validation and compute metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    val_collector = ValidationMetricsCollector()
    
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