"""
Utilities for calculating and tracking hydrophone-specific metrics.
"""

import numpy as np
import torch
import wandb
from collections import defaultdict

def extract_hydrophone(source):
    """Extract hydrophone name from source string."""
    # Convert bytes to string if needed
    if isinstance(source, bytes):
        source = source.decode('utf-8')
    # Extract the hydrophone name (everything before the first underscore)
    return source.split('_')[0]

def calculate_binary_metrics(predictions, targets, threshold=0.5):
    """Calculate binary classification metrics."""
    predictions = predictions > threshold
    targets = targets > 0.5
    
    tp = np.sum((predictions == 1) & (targets == 1))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F2 score (beta=2 puts more emphasis on recall)
    beta = 2
    f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f2

def calculate_hydrophone_metrics(predictions, targets, sources):
    """Calculate metrics globally and per hydrophone."""
    # Global metrics
    global_precision, global_recall, global_f2 = calculate_binary_metrics(predictions, targets)
    
    # Per hydrophone metrics
    hydrophone_metrics = {}
    unique_hydrophones = np.unique([extract_hydrophone(s) for s in sources])
    
    for hydrophone in unique_hydrophones:
        # Create mask for this hydrophone
        mask = np.array([extract_hydrophone(s) == hydrophone for s in sources])
        if np.sum(mask) > 0:  # Only calculate if we have samples for this hydrophone
            h_pred = predictions[mask]
            h_targets = targets[mask]
            h_precision, h_recall, h_f2 = calculate_binary_metrics(h_pred, h_targets)
            hydrophone_metrics[hydrophone] = {
                'precision': h_precision,
                'recall': h_recall,
                'f2': h_f2,
                'count': np.sum(mask)
            }
    
    return global_precision, global_recall, global_f2, hydrophone_metrics

def log_hydrophone_metrics_to_wandb(metrics_dict, hydrophone_metrics, prefix="", epoch=None):
    """Log hydrophone metrics to wandb with proper organization."""
    # Log global metrics
    wandb.log(metrics_dict)
    
    # Log per-hydrophone metrics
    for hydrophone, metrics in hydrophone_metrics.items():
        wandb.log({
            f"{prefix}Precision/{hydrophone}": metrics['precision'],
            f"{prefix}Recall/{hydrophone}": metrics['recall'],
            f"{prefix}F2/{hydrophone}": metrics['f2'],
            f"{prefix}Sample_Count/{hydrophone}": metrics['count']
        })
    
    # Create custom wandb.Table for sample distribution periodically
    if isinstance(epoch, int) and (epoch == 1 or epoch % 10 == 0):
        table_data = [[hydrophone, metrics['count']] for hydrophone, metrics in hydrophone_metrics.items()]
        wandb.log({
            f"{prefix}Sample_Distribution": wandb.Table(
                data=table_data,
                columns=["Hydrophone", "Sample Count"]
            )
        })

def print_hydrophone_metrics(hydrophone_metrics, global_precision=None, global_recall=None, global_f2=None):
    """Print formatted hydrophone metrics to console."""
    if global_precision is not None:
        print("\nGlobal Metrics:")
        print(f"Precision: {global_precision:.4f}")
        print(f"Recall: {global_recall:.4f}")
        print(f"F2: {global_f2:.4f}")
    
    if hydrophone_metrics:
        print("\nPer-Hydrophone Metrics:")
        for hydrophone, metrics in hydrophone_metrics.items():
            print(f"\n{hydrophone}:")
            print(f"  Samples: {metrics['count']}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F2: {metrics['f2']:.4f}") 