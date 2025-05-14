import numpy as np
from scipy import stats
from sklearn import metrics
import torch
import h5py
from tqdm import tqdm

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc
                }
        stats.append(dict)

    return stats

def calculate_dataset_stats(h5_path, batch_size=100):
    """
    Calculate mean and standard deviation of a HDF5 dataset using online/streaming calculations.
    Uses Welford's online algorithm for numerical stability.
    
    Args:
        h5_path: Path to HDF5 file
        batch_size: Number of samples to process at once
    
    Returns:
        mean: Mean value of the dataset
        std: Standard deviation of the dataset
    """
    with h5py.File(h5_path, 'r') as f:
        spectrograms = f['spectrograms']
        total_samples = len(spectrograms)
        
        # Initialize variables for Welford's online algorithm
        mean = 0
        M2 = 0  # Sum of squared distances from mean
        count = 0
        
        # Process data in batches
        num_batches = (total_samples + batch_size - 1) // batch_size
        with tqdm(total=num_batches, desc="Calculating dataset statistics") as pbar:
            for i in range(0, total_samples, batch_size):
                batch = spectrograms[i:min(i + batch_size, total_samples)][:]
                batch_size_actual = batch.shape[0]
                
                # Flatten all dimensions except the batch dimension
                batch_flat = batch.reshape(batch_size_actual, -1)
                
                # Update running statistics for each sample in the batch
                for x in batch_flat:
                    count += 1
                    delta = x - mean
                    mean += delta / count
                    delta2 = x - mean
                    M2 += delta * delta2
                
                pbar.update(1)
        
        # Calculate final statistics
        std = np.sqrt(M2 / (count - 1))
        
        # Average across all dimensions
        mean = float(np.mean(mean))
        std = float(np.mean(std))
        
        print(f"\nDataset statistics:")
        print(f"Mean: {mean:.6f}")
        print(f"Std:  {std:.6f}")
        
        return mean, std

