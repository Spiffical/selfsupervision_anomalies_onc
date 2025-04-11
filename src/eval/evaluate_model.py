import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, fbeta_score
from torch.utils.data import DataLoader
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set
import h5py
from ..onc_dataset import ONCSpectrogramDataset
import logging
from pathlib import Path
import pandas as pd

def calculate_f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F2 score which weighs recall higher than precision
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if precision + recall == 0:
        return 0.0
    
    f2 = 5 * (precision * recall) / (4 * precision + recall)
    return f2

def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
    threshold: float = 0.5,
    task: str = 'ft_cls'
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, List[str]]:
    """
    Evaluate model on test set and return metrics
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        device: Device to run evaluation on
        threshold: Classification threshold
        task: Task for the model
    
    Returns:
        metrics: Dictionary containing various metrics
        y_true: True labels
        y_pred: Predicted labels
        sources: List of hydrophone sources
    """
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    sources = []
    
    with torch.no_grad():
        for data, labels, source in test_loader:
            data = data.to(device)
            outputs = model(data, task=task)
            
            # Get predictions
            scores = torch.sigmoid(outputs).cpu().numpy()
            preds = (scores >= threshold).astype(int)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_scores.extend(scores)
            sources.extend(source)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Ensure y_pred is 1D
    if len(y_pred.shape) > 1:
        y_pred = y_pred.squeeze()
    
    # Calculate confusion matrix
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * (precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate per-class accuracy
    normal_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    anomaly_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'normal_acc': normal_acc,
        'anomaly_acc': anomaly_acc
    }
    
    return metrics, y_true, y_pred, sources

def evaluate_by_hydrophone(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sources: List[str]
) -> pd.DataFrame:
    """
    Calculate metrics broken down by hydrophone
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sources: List of hydrophone sources
        
    Returns:
        DataFrame containing metrics for each hydrophone
    """
    results = []
    unique_sources = np.unique(sources)
    
    for source in unique_sources:
        mask = np.array(sources) == source
        if mask.sum() == 0:
            continue
            
        source_true = y_true[mask]
        source_pred = y_pred[mask]
        
        # Check if we have any samples
        if len(source_true) == 0:
            continue
            
        # Calculate metrics safely
        try:
            # Handle case where all predictions are the same class
            if len(np.unique(source_pred)) == 1:
                pred_class = source_pred[0]
                if pred_class == 0:  # All normal predictions
                    precision = 0.0 if np.any(source_true == 1) else 1.0
                    recall = 0.0
                else:  # All anomaly predictions
                    precision = 1.0 if np.any(source_true == 1) else 0.0
                    recall = 1.0 if np.any(source_true == 1) else 0.0
            else:
                precision = precision_score(source_true, source_pred, zero_division=0)
                recall = recall_score(source_true, source_pred, zero_division=0)
            
            # Calculate F1 and F2 scores
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f2 = 5 * (precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0
            
            results.append({
                'hydrophone': source,
                'samples': mask.sum(),
                'anomaly_rate': float(np.mean(source_true == 1)),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'f2': float(f2),
                'num_normal': int(np.sum(source_true == 0)),
                'num_anomaly': int(np.sum(source_true == 1)),
                'pred_normal': int(np.sum(source_pred == 0)),
                'pred_anomaly': int(np.sum(source_pred == 1))
            })
        except Exception as e:
            print(f"Warning: Error calculating metrics for hydrophone {source}: {str(e)}")
            print(f"Number of samples: {len(source_true)}")
            print(f"True label distribution: {np.unique(source_true, return_counts=True)}")
            print(f"Predicted label distribution: {np.unique(source_pred, return_counts=True)}")
            continue
    
    if not results:
        return pd.DataFrame(columns=[
            'hydrophone', 'samples', 'anomaly_rate', 'precision', 'recall', 
            'f1', 'f2', 'num_normal', 'num_anomaly', 'pred_normal', 'pred_anomaly'
        ])
    
    return pd.DataFrame(results)

def get_sample_anomalies(sample_info: Dict) -> Set[str]:
    """Extract set of anomaly types present in a sample."""
    return set(label for label in sample_info['labels'] if label != 'normal')

def evaluate_by_anomaly_type(
    test_dataset: ONCSpectrogramDataset,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate detailed metrics for each anomaly type.
    
    Args:
        test_dataset: The test dataset
        y_pred: Raw model predictions (before thresholding)
        threshold: Classification threshold
    
    Returns:
        Tuple of (metrics_df, co_occurrence_df):
            - metrics_df: DataFrame with metrics per anomaly type
            - co_occurrence_df: DataFrame with co-occurrence statistics
    """
    # Convert predictions to binary using threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Get all unique anomaly types from the dataset
    anomaly_types = set()
    for sample in test_dataset.sample_info:
        anomalies = get_sample_anomalies(sample)
        anomaly_types.update(anomalies)
    anomaly_types = sorted(list(anomaly_types))
    
    # Track metrics per anomaly type
    type_metrics = {atype: {
        'total_occurrences': 0,
        'solo_occurrences': 0,
        'co_occurrences': 0,
        'true_positives': 0,
        'false_negatives': 0,
        'samples_with_type': []
    } for atype in anomaly_types}
    
    # Track co-occurrences
    co_occurrences = []
    
    # First pass: count occurrences and co-occurrences
    for idx, sample in enumerate(test_dataset.sample_info):
        anomalies = get_sample_anomalies(sample)
        
        # Record co-occurrences
        if len(anomalies) > 1:
            for a1 in anomalies:
                for a2 in anomalies:
                    if a1 < a2:  # Only count each pair once
                        co_occurrences.append((a1, a2))
        
        # Update metrics for each anomaly type
        for atype in anomalies:
            type_metrics[atype]['total_occurrences'] += 1
            type_metrics[atype]['samples_with_type'].append(idx)
            
            if len(anomalies) == 1:
                type_metrics[atype]['solo_occurrences'] += 1
            else:
                type_metrics[atype]['co_occurrences'] += 1
                
            # Track detection success
            if y_pred_binary[idx] == 1:
                type_metrics[atype]['true_positives'] += 1
            else:
                type_metrics[atype]['false_negatives'] += 1
    
    # Calculate metrics for each anomaly type
    results = []
    for atype, metrics in type_metrics.items():
        if metrics['total_occurrences'] > 0:
            # Calculate detection rate
            detection_rate = metrics['true_positives'] / metrics['total_occurrences']
            
            # Calculate solo vs co-occurrence detection rates
            solo_detection_rate = 0
            co_detection_rate = 0
            
            if metrics['solo_occurrences'] > 0:
                solo_indices = [idx for idx in metrics['samples_with_type'] 
                              if len(get_sample_anomalies(test_dataset.sample_info[idx])) == 1]
                solo_detection_rate = np.mean(y_pred_binary[solo_indices])
            
            if metrics['co_occurrences'] > 0:
                co_indices = [idx for idx in metrics['samples_with_type'] 
                            if len(get_sample_anomalies(test_dataset.sample_info[idx])) > 1]
                co_detection_rate = np.mean(y_pred_binary[co_indices])
            
            # Get most common co-occurring anomalies
            co_occurring = []
            for idx in metrics['samples_with_type']:
                anomalies = get_sample_anomalies(test_dataset.sample_info[idx])
                if len(anomalies) > 1:
                    co_occurring.extend(a for a in anomalies if a != atype)
            
            top_co_occurring = pd.Series(co_occurring).value_counts().head(3).to_dict() if co_occurring else {}
            
            results.append({
                'anomaly_type': atype,
                'total_samples': metrics['total_occurrences'],
                'solo_occurrences': metrics['solo_occurrences'],
                'co_occurrences': metrics['co_occurrences'],
                'detection_rate': detection_rate,
                'solo_detection_rate': solo_detection_rate,
                'co_detection_rate': co_detection_rate,
                'top_co_occurring': top_co_occurring
            })
    
    metrics_df = pd.DataFrame(results)
    
    # Create co-occurrence DataFrame
    if co_occurrences:
        co_df = pd.DataFrame(co_occurrences, columns=['anomaly1', 'anomaly2'])
        co_df = co_df.groupby(['anomaly1', 'anomaly2']).size().reset_index(name='count')
        co_df = co_df.sort_values('count', ascending=False)
    else:
        co_df = pd.DataFrame(columns=['anomaly1', 'anomaly2', 'count'])
    
    return metrics_df, co_df

def plot_anomaly_type_metrics(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Create visualization of anomaly type metrics.
    
    Args:
        metrics_df: DataFrame from evaluate_by_anomaly_type
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Sort by total samples for better visualization
    metrics_df = metrics_df.sort_values('total_samples', ascending=True)
    
    # Create bar plot
    y_pos = np.arange(len(metrics_df))
    
    # Plot bars for different detection rates
    overall_bars = plt.barh(y_pos, metrics_df['detection_rate'], 
             label='Overall Detection Rate', alpha=0.3, color='blue')
    solo_bars = plt.barh(y_pos, metrics_df['solo_detection_rate'], 
             label='Solo Detection Rate', alpha=0.3, color='green')
    co_bars = plt.barh(y_pos, metrics_df['co_detection_rate'], 
             label='Co-occurrence Detection Rate', alpha=0.3, color='red')
    
    # Add anomaly type labels
    plt.yticks(y_pos, metrics_df['anomaly_type'])
    
    # Add count information on the left
    for i, row in enumerate(metrics_df.itertuples()):
        plt.text(-0.02, i, f'n={row.total_samples} ({row.solo_occurrences}s+{row.co_occurrences}c)', 
                va='center', ha='right')
    
    # Add percentage annotations at the end of each bar
    for i, row in enumerate(metrics_df.itertuples()):
        # Overall rate
        if row.detection_rate > 0:
            plt.text(row.detection_rate + 0.01, i, 
                    f'{row.detection_rate:.1%} overall',
                    va='center', color='blue')
        
        # Solo rate
        if row.solo_detection_rate > 0:
            plt.text(row.solo_detection_rate + 0.01, i - 0.2,
                    f'{row.solo_detection_rate:.1%} solo',
                    va='center', color='green', fontsize=8)
        
        # Co-occurrence rate
        if row.co_detection_rate > 0:
            plt.text(row.co_detection_rate + 0.01, i + 0.2,
                    f'{row.co_detection_rate:.1%} co-occur',
                    va='center', color='red', fontsize=8)
    
    plt.xlabel('Detection Rate')
    plt.title('Anomaly Detection Performance by Type')
    plt.legend()
    
    # Set x-axis limits to make room for annotations
    plt.xlim(-0.2, 1.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_co_occurrence_matrix(
    test_dataset: ONCSpectrogramDataset,
    save_path: Optional[str] = None
) -> None:
    """
    Create a heatmap visualization of anomaly co-occurrences.
    
    Args:
        test_dataset: The test dataset
        save_path: Optional path to save the plot
    """
    # Get all unique anomaly types from the dataset
    anomaly_types = set()
    for sample in test_dataset.sample_info:
        anomalies = get_sample_anomalies(sample)
        anomaly_types.update(anomalies)
    anomaly_types = sorted(list(anomaly_types))
    
    # Create co-occurrence matrix
    n_types = len(anomaly_types)
    co_matrix = np.zeros((n_types, n_types))
    
    # Fill matrix
    for sample in test_dataset.sample_info:
        anomalies = get_sample_anomalies(sample)
        if len(anomalies) > 1:
            # Add all co-occurrences to matrix
            anomaly_indices = [anomaly_types.index(a) for a in anomalies]
            for i in anomaly_indices:
                for j in anomaly_indices:
                    if i != j:
                        co_matrix[i, j] += 1
    
    # Calculate percentages
    total_co_occurrences = co_matrix.sum()
    percentage_matrix = (co_matrix / total_co_occurrences * 100) if total_co_occurrences > 0 else co_matrix
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with colorbar
    heatmap = sns.heatmap(co_matrix, 
                         xticklabels=anomaly_types,
                         yticklabels=anomaly_types,
                         cmap='YlOrRd',
                         annot=True,
                         fmt='g',  # Use integer format for counts
                         cbar_kws={'label': 'Number of Co-occurrences'})
    
    # Add percentage annotations
    for i in range(n_types):
        for j in range(n_types):
            if co_matrix[i, j] > 0:
                # Get the current text annotation
                text = heatmap.texts[i * n_types + j]
                # Update text to include percentage
                text.set_text(f'{int(co_matrix[i, j])}\n({percentage_matrix[i, j]:.1f}%)')
                # Adjust font size if needed
                text.set_fontsize(8)
    
    plt.title('Anomaly Co-occurrence Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix with category labels (TP, TN, FP, FN) in each square
    """
    # Calculate confusion matrix manually to verify
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    
    
    
    # Create category labels
    category_labels = np.array([
        ['True Negative\n(TN)', 'False Positive\n(FP)'],
        ['False Negative\n(FN)', 'True Positive\n(TP)']
    ])
    
    # Create annotation text combining count with category
    annotations = np.array([
        [f'{cm[0,0]}\n{category_labels[0,0]}', f'{cm[0,1]}\n{category_labels[0,1]}'],
        [f'{cm[1,0]}\n{category_labels[1,0]}', f'{cm[1,1]}\n{category_labels[1,1]}']
    ])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                yticklabels=['Actual Normal', 'Actual Anomaly'])
    
    plt.title('Confusion Matrix', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_example_spectrograms(
    test_dataset: ONCSpectrogramDataset,
    y_pred: np.ndarray,
    num_examples: int = 4,
    samples_per_row: int = 4,  # Control number of samples per row
    save_dir: Optional[str] = None
) -> None:
    """
    Plot example spectrograms for true/false positives/negatives with label strings
    
    Args:
        test_dataset: The test dataset
        y_pred: Model predictions
        num_examples: Number of examples to show for each category
        samples_per_row: Number of samples to show in each row
        save_dir: Optional directory to save the plots
    """
    # Get indices for each category
    # Convert sample_info labels to numpy array for comparison
    true_labels = np.array([sample['is_anomalous'] for sample in test_dataset.sample_info])
    
    # Ensure y_pred is 1D
    if len(y_pred.shape) > 1:
        y_pred = y_pred.squeeze()
    
    # Ensure boolean comparison
    true_pos = np.where((y_pred.astype(bool)) & (true_labels.astype(bool)))[0]
    true_neg = np.where((~y_pred.astype(bool)) & (~true_labels.astype(bool)))[0]
    false_pos = np.where((y_pred.astype(bool)) & (~true_labels.astype(bool)))[0]
    false_neg = np.where((~y_pred.astype(bool)) & (true_labels.astype(bool)))[0]
    
    print("\nCategory sizes:")
    print(f"True Positives: {len(true_pos)}")
    print(f"True Negatives: {len(true_neg)}")
    print(f"False Positives: {len(false_pos)}")
    print(f"False Negatives: {len(false_neg)}")
    
    categories = {
        'True Positive': true_pos,
        'True Negative': true_neg,
        'False Positive': false_pos,
        'False Negative': false_neg
    }
    
    for category, indices in categories.items():
        if len(indices) == 0:
            continue
        
        # Calculate grid dimensions
        n_samples = min(num_examples, len(indices))
        n_rows = (n_samples + samples_per_row - 1) // samples_per_row
        
        # Create figure with a title section at the top
        fig = plt.figure(figsize=(5*samples_per_row, 1 + 3*n_rows))
        
        # Add category header at the top
        fig.suptitle(f'{category} Examples\n(Total: {len(indices)})', 
                    fontsize=16, y=1.0)
        
        for i in range(min(num_examples, len(indices))):
            idx = np.random.choice(indices)
            spec, label, source = test_dataset[idx]
            raw_labels = test_dataset.sample_info[idx]['labels']
            label_string = ';'.join(raw_labels)  # Join the labels with semicolons for display
            is_anomalous = test_dataset.sample_info[idx]['is_anomalous']
            
            plt.subplot(n_rows, samples_per_row, i + 1)
            plt.imshow(spec.numpy()[0], aspect='auto', origin='lower')
            
            # Add title with source and debug info
            title = f'Source: {source}'
            if label_string:
                # Split long label strings into multiple lines
                label_lines = [label_string[i:i+20] for i in range(0, len(label_string), 20)]
                title += '\n' + '\n'.join(label_lines)
            
            # Add debug information to title
            title += f'\nTrue label: {is_anomalous}'
            title += f'\nPred: {y_pred[idx]:.3f}'
            
            plt.title(title, fontsize=8)
            plt.colorbar()
            
            # Add label text directly on the spectrogram in white
            if label_string:
                # Position text at the bottom of the spectrogram
                plt.text(0.02, 0.02, label_string, 
                        color='white', fontsize=6,
                        transform=plt.gca().transAxes,
                        bbox=dict(facecolor='black', alpha=0.7),
                        wrap=True)
        
        plt.tight_layout()
        # Adjust layout to make room for the main title
        plt.subplots_adjust(top=0.94)
        
        if save_dir:
            save_path = Path(save_dir) / f'{category.lower().replace(" ", "_")}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def plot_threshold_f1f2_curve(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
    task: str = 'ft_cls',
    thresholds: np.ndarray = None,
    save_path: Optional[str] = None
) -> Tuple[float, float, float, float]:
    """
    Plot F1 and F2 scores as a function of classification threshold and find optimal thresholds.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        device: Device to run evaluation on
        task: Task for the model
        thresholds: Array of thresholds to evaluate (default: np.linspace(0, 1, 100))
        save_path: Optional path to save the plot
        
    Returns:
        optimal_threshold_f1: Threshold that maximizes F1 score
        optimal_threshold_f2: Threshold that maximizes F2 score
        best_f1: Best F1 score achieved
        best_f2: Best F2 score achieved
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import f1_score
    
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    # Collect all predictions and true labels
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.to(device)
            outputs = model(data, task=task)
            scores = torch.sigmoid(outputs).cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Calculate F1 and F2 scores for each threshold
    f1_scores = []
    f2_scores = []
    for threshold in thresholds:
        predictions = (all_scores >= threshold).astype(int)
        f1 = f1_score(all_labels, predictions)
        f2 = calculate_f2_score(all_labels, predictions)
        f1_scores.append(f1)
        f2_scores.append(f2)
    
    f1_scores = np.array(f1_scores)
    f2_scores = np.array(f2_scores)
    
    # Find optimal thresholds
    optimal_idx_f1 = np.argmax(f1_scores)
    optimal_idx_f2 = np.argmax(f2_scores)
    optimal_threshold_f1 = thresholds[optimal_idx_f1]
    optimal_threshold_f2 = thresholds[optimal_idx_f2]
    best_f1 = f1_scores[optimal_idx_f1]
    best_f2 = f2_scores[optimal_idx_f2]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
    plt.plot(thresholds, f2_scores, 'r-', label='F2 Score')
    
    # Plot optimal points
    plt.plot(optimal_threshold_f1, best_f1, 'bo', label=f'Best F1 = {best_f1:.3f} @ {optimal_threshold_f1:.3f}')
    plt.plot(optimal_threshold_f2, best_f2, 'ro', label=f'Best F2 = {best_f2:.3f} @ {optimal_threshold_f2:.3f}')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('F1 and F2 Scores vs Classification Threshold')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return optimal_threshold_f1, optimal_threshold_f2, best_f1, best_f2

def plot_all_false_positives(
    test_dataset: ONCSpectrogramDataset,
    y_pred: np.ndarray,
    save_dir: Optional[str] = None,
    max_samples: int = 100,  # Maximum number of samples to plot
    samples_per_row: int = 5  # Number of samples to show in each row
) -> None:
    """
    Plot all false positive predictions in a grid layout.
    
    Args:
        test_dataset: The test dataset
        y_pred: Model predictions
        save_dir: Optional directory to save the plots
        max_samples: Maximum number of samples to plot
        samples_per_row: Number of samples to show in each row
    """
    # Ensure y_pred is 1D
    if len(y_pred.shape) > 1:
        y_pred = y_pred.squeeze()
    
    # Get false positives
    true_labels = np.array([sample['is_anomalous'] for sample in test_dataset.sample_info])
    false_pos = np.where((y_pred.astype(bool)) & (~true_labels.astype(bool)))[0]
    
    if len(false_pos) == 0:
        print("No false positives found.")
        return
    
    # Limit number of samples to plot
    n_samples = min(len(false_pos), max_samples)
    indices = false_pos[:n_samples]
    
    # Calculate grid dimensions
    n_rows = (n_samples + samples_per_row - 1) // samples_per_row
    
    # Create figure
    plt.figure(figsize=(4*samples_per_row, 3*n_rows))
    
    for i, idx in enumerate(indices):
        spec, label, source = test_dataset[idx]
        raw_labels = test_dataset.sample_info[idx]['labels']
        label_string = ';'.join(raw_labels)  # Join the labels with semicolons for display
        is_anomalous = test_dataset.sample_info[idx]['is_anomalous']
        
        plt.subplot(n_rows, samples_per_row, i + 1)
        plt.imshow(spec.numpy()[0], aspect='auto', origin='lower')
        
        # Add title with source and debug info
        title = f'Source: {source}'
        if label_string:
            # Split long label strings into multiple lines
            label_lines = [label_string[i:i+20] for i in range(0, len(label_string), 20)]
            title += '\n' + '\n'.join(label_lines)
        
        plt.title(title, fontsize=8)
        plt.colorbar()
        
        # Add prediction info directly on the spectrogram
        plt.text(0.02, 0.02, f'Pred: {y_pred[idx]:.3f}', 
                color='white', fontsize=6,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'all_false_positives.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_all_false_negatives(
    test_dataset: ONCSpectrogramDataset,
    y_pred: np.ndarray,
    save_dir: Optional[str] = None,
    max_samples: int = 100,  # Maximum number of samples to plot
    samples_per_row: int = 5  # Number of samples to show in each row
) -> None:
    """
    Plot all false negative predictions in a grid layout.
    
    Args:
        test_dataset: The test dataset
        y_pred: Model predictions
        save_dir: Optional directory to save the plots
        max_samples: Maximum number of samples to plot
        samples_per_row: Number of samples to show in each row
    """
    # Ensure y_pred is 1D
    if len(y_pred.shape) > 1:
        y_pred = y_pred.squeeze()
    
    # Get false negatives
    true_labels = np.array([sample['is_anomalous'] for sample in test_dataset.sample_info])
    false_neg = np.where((~y_pred.astype(bool)) & (true_labels.astype(bool)))[0]
    
    if len(false_neg) == 0:
        print("No false negatives found.")
        return
    
    # Limit number of samples to plot
    n_samples = min(len(false_neg), max_samples)
    indices = false_neg[:n_samples]
    
    # Calculate grid dimensions
    n_rows = (n_samples + samples_per_row - 1) // samples_per_row
    
    # Create figure
    plt.figure(figsize=(4*samples_per_row, 3*n_rows))
    
    for i, idx in enumerate(indices):
        spec, label, source = test_dataset[idx]
        raw_labels = test_dataset.sample_info[idx]['labels']
        label_string = ';'.join(raw_labels)  # Join the labels with semicolons for display
        is_anomalous = test_dataset.sample_info[idx]['is_anomalous']
        
        plt.subplot(n_rows, samples_per_row, i + 1)
        plt.imshow(spec.numpy()[0], aspect='auto', origin='lower')
        
        # Add title with source and debug info
        title = f'Source: {source}'
        if label_string:
            # Split long label strings into multiple lines
            label_lines = [label_string[i:i+20] for i in range(0, len(label_string), 20)]
            title += '\n' + '\n'.join(label_lines)
        
        plt.title(title, fontsize=8)
        plt.colorbar()
        
        # Add prediction info directly on the spectrogram
        plt.text(0.02, 0.02, f'Pred: {y_pred[idx]:.3f}', 
                color='white', fontsize=6,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'all_false_negatives.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_anomaly_type_by_hydrophone(
    test_dataset: ONCSpectrogramDataset,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    min_samples: int = 5,  # Minimum samples required for a hydrophone to be included
    save_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Analyze and plot anomaly detection performance broken down by both type and hydrophone.
    
    Args:
        test_dataset: The test dataset
        y_pred: Raw model predictions (before thresholding)
        threshold: Classification threshold
        min_samples: Minimum number of samples required for a hydrophone to be included
        save_dir: Optional directory to save the plots (one per hydrophone)
    
    Returns:
        Dictionary mapping hydrophone names to their metrics DataFrames
    """
    # Convert predictions to binary using threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Get all unique anomaly types and hydrophones
    anomaly_types = set()
    hydrophones = set()
    for sample in test_dataset.sample_info:
        anomalies = get_sample_anomalies(sample)
        anomaly_types.update(anomalies)
        if sample['source']:  # Only include if source is available
            hydrophones.add(sample['source'])
    
    anomaly_types = sorted(list(anomaly_types))
    hydrophones = sorted(list(hydrophones))
    
    # Create metrics dictionary for each hydrophone
    hydrophone_metrics = {}
    
    for hydrophone in hydrophones:
        # Initialize metrics for this hydrophone
        type_metrics = {atype: {
            'total_occurrences': 0,
            'solo_occurrences': 0,
            'co_occurrences': 0,
            'true_positives': 0,
            'false_negatives': 0,
            'samples_with_type': []
        } for atype in anomaly_types}
        
        # Get indices for this hydrophone
        hydrophone_indices = [idx for idx, sample in enumerate(test_dataset.sample_info)
                            if sample['source'] == hydrophone]
        
        if len(hydrophone_indices) < min_samples:
            continue
        
        # Process samples for this hydrophone
        for idx in hydrophone_indices:
            sample = test_dataset.sample_info[idx]
            anomalies = get_sample_anomalies(sample)
            
            # Update metrics for each anomaly type
            for atype in anomalies:
                type_metrics[atype]['total_occurrences'] += 1
                type_metrics[atype]['samples_with_type'].append(idx)
                
                if len(anomalies) == 1:
                    type_metrics[atype]['solo_occurrences'] += 1
                else:
                    type_metrics[atype]['co_occurrences'] += 1
                    
                # Track detection success
                if y_pred_binary[idx] == 1:
                    type_metrics[atype]['true_positives'] += 1
                else:
                    type_metrics[atype]['false_negatives'] += 1
        
        # Calculate metrics for each anomaly type
        results = []
        for atype, metrics in type_metrics.items():
            if metrics['total_occurrences'] > 0:
                # Calculate detection rates
                detection_rate = metrics['true_positives'] / metrics['total_occurrences']
                
                # Calculate solo vs co-occurrence detection rates
                solo_detection_rate = 0
                co_detection_rate = 0
                
                if metrics['solo_occurrences'] > 0:
                    solo_indices = [idx for idx in metrics['samples_with_type'] 
                                  if len(get_sample_anomalies(test_dataset.sample_info[idx])) == 1]
                    solo_detection_rate = np.mean(y_pred_binary[solo_indices])
                
                if metrics['co_occurrences'] > 0:
                    co_indices = [idx for idx in metrics['samples_with_type'] 
                                if len(get_sample_anomalies(test_dataset.sample_info[idx])) > 1]
                    co_detection_rate = np.mean(y_pred_binary[co_indices])
                
                results.append({
                    'anomaly_type': atype,
                    'total_samples': metrics['total_occurrences'],
                    'solo_occurrences': metrics['solo_occurrences'],
                    'co_occurrences': metrics['co_occurrences'],
                    'detection_rate': detection_rate,
                    'solo_detection_rate': solo_detection_rate,
                    'co_detection_rate': co_detection_rate
                })
        
        if results:  # Only include hydrophones with anomalies
            metrics_df = pd.DataFrame(results)
            hydrophone_metrics[hydrophone] = metrics_df
            
            # Create plot for this hydrophone
            plt.figure(figsize=(15, 10))
            
            # Sort by total samples
            metrics_df = metrics_df.sort_values('total_samples', ascending=True)
            y_pos = np.arange(len(metrics_df))
            
            # Plot bars
            plt.barh(y_pos, metrics_df['detection_rate'], 
                    label='Overall Detection Rate', alpha=0.3, color='blue')
            plt.barh(y_pos, metrics_df['solo_detection_rate'], 
                    label='Solo Detection Rate', alpha=0.3, color='green')
            plt.barh(y_pos, metrics_df['co_detection_rate'], 
                    label='Co-occurrence Detection Rate', alpha=0.3, color='red')
            
            # Add anomaly type labels
            plt.yticks(y_pos, metrics_df['anomaly_type'])
            
            # Add count information on the left
            for i, row in enumerate(metrics_df.itertuples()):
                plt.text(-0.02, i, f'n={row.total_samples} ({row.solo_occurrences}s+{row.co_occurrences}c)', 
                        va='center', ha='right')
            
            # Add percentage annotations
            for i, row in enumerate(metrics_df.itertuples()):
                # Overall rate
                if row.detection_rate > 0:
                    plt.text(row.detection_rate + 0.01, i, 
                            f'{row.detection_rate:.1%} overall',
                            va='center', color='blue')
                
                # Solo rate
                if row.solo_detection_rate > 0:
                    plt.text(row.solo_detection_rate + 0.01, i - 0.2,
                            f'{row.solo_detection_rate:.1%} solo',
                            va='center', color='green', fontsize=8)
                
                # Co-occurrence rate
                if row.co_detection_rate > 0:
                    plt.text(row.co_detection_rate + 0.01, i + 0.2,
                            f'{row.co_detection_rate:.1%} co-occur',
                            va='center', color='red', fontsize=8)
            
            plt.xlabel('Detection Rate')
            plt.title(f'Anomaly Detection Performance by Type - {hydrophone}')
            plt.legend()
            
            # Set x-axis limits to make room for annotations
            plt.xlim(-0.2, 1.3)
            
            # Adjust layout
            plt.tight_layout()
            
            if save_dir:
                # Save plot
                save_path = Path(save_dir) / f'anomaly_metrics_{hydrophone}.png'
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
            else:
                # Display plot in notebook
                plt.show()
    
    return hydrophone_metrics

def get_dataset_samples(dataset):
    """Helper function to get sample_info from either a single dataset or ConcatDataset"""
    if hasattr(dataset, 'sample_info'):
        return dataset.sample_info
    elif isinstance(dataset, torch.utils.data.ConcatDataset):
        # For ConcatDataset, combine sample_info from all datasets
        all_samples = []
        cumulative_size = 0
        for d in dataset.datasets:
            if hasattr(d, 'sample_info'):
                all_samples.extend(d.sample_info)
            else:
                raise AttributeError(f"Dataset of type {type(d)} has no sample_info attribute")
        return all_samples
    else:
        raise AttributeError(f"Dataset of type {type(dataset)} has no sample_info attribute")

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate various metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities or scores
        threshold: Decision threshold for binary classification
    
    Returns:
        Dictionary containing various metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Convert predictions to binary using threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    
    # Calculate F1 score
    f1 = f1_score(y_true, y_pred_binary)
    
    # Calculate F2 score
    f2 = fbeta_score(y_true, y_pred_binary, beta=2)
    
    # Calculate AUROC only if both classes are present
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        auroc = roc_auc_score(y_true, y_pred)
    else:
        print("Warning: Only one class present in labels. AUROC score cannot be calculated.")
        auroc = None
    
    return {
        'auroc': auroc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2
    }

def analyze_excluded_data(test_dataset, y_pred, excluded_labels):
    """Analyze performance on excluded data.
    
    Args:
        test_dataset: Dataset containing test samples (can be single dataset or ConcatDataset)
        y_pred: Model predictions
        excluded_labels: List of labels that were excluded during training
        
    Returns:
        Dictionary containing metrics and detailed analysis
    """
    try:
        # Get samples using the helper function that handles both single datasets and ConcatDatasets
        samples = get_dataset_samples(test_dataset)
    except AttributeError as e:
        print(f"Error getting samples: {e}")
        return None
    
    # Initialize results dictionaries
    overall_metrics = {'total_samples': 0, 'total_detected': 0}
    detailed_results = {label: {'solo': [], 'with_other_excluded': [], 'with_non_excluded': []} 
                       for label in excluded_labels}
    excluded_samples = []
    
    # Analyze each sample
    for idx, sample in enumerate(samples):
        sample_labels = sample['labels']
        excluded_in_sample = [label for label in excluded_labels if label in sample_labels]
        
        if not excluded_in_sample:
            continue
            
        # Add to excluded samples list
        sample_info = {
            'index': sample['index'],
            'labels': sample_labels,
            'prediction': y_pred[idx],
            'source': sample.get('source', None)
        }
        excluded_samples.append(sample_info)
        overall_metrics['total_samples'] += 1
        
        # Categorize the sample for each excluded label it contains
        for excluded_label in excluded_in_sample:
            other_labels = set(sample_labels) - {excluded_label}
            other_excluded = any(label in other_labels for label in excluded_labels)
            other_non_excluded = any(label not in excluded_labels for label in other_labels)
            
            if not other_labels:  # Solo label
                detailed_results[excluded_label]['solo'].append(sample_info)
            elif other_excluded and not other_non_excluded:  # Only other excluded labels
                detailed_results[excluded_label]['with_other_excluded'].append(sample_info)
            else:  # Has non-excluded labels
                detailed_results[excluded_label]['with_non_excluded'].append(sample_info)
    
    # Calculate overall metrics
    if excluded_samples:
        excluded_preds = np.array([sample['prediction'] for sample in excluded_samples])
        overall_metrics.update(calculate_metrics(
            np.ones(len(excluded_samples)),  # All samples should be detected as anomalous
            excluded_preds
        ))
    
    # Calculate metrics for each excluded label
    for label in excluded_labels:
        for category in ['solo', 'with_other_excluded', 'with_non_excluded']:
            samples = detailed_results[label][category]
            if samples:
                preds = np.array([s['prediction'] for s in samples])
                metrics = calculate_metrics(
                    np.ones(len(samples)),  # All should be detected
                    preds
                )
                detailed_results[label][f'{category}_metrics'] = metrics
                detailed_results[label][f'{category}_count'] = len(samples)
            else:
                detailed_results[label][f'{category}_metrics'] = None
                detailed_results[label][f'{category}_count'] = 0
    
    # Display the analysis
    display_excluded_analysis(overall_metrics, detailed_results, excluded_labels)
    
    return {
        'overall_metrics': overall_metrics,
        'detailed_results': detailed_results,
        'excluded_samples': excluded_samples,
        'num_samples': len(excluded_samples)
    }

def display_excluded_analysis(overall_metrics, detailed_results, excluded_labels):
    """Display the analysis of excluded data performance.
    
    Args:
        overall_metrics: Dictionary containing overall performance metrics
        detailed_results: Dictionary containing detailed metrics by label and category
        excluded_labels: List of labels that were excluded during training
    """
    print("\n" + "="*80)
    print("EXCLUDED LABELS ANALYSIS")
    print("="*80)
    
    print("\nOVERALL PERFORMANCE ON EXCLUDED SAMPLES:")
    print(f"Total samples with excluded labels: {overall_metrics['total_samples']}")
    if 'precision' in overall_metrics:
        print(f"Precision: {overall_metrics['precision']:.4f}")
        print(f"Recall: {overall_metrics['recall']:.4f}")
        print(f"F1 Score: {overall_metrics['f1']:.4f}")
        print(f"F2 Score: {overall_metrics['f2']:.4f}")
        if overall_metrics.get('auroc') is not None:
            print(f"AUROC: {overall_metrics['auroc']:.4f}")
        else:
            print("AUROC: Not applicable (only one class present)")
    
    # Print detailed analysis for each excluded label
    for label in excluded_labels:
        print(f"\n{'-'*40}")
        print(f"DETAILED ANALYSIS FOR: {label}")
        print(f"{'-'*40}")
        
        # Solo occurrences
        solo_count = len(detailed_results[label]['solo'])
        print(f"\nAppearing alone: {solo_count} samples")
        if solo_count > 0 and detailed_results[label]['solo_metrics']:
            metrics = detailed_results[label]['solo_metrics']
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  F2 Score: {metrics['f2']:.4f}")
        
        # With other excluded labels
        other_count = len(detailed_results[label]['with_other_excluded'])
        print(f"\nWith other excluded labels: {other_count} samples")
        if other_count > 0 and detailed_results[label]['with_other_excluded_metrics']:
            metrics = detailed_results[label]['with_other_excluded_metrics']
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  F2 Score: {metrics['f2']:.4f}")
        
        # With non-excluded labels
        non_excluded_count = len(detailed_results[label]['with_non_excluded'])
        print(f"\nWith non-excluded labels: {non_excluded_count} samples")
        if non_excluded_count > 0 and detailed_results[label]['with_non_excluded_metrics']:
            metrics = detailed_results[label]['with_non_excluded_metrics']
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  F2 Score: {metrics['f2']:.4f}")
    
    print("\n" + "="*80)

def analyze_full_test_set(test_dataset, y_pred, excluded_labels=None):
    """Analyze performance on the full test set, with optional breakdown by excluded vs non-excluded samples"""
    samples = get_dataset_samples(test_dataset)
    
    # Separate samples into normal, excluded anomalies, and other anomalies
    normal_samples = []
    excluded_anomaly_samples = []
    other_anomaly_samples = []
    
    for idx, sample in enumerate(samples):
        anomalies = set(label for label in sample['labels'] if label != 'normal')
        is_normal = len(anomalies) == 0
        
        sample_data = {
            'index': idx,
            'labels': sample['labels'],
            'prediction': y_pred[idx],
            'source': sample['source']
        }
        
        if is_normal:
            normal_samples.append(sample_data)
        elif excluded_labels and any(label in excluded_labels for label in anomalies):
            excluded_anomaly_samples.append(sample_data)
        else:
            other_anomaly_samples.append(sample_data)
    
    # Calculate overall metrics
    y_true = []
    y_pred_list = []
    for samples in [normal_samples, excluded_anomaly_samples, other_anomaly_samples]:
        for sample in samples:
            y_true.append(1 if sample in excluded_anomaly_samples + other_anomaly_samples else 0)
            y_pred_list.append(sample['prediction'])
    
    overall_metrics = calculate_metrics(y_true, y_pred_list)
    
    print("\nFull Test Set Analysis:")
    print(f"Total samples: {len(samples)}")
    print(f"Normal samples: {len(normal_samples)}")
    print(f"Anomalous samples (non-excluded): {len(other_anomaly_samples)}")
    if excluded_labels:
        print(f"Anomalous samples (excluded types): {len(excluded_anomaly_samples)}")
    
    print("\nOverall Metrics:")
    print(f"AUROC: {overall_metrics['auroc']:.4f}")
    print(f"F1 Score: {overall_metrics['f1']:.4f}")
    print(f"F2 Score: {overall_metrics['f2']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    
    # Calculate metrics for different subsets
    results = {
        'overall_metrics': overall_metrics,
        'sample_counts': {
            'normal': len(normal_samples),
            'anomalous_non_excluded': len(other_anomaly_samples),
            'anomalous_excluded': len(excluded_anomaly_samples),
            'total': len(samples)
        }
    }
    
    # Add subset metrics if we have excluded labels
    if excluded_labels:
        # Metrics for non-excluded samples only
        non_excluded_y_true = []
        non_excluded_y_pred = []
        for sample in normal_samples + other_anomaly_samples:
            non_excluded_y_true.append(1 if sample in other_anomaly_samples else 0)
            non_excluded_y_pred.append(sample['prediction'])
        
        non_excluded_metrics = calculate_metrics(non_excluded_y_true, non_excluded_y_pred)
        results['non_excluded_metrics'] = non_excluded_metrics
        
        print("\nMetrics (Non-excluded Samples Only):")
        print(f"AUROC: {non_excluded_metrics['auroc']:.4f}")
        print(f"F1 Score: {non_excluded_metrics['f1']:.4f}")
        print(f"F2 Score: {non_excluded_metrics['f2']:.4f}")
        print(f"Precision: {non_excluded_metrics['precision']:.4f}")
        print(f"Recall: {non_excluded_metrics['recall']:.4f}")
    
    return results

def main(
    model_path: str,
    data_path: str,
    output_dir: str,
    batch_size: int = 32,
    device: str = 'cuda',
    threshold: float = 0.5
):
    """
    Main evaluation function
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        filename=output_dir / 'evaluation.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # Load model
    model = torch.load(model_path)
    model = model.to(device)
    
    # Create test dataset and loader
    test_dataset = ONCSpectrogramDataset(
        data_path=data_path,
        split='test',
        supervised=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate model
    metrics, y_true, y_pred, sources = evaluate_model(
        model,
        test_loader,
        device,
        threshold
    )
    
    # Log overall metrics
    logging.info("Overall Metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Evaluate by hydrophone
    hydrophone_metrics = evaluate_by_hydrophone(y_true, y_pred, sources)
    hydrophone_metrics.to_csv(output_dir / 'hydrophone_metrics.csv')
    logging.info("\nMetrics by Hydrophone:")
    logging.info(hydrophone_metrics.to_string())
    
    # Evaluate by anomaly type
    anomaly_metrics, co_occurrence_metrics = evaluate_by_anomaly_type(test_dataset, y_pred)
    anomaly_metrics.to_csv(output_dir / 'anomaly_metrics.csv')
    logging.info("\nMetrics by Anomaly Type:")
    logging.info(anomaly_metrics.to_string())
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, output_dir / 'confusion_matrix.png')
    
    # Plot example spectrograms
    plot_example_spectrograms(
        test_dataset,
        y_pred,
        num_examples=4,
        save_dir=output_dir / 'examples'
    )

    # Plot all false positives
    plot_all_false_positives(
        test_dataset,
        y_pred,
        save_dir=output_dir / 'false_positives'
    )

    # Plot all false negatives
    plot_all_false_negatives(
        test_dataset,
        y_pred,
        save_dir=output_dir / 'false_negatives'
    )

    # Plot anomaly type metrics
    plot_anomaly_type_metrics(anomaly_metrics, output_dir / 'anomaly_type_metrics.png')

    # Plot co-occurrence matrix
    plot_co_occurrence_matrix(test_dataset, output_dir / 'co_occurrence_matrix.png')

    # Plot anomaly type by hydrophone
    plot_anomaly_type_by_hydrophone(test_dataset, y_pred, threshold, min_samples=5, save_dir=output_dir / 'anomaly_type_by_hydrophone')

    # # First analysis: Just the excluded data
    # excluded_metrics = analyze_excluded_data(test_dataset, y_pred, excluded_labels)
    # print("\nAnalysis of Excluded Data Only:")
    # print(excluded_metrics)

    # # Second analysis: Full test set
    # full_metrics = analyze_full_test_set(test_dataset, y_pred, excluded_labels)
    # print("\nAnalysis of Full Test Set:")
    # print(full_metrics)

    # # First analysis: Just the excluded data
    # excluded_metrics = analyze_excluded_data(test_dataset, y_pred, excluded_labels)
    # plot_excluded_analysis(excluded_metrics, save_path='excluded_analysis.png')

    # # Second analysis: Full test set
    # full_metrics = analyze_full_test_set(test_dataset, y_pred, excluded_labels)
    # plot_full_test_analysis(full_metrics, save_path='full_test_analysis.png')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--model-path', required=True, help='Path to saved model')
    parser.add_argument('--data-path', required=True, help='Path to HDF5 dataset')
    parser.add_argument('--output-dir', required=True, help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device to run evaluation on')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    args = parser.parse_args()
    main(**vars(args))