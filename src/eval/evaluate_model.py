import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader
import seaborn as sns
from typing import Dict, List, Tuple, Optional
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
    """
    results = []
    unique_sources = np.unique(sources)
    
    for source in unique_sources:
        mask = np.array(sources) == source
        if mask.sum() == 0:
            continue
            
        source_true = y_true[mask]
        source_pred = y_pred[mask]
        
        results.append({
            'hydrophone': source,
            'samples': mask.sum(),
            'anomaly_rate': source_true.mean(),
            'precision': precision_score(source_true, source_pred),
            'recall': recall_score(source_true, source_pred),
            'f1': f1_score(source_true, source_pred),
            'f2': calculate_f2_score(source_true, source_pred)
        })
    
    return pd.DataFrame(results)

def evaluate_by_anomaly_type(
    test_dataset: ONCSpectrogramDataset,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Calculate metrics broken down by anomaly type
    """
    results = []
    
    # Get original multi-class labels
    with h5py.File(test_dataset.data_path, 'r') as hf:
        label_names = [name.decode('utf-8') for name in hf['label_names'][:]]
        
        for idx, sample in enumerate(test_dataset.sample_info):
            original_labels = sample['labels']
            for label_idx, has_anomaly in enumerate(original_labels):
                if has_anomaly:
                    results.append({
                        'anomaly_type': label_names[label_idx],
                        'predicted': y_pred[idx]
                    })
    
    # Calculate metrics per anomaly type
    summary = []
    for anomaly_type in np.unique([r['anomaly_type'] for r in results]):
        type_results = [r['predicted'] for r in results if r['anomaly_type'] == anomaly_type]
        summary.append({
            'anomaly_type': anomaly_type,
            'samples': len(type_results),
            'detection_rate': np.mean(type_results)
        })
    
    return pd.DataFrame(summary)

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
    
    # Debug prints
    print("\nDebug: plot_confusion_matrix")
    print(f"Manual confusion matrix calculation:")
    print(f"TN: {tn}, FP: {fp}")
    print(f"FN: {fn}, TP: {tp}")
    print(f"Total: {tn + fp + fn + tp}")
    print(f"Dataset size: {len(y_true)}")
    
    # Verify with sklearn's confusion_matrix
    cm_sklearn = confusion_matrix(y_true, y_pred)
    print("\nSklearn confusion matrix:")
    print(cm_sklearn)
    
    if not np.array_equal(cm, cm_sklearn):
        print("WARNING: Manual calculation differs from sklearn!")
    
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
            label_string = test_dataset.sample_info[idx]['label_string']
            is_anomalous = test_dataset.sample_info[idx]['is_anomalous']
            raw_labels = test_dataset.sample_info[idx]['labels']
            
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
        label_string = test_dataset.sample_info[idx]['label_string']
        is_anomalous = test_dataset.sample_info[idx]['is_anomalous']
        raw_labels = test_dataset.sample_info[idx]['labels']
        
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
        label_string = test_dataset.sample_info[idx]['label_string']
        is_anomalous = test_dataset.sample_info[idx]['is_anomalous']
        raw_labels = test_dataset.sample_info[idx]['labels']
        
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
    anomaly_metrics = evaluate_by_anomaly_type(test_dataset, y_pred)
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