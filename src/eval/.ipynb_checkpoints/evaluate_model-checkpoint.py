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
    
    # Calculate metrics
    metrics = {
        'accuracy': (y_true == y_pred).mean(),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'f2': calculate_f2_score(y_true, y_pred)
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
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_example_spectrograms(
    test_dataset: ONCSpectrogramDataset,
    y_pred: np.ndarray,
    num_examples: int = 4,
    save_dir: Optional[str] = None
) -> None:
    """
    Plot example spectrograms for true/false positives/negatives
    """
    # Get indices for each category
    true_pos = np.where((y_pred == 1) & (test_dataset.sample_info['labels'] == 1))[0]
    true_neg = np.where((y_pred == 0) & (test_dataset.sample_info['labels'] == 0))[0]
    false_pos = np.where((y_pred == 1) & (test_dataset.sample_info['labels'] == 0))[0]
    false_neg = np.where((y_pred == 0) & (test_dataset.sample_info['labels'] == 1))[0]
    
    categories = {
        'True Positive': true_pos,
        'True Negative': true_neg,
        'False Positive': false_pos,
        'False Negative': false_neg
    }
    
    for category, indices in categories.items():
        if len(indices) == 0:
            continue
            
        plt.figure(figsize=(15, 5))
        for i in range(min(num_examples, len(indices))):
            idx = np.random.choice(indices)
            spec, label, source = test_dataset[idx]
            
            plt.subplot(1, num_examples, i + 1)
            plt.imshow(spec.numpy()[0], aspect='auto', origin='lower')
            plt.title(f'{category}\nSource: {source}')
            plt.colorbar()
        
        if save_dir:
            save_path = Path(save_dir) / f'{category.lower().replace(" ", "_")}.png'
            plt.savefig(save_path)
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