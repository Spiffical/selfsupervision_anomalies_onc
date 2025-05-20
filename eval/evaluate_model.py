import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, fbeta_score
from torch.utils.data import DataLoader, ConcatDataset
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set, Union, Any
import h5py
from src.ssamba.dataset import ONCSpectrogramDataset
import logging
from pathlib import Path
import pandas as pd
import matplotx
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

def setup_presentation_style(style: str = 'default') -> Dict[str, Any]:
    """
    Set up plot styling for different contexts.
    
    Args:
        style: One of ['default', 'presentation', 'poster']
        
    Returns:
        Dictionary containing style parameters
    """
    # Base style that applies to all contexts
    base_style = {
        'figure.dpi': 300,
        'figure.figsize': (12, 8),
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.5,
    }
    
    style_configs = {
        'default': {
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'lines.linewidth': 1.5,
            'axes.titlepad': 10,
            'legend.frameon': True,
            'legend.framealpha': 0.8,
            'legend.edgecolor': '0.8',
        },
        'presentation': {
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'lines.linewidth': 2.0,
            'axes.titlepad': 15,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': '0.8',
        },
        'poster': {
            'font.size': 18,
            'axes.titlesize': 24,
            'axes.labelsize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16,
            'lines.linewidth': 2.5,
            'axes.titlepad': 20,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': '0.8',
        }
    }
    
    # Try to use matplotx style
    try:
        import matplotx
        plt.style.use(matplotx.styles.tableau)  # Using a light theme (tableau style)
        print("Using matplotx style: matplotx.styles.tableau")
    except (ImportError, AttributeError):
        print("matplotx styles not available. Using default matplotlib style.")
        plt.style.use('seaborn-v0_8-whitegrid')  # Fallback to a clean default style
    
    # Update with base style
    plt.rcParams.update(base_style)
    
    # Update with context-specific style
    if style in style_configs:
        plt.rcParams.update(style_configs[style])
    
    # Return the combined style dict for reference
    return {**base_style, **style_configs[style]}

def setup_axis_style(ax: plt.Axes) -> None:
    """
    Apply consistent styling to a matplotlib axis.
    
    Args:
        ax: Matplotlib axis object
    """
    # Add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Style the grid
    ax.grid(True, which='major', linestyle='-', alpha=0.2)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1)
    
    # Add light box around plot
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#666666')
    
    # Adjust tick parameters
    ax.tick_params(which='both', direction='out')
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)

# Colormap function
def colmap_hyd_py(mapsize=64, idev=1):
    iflip = 0
    if idev < 0:
        iflip = 1
        idev = abs(idev)
    
    if idev < 10:
        cmid = mapsize / 2
        np_ = int((mapsize + 1) / 5)
        top = np.ones(np_)
        bot = np.zeros(np_)
        x = np.arange(1, 2 * np_ + 2)  # Equivalent to MATLAB's 1:2*np+1 (generates 1, ..., 2*np_+1)

        if idev != 2:
            # Screen display
            wave = np.sin((x / np.max(x)) * np.pi)
            slope1 = 1.5
            slope2 = 1
        else:
            # Postscript printers
            wave = ((1 - np.cos((x / np.max(x)) * 2 * np.pi)) / 2) ** 1
            slope1 = 1
            slope2 = 2

        wave = wave[np_: 2 * np_]  # MATLAB's wave(np+1:2*np)
        evaw = np.flip(wave)
        red = np.concatenate([
            evaw ** slope1,
            top,
            wave ** slope2,
            bot,
            bot,
            bot
        ])
        grn = np.concatenate([
            bot,
            evaw,
            top,
            top,
            wave,
            bot
        ])
        blu = np.concatenate([
            bot,
            bot,
            bot,
            evaw ** slope2,
            top,
            wave ** slope1
        ])
        colmap0 = np.vstack((red, grn, blu)).T
        mc, nc = colmap0.shape
        dif = int((mc - mapsize) / 2)
        if dif > 0:
            cmap = np.flipud(colmap0[dif: mc - dif, :])
        else:
            cmap = np.flipud(colmap0)
        
        if idev == 3:
            # Fade to black at blue end
            b1 = cmap[0, 2] * 0.9
            zbk = np.linspace(b1 / 8, b1, 8)
            z = np.zeros(8)
            cmbk = np.column_stack((z, z, zbk))
            z0 = np.zeros(4)
            cmb = np.column_stack((z0, z0, z0))
            cmap = np.vstack((cmb, cmbk, cmap))
        elif idev == 4:
            # Fade to white at blue end
            mc_cmap, nc_cmap = cmap.shape # Use different var name to avoid conflict with mc outside
            mm = int(0.15 * mc_cmap) + 1
            cmap = cmap[int(mm / 1.5):, :]
            sgr = np.linspace(1, 0, mm)
            b1 = cmap[0, 2]
            sb = np.linspace(1, b1, mm)
            red_channel = np.concatenate((sgr, cmap[:, 0]))
            grn_channel = np.concatenate((sgr, cmap[:, 1]))
            blu_channel = np.concatenate((sb, cmap[:, 2]))
            cmap = np.column_stack((red_channel, grn_channel, blu_channel))
        
        # This check should be on the shape of cmap *after* idev==3 or idev==4 modifications
        # mc, nc = cmap.shape # Re-evaluate shape
        cmap = np.vstack((cmap, np.array([1, 1, 1])))
    
    elif idev > 9:
        # Grayscale
        maxblk = 0.2
        if idev == 10:
            grey0 = np.linspace(maxblk + (1 - maxblk) / mapsize, 1, mapsize)
            cmap = np.column_stack((grey0, grey0, grey0))
        elif idev == 11:
            nc2 = int(mapsize / 2)
            # Ensure nc2 is not zero to prevent division by zero if mapsize is 1
            if nc2 == 0 and mapsize == 1: nc2 = 1 # Handle mapsize=1 for grayscale
            
            grey1_part1 = np.linspace(maxblk + (1 - maxblk) / nc2, 1 - (1 - maxblk) / nc2, nc2) if nc2 > 0 else np.array([])
            grey1_part2 = np.linspace(1, maxblk + (1 - maxblk) / nc2, nc2) if nc2 > 0 else np.array([])
            
            # Handle odd mapsize for idev=11
            if mapsize % 2 == 1 and nc2 > 0:
                 # Simple strategy: duplicate middle element of first part or just make it '1'
                 # Or, more robustly, ensure total length is mapsize
                 if len(grey1_part1) > 0:
                     grey1_part1_adjusted = np.linspace(maxblk + (1-maxblk)/(nc2+1), 1 - (1-maxblk)/(nc2+1), nc2+1)
                     grey1_part2_adjusted = np.linspace(1, maxblk + (1-maxblk)/nc2, nc2) if nc2 > 0 else np.array([])
                     grey1 = np.concatenate((grey1_part1_adjusted, grey1_part2_adjusted))
                 else: # mapsize = 1, nc2 = 0 (after adjustment)
                     grey1 = np.array([1.0]) # Or some other default for mapsize=1
            else: # mapsize is even or nc2=0 (mapsize=0 or 1 initially handled)
                grey1 = np.concatenate((grey1_part1, grey1_part2))

            # Ensure grey1 has mapsize elements, especially for small mapsizes
            if len(grey1) != mapsize:
                # Fallback if logic above doesn't perfectly yield mapsize elements (e.g. mapsize=1)
                # For mapsize=1, it might be better to just have grey1 = np.array([ (maxblk+1)/2 ]) or similar
                if mapsize > 0:
                    grey1 = np.linspace(maxblk, 1, mapsize) # Simplified fallback
                else: # mapsize = 0
                    grey1 = np.array([])


            cmap = np.column_stack((grey1, grey1, grey1))
            cmap = np.vstack((cmap, np.array([1.0, 1.0, 1.0])))
    else: # Should not be reached if idev is positive integer as per MATLAB original
        cmap = np.zeros((mapsize, 3))

    if iflip == 1:
        cmap = np.flipud(cmap)
    
    # Clip values to be strictly in [0, 1] as mcolors.ListedColormap expects this.
    cmap = np.clip(cmap, 0, 1)

    return cmap

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
    test_dataset: Union[ONCSpectrogramDataset, ConcatDataset],
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate detailed metrics for each anomaly type.
    
    Args:
        test_dataset: The test dataset (can be single dataset or ConcatDataset)
        y_pred: Raw model predictions (before thresholding)
        threshold: Classification threshold
    
    Returns:
        Tuple of (metrics_df, co_occurrence_df):
            - metrics_df: DataFrame with metrics per anomaly type
            - co_occurrence_df: DataFrame with co-occurrence statistics
    """
    # Convert predictions to binary using threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Use the helper function to get samples from either dataset type
    try:
        samples = get_dataset_samples(test_dataset)
    except AttributeError as e:
        print(f"Error getting samples: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Get all unique anomaly types from the dataset
    anomaly_types = set()
    for sample in samples:
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
    for idx, sample in enumerate(samples):
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
                              if len(get_sample_anomalies(samples[idx])) == 1]
                solo_detection_rate = np.mean(y_pred_binary[solo_indices])
            
            if metrics['co_occurrences'] > 0:
                co_indices = [idx for idx in metrics['samples_with_type'] 
                            if len(get_sample_anomalies(samples[idx])) > 1]
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
    
    metrics_df = pd.DataFrame(results) if results else pd.DataFrame(
        columns=['anomaly_type', 'total_samples', 'solo_occurrences', 
                'co_occurrences', 'detection_rate', 'solo_detection_rate', 
                'co_detection_rate'])
    
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
    save_path: Optional[str] = None,
    style: str = 'default'
) -> None:
    """
    Create visualization of anomaly type metrics with improved visual distinction.
    
    Args:
        metrics_df: DataFrame containing metrics
        save_path: Optional path to save the plot
        style: Plot style ('default', 'presentation', or 'poster')
    """
    # Set up the plotting style
    setup_presentation_style(style)
    
    # Sort by total samples for better visualization
    metrics_df = metrics_df.sort_values('total_samples', ascending=True)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Set up spacing
    y_pos = np.arange(len(metrics_df)) * 1.5
    bar_height = 0.3
    
    # Define colors with better contrast using matplotx's color palette
    colors = matplotx.color_palette('colorblind')
    
    # Plot bars with enhanced styling
    overall_bars = ax.barh(y_pos, metrics_df['detection_rate'],
                          label='Overall Detection Rate', 
                          alpha=0.8, 
                          color=colors[0],
                          height=bar_height,
                          edgecolor='black',
                          linewidth=1)
    
    solo_bars = ax.barh(y_pos - bar_height, metrics_df['solo_detection_rate'],
                       label='Solo Detection Rate', 
                       alpha=0.8, 
                       color=colors[1],
                       height=bar_height,
                       hatch='//',
                       edgecolor='black',
                       linewidth=1)
    
    co_bars = ax.barh(y_pos + bar_height, metrics_df['co_detection_rate'],
                     label='Co-occurrence Detection Rate', 
                     alpha=0.8, 
                     color=colors[2],
                     height=bar_height,
                     hatch='xx',
                     edgecolor='black',
                     linewidth=1)
    
    # Customize the axis
    setup_axis_style(ax)
    
    # Add anomaly type labels with improved formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics_df['anomaly_type'], fontweight='bold')
    
    # Add count information with better formatting
    for i, row in enumerate(metrics_df.itertuples()):
        ax.text(-0.05, y_pos[i], 
                f'n={row.total_samples}\n({row.solo_occurrences}s+{row.co_occurrences}c)', 
                va='center', 
                ha='right',
                fontsize=plt.rcParams['font.size'] * 0.9,
                bbox=dict(facecolor='white', 
                         edgecolor='#CCCCCC',
                         alpha=0.9,
                         pad=3,
                         boxstyle='round,pad=0.5'))
    
    # Add percentage annotations with improved positioning and formatting
    for i, row in enumerate(metrics_df.itertuples()):
        # Overall rate
        if row.detection_rate > 0:
            ax.text(row.detection_rate + 0.01, y_pos[i],
                   f'{row.detection_rate:.1%}',
                   va='center',
                   color=colors[0],
                   fontweight='bold',
                   fontsize=plt.rcParams['font.size'] * 0.9)
        
        # Solo rate
        if row.solo_detection_rate > 0:
            ax.text(row.solo_detection_rate + 0.01, y_pos[i] - bar_height,
                   f'{row.solo_detection_rate:.1%}',
                   va='center',
                   color=colors[1],
                   fontweight='bold',
                   fontsize=plt.rcParams['font.size'] * 0.9)
        
        # Co-occurrence rate
        if row.co_detection_rate > 0:
            ax.text(row.co_detection_rate + 0.01, y_pos[i] + bar_height,
                   f'{row.co_detection_rate:.1%}',
                   va='center',
                   color=colors[2],
                   fontweight='bold',
                   fontsize=plt.rcParams['font.size'] * 0.9)
    
    # Enhance axis labels and title
    ax.set_xlabel('Detection Rate', fontweight='bold', labelpad=10)
    ax.set_title('Anomaly Detection Performance by Type', 
                fontweight='bold',
                pad=20)
    
    # Enhance legend
    ax.legend(bbox_to_anchor=(0.5, -0.15),
             loc='upper center',
             ncol=3,
             frameon=True,
             edgecolor='black',
             fancybox=True,
             shadow=True)
    
    # Set x-axis limits and add subtle spines
    ax.set_xlim(-0.25, 1.3)
    
    # Add a light background color to the plot
    ax.set_facecolor('#F8F8F8')
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, 
                    bbox_inches='tight', 
                    dpi=300,
                    facecolor='white',
                    edgecolor='none')
        plt.close()
    else:
        plt.show()

def plot_co_occurrence_matrix(
    test_dataset: ONCSpectrogramDataset,
    save_path: Optional[str] = None,
    style: str = 'default'
) -> None:
    """
    Create a heatmap visualization of anomaly co-occurrences.
    
    Args:
        test_dataset: The test dataset
        save_path: Optional path to save the plot
        style: Plot style ('default', 'presentation', or 'poster')
    """
    # Set up the plotting style
    setup_presentation_style(style)
    
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
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Create heatmap with enhanced styling
    heatmap = sns.heatmap(co_matrix, 
                         xticklabels=anomaly_types,
                         yticklabels=anomaly_types,
                         cmap=cmap,
                         annot=True,
                         fmt='g',  # Use integer format for counts
                         cbar_kws={'label': 'Number of Co-occurrences'},
                         ax=ax)
    
    # Customize the axis
    setup_axis_style(ax)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add percentage annotations
    for i in range(n_types):
        for j in range(n_types):
            if co_matrix[i, j] > 0:
                # Get the current text annotation
                text = heatmap.texts[i * n_types + j]
                # Update text to include percentage
                text.set_text(f'{int(co_matrix[i, j])}\n({percentage_matrix[i, j]:.1f}%)')
                # Adjust font size based on style
                text.set_fontsize(plt.rcParams['font.size'] * 0.8)
                # Center align the text
                text.set_ha('center')
                text.set_va('center')
    
    # Enhance title and labels
    ax.set_title('Anomaly Co-occurrence Matrix', pad=20, fontweight='bold')
    ax.set_xlabel('Anomaly Type', labelpad=10, fontweight='bold')
    ax.set_ylabel('Anomaly Type', labelpad=10, fontweight='bold')
    
    # Add a light background color
    ax.set_facecolor('#F8F8F8')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    style: str = 'default'
) -> None:
    """
    Plot confusion matrix with category labels (TP, TN, FP, FN) in each square
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the plot
        style: Plot style ('default', 'presentation', or 'poster')
    """
    # Set up the plotting style
    setup_presentation_style(style)
    
    # Calculate confusion matrix manually to verify
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Calculate percentages for annotations
    total = np.sum(cm)
    cm_percentages = cm / total * 100
    
    # Create category labels with counts and percentages
    category_labels = [
        [f'True Negative\n{tn}\n({cm_percentages[0,0]:.1f}%)', 
         f'False Positive\n{fp}\n({cm_percentages[0,1]:.1f}%)'],
        [f'False Negative\n{fn}\n({cm_percentages[1,0]:.1f}%)', 
         f'True Positive\n{tp}\n({cm_percentages[1,1]:.1f}%)']
    ]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap with improved aesthetics
    sns.heatmap(cm, annot=category_labels, fmt='', cmap='Blues',
                xticklabels=['Predicted\nNormal', 'Predicted\nAnomaly'],
                yticklabels=['Actual\nNormal', 'Actual\nAnomaly'],
                ax=ax, cbar_kws={'label': 'Count'})
    
    # Customize the axis
    setup_axis_style(ax)
    
    # Additional customization
    ax.set_title('Confusion Matrix', pad=20, weight='bold')
    ax.set_ylabel('True Label', weight='bold')
    ax.set_xlabel('Predicted Label', weight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
    else:
        plt.show()

def plot_example_spectrograms(
    test_dataset: ONCSpectrogramDataset,
    y_pred: np.ndarray,
    num_examples: int = 4,
    samples_per_row: int = 4,
    save_dir: Optional[str] = None,
    style: str = 'default'
) -> None:
    """
    Plot example spectrograms for true/false positives/negatives with label strings
    
    Args:
        test_dataset: The test dataset
        y_pred: Model predictions
        num_examples: Number of examples to show for each category
        samples_per_row: Number of samples to show in each row
        save_dir: Optional directory to save the plots
        style: Plot style ('default', 'presentation', or 'poster')
    """
    # Set up the plotting style
    setup_presentation_style(style)
    
    # Create the custom colormap
    cmap_array = colmap_hyd_py(36, 3) 
    custom_cmap = mcolors.ListedColormap(cmap_array)

    true_labels = np.array([sample['is_anomalous'] for sample in test_dataset.sample_info])
    
    if len(y_pred.shape) > 1:
        y_pred = y_pred.squeeze()
    
    # Ensure y_pred and true_labels are boolean for logical operations
    y_pred_bool = y_pred.astype(bool)
    true_labels_bool = true_labels.astype(bool)
    
    true_pos = np.where(y_pred_bool & true_labels_bool)[0]
    true_neg = np.where(~y_pred_bool & ~true_labels_bool)[0]
    false_pos = np.where(y_pred_bool & ~true_labels_bool)[0]
    false_neg = np.where(~y_pred_bool & true_labels_bool)[0]
    
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
            print(f"No examples for {category}.")
            continue
        
        n_samples = min(num_examples, len(indices))
        n_rows = (n_samples + samples_per_row - 1) // samples_per_row
        
        # Create figure with subplots
        fig = plt.figure(figsize=(5 * samples_per_row, 1 + 3 * n_rows))
        fig.suptitle(f'{category} Examples\n(Total: {len(indices)}, Showing: {n_samples})', 
                     fontsize=plt.rcParams['axes.titlesize'] * 1.2,
                     y=1.0,
                     fontweight='bold')
        
        for i in range(n_samples):
            idx = np.random.choice(indices)
            spec_tensor, label_bool, source_str = test_dataset[idx]
            spec_numpy = spec_tensor.numpy() if not hasattr(spec_tensor, '_data') else spec_tensor._data
            
            raw_labels = test_dataset.sample_info[idx]['labels']
            label_string = ';'.join(raw_labels)
            is_anomalous_true = test_dataset.sample_info[idx]['is_anomalous']
            
            # Create subplot with proper axes object
            ax = plt.subplot(n_rows, samples_per_row, i + 1)
            
            # Plot spectrogram with custom colormap
            im = ax.imshow(spec_numpy[0], aspect='auto', origin='lower', cmap=custom_cmap)
            
            # Add colorbar with proper sizing
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            
            # Customize the axis
            setup_axis_style(ax)
            
            # Create title with source and labels
            title = f'Source: {source_str}'
            if label_string:
                # Split long label strings into multiple lines
                label_lines = [label_string[j:j+20] for j in range(0, len(label_string), 20)]
                title += '\nLabels: ' + '\n'.join(label_lines)
            
            title += f'\nTrue: {"Anomalous" if is_anomalous_true else "Normal"}'
            title += f'\nPred Score: {y_pred[idx]:.3f}'
            title += f'\n({"Anomalous" if y_pred_bool[idx] else "Normal"})'
            
            ax.set_title(title, fontsize=plt.rcParams['font.size'] * 0.8)
            
            # Add label string in bottom-left corner with improved styling
            if label_string:
                ax.text(0.02, 0.02, label_string, 
                        color='white',
                        fontsize=plt.rcParams['font.size'] * 0.6,
                        transform=ax.transAxes,
                        bbox=dict(facecolor='black',
                                alpha=0.7,
                                edgecolor='white',
                                linewidth=0.5,
                                pad=3),
                        wrap=True)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90 if n_rows > 1 else 0.85)
        
        if save_dir:
            output_path = Path(save_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            save_path = output_path / f'{category.lower().replace(" ", "_")}_examples.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
            print(f"Saved {category} examples to {save_path}")
            plt.close(fig)
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
    max_samples: int = 100,
    samples_per_row: int = 5,
    style: str = 'default'
) -> None:
    """
    Plot all false positive predictions in a grid layout.
    """
    # Create the custom colormap
    cmap_array = colmap_hyd_py(36, 3)
    custom_cmap = mcolors.ListedColormap(cmap_array)

    if len(y_pred.shape) > 1:
        y_pred = y_pred.squeeze()
    
    true_labels = np.array([sample['is_anomalous'] for sample in test_dataset.sample_info])
    
    # Assuming y_pred are scores/probabilities, apply a threshold (e.g., 0.5)
    # If y_pred is already binary (0/1), this can be simplified.
    y_pred_bool = (y_pred > 0.5).astype(bool) # Example threshold
    true_labels_bool = true_labels.astype(bool)

    false_pos_indices = np.where(y_pred_bool & ~true_labels_bool)[0]
    
    if len(false_pos_indices) == 0:
        print("No false positives found.")
        return
    
    print(f"Found {len(false_pos_indices)} false positives. Plotting up to {max_samples}.")
    
    n_samples_to_plot = min(len(false_pos_indices), max_samples)
    indices_to_plot = false_pos_indices[:n_samples_to_plot]
    
    n_rows = (n_samples_to_plot + samples_per_row - 1) // samples_per_row
    
    fig = plt.figure(figsize=(4 * samples_per_row, 1 + 3 * n_rows)) # Added 1 to height for suptitle
    fig.suptitle(f'All False Positives (Showing {n_samples_to_plot} of {len(false_pos_indices)})',
                 fontsize=16, y=1.0)


    for i, idx in enumerate(indices_to_plot):
        spec_tensor, _, source_str = test_dataset[idx]
        spec_numpy = spec_tensor.numpy() if not hasattr(spec_tensor, '_data') else spec_tensor._data

        raw_labels = test_dataset.sample_info[idx]['labels']
        label_string = ';'.join(raw_labels)
        
        plt.subplot(n_rows, samples_per_row, i + 1)
        # Use the custom colormap here
        plt.imshow(spec_numpy[0], aspect='auto', origin='lower', cmap=custom_cmap)
        
        title = f'Source: {source_str}'
        if label_string:
            label_lines = [label_string[j:j+20] for j in range(0, len(label_string), 20)]
            title += '\nLabel: ' + '\n'.join(label_lines)
        
        plt.title(title, fontsize=8)
        #plt.colorbar()
        
        #plt.text(0.02, 0.02, f'Pred Score: {y_pred[idx]:.3f}', 
        #         color='white', fontsize=6,
        #         transform=plt.gca().transAxes,
        #         bbox=dict(facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92 if n_rows > 1 else 0.88) # Adjust top for suptitle

    if save_dir:
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / 'all_false_positives.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved false positives plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_all_false_negatives(
    test_dataset: ONCSpectrogramDataset,
    y_pred: np.ndarray,
    save_dir: Optional[str] = None,
    max_samples: int = 100,
    samples_per_row: int = 5,
    style: str = 'default'
) -> None:
    """
    Plot all false negative predictions in a grid layout.
    """
    # Create the custom colormap
    cmap_array = colmap_hyd_py(36, 3)
    custom_cmap = mcolors.ListedColormap(cmap_array)

    if len(y_pred.shape) > 1:
        y_pred = y_pred.squeeze()
        
    true_labels = np.array([sample['is_anomalous'] for sample in test_dataset.sample_info])

    # Assuming y_pred are scores/probabilities, apply a threshold (e.g., 0.5)
    y_pred_bool = (y_pred > 0.5).astype(bool) # Example threshold
    true_labels_bool = true_labels.astype(bool)

    false_neg_indices = np.where(~y_pred_bool & true_labels_bool)[0]
    
    if len(false_neg_indices) == 0:
        print("No false negatives found.")
        return

    print(f"Found {len(false_neg_indices)} false negatives. Plotting up to {max_samples}.")

    n_samples_to_plot = min(len(false_neg_indices), max_samples)
    indices_to_plot = false_neg_indices[:n_samples_to_plot]
    
    n_rows = (n_samples_to_plot + samples_per_row - 1) // samples_per_row
    
    fig = plt.figure(figsize=(4 * samples_per_row, 1 + 3 * n_rows)) # Added 1 to height for suptitle
    fig.suptitle(f'All False Negatives (Showing {n_samples_to_plot} of {len(false_neg_indices)})',
                 fontsize=16, y=1.0)

    for i, idx in enumerate(indices_to_plot):
        spec_tensor, _, source_str = test_dataset[idx]
        spec_numpy = spec_tensor.numpy() if not hasattr(spec_tensor, '_data') else spec_tensor._data
        
        raw_labels = test_dataset.sample_info[idx]['labels']
        label_string = ';'.join(raw_labels)
        
        plt.subplot(n_rows, samples_per_row, i + 1)
        # Use the custom colormap here
        plt.imshow(spec_numpy[0], aspect='auto', origin='lower', cmap=custom_cmap)
        
        title = f'Source: {source_str}'
        if label_string:
            label_lines = [label_string[j:j+20] for j in range(0, len(label_string), 20)]
            title += '\nLabels: ' + '\n'.join(label_lines)
            
        plt.title(title, fontsize=8)
        plt.colorbar()
        
        plt.text(0.02, 0.02, f'Pred Score: {y_pred[idx]:.3f}', 
                 color='white', fontsize=6,
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92 if n_rows > 1 else 0.88) # Adjust top for suptitle

    if save_dir:
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / 'all_false_negatives.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved false negatives plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_anomaly_type_by_hydrophone(
    test_dataset: ONCSpectrogramDataset,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    min_samples: int = 5,  # Minimum samples required for a hydrophone to be included
    save_dir: Optional[str] = None,
    style: str = 'default'
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
    threshold: float = 0.5,
    plot_style: str = 'default'
):
    """
    Main evaluation function
    
    Args:
        model_path: Path to saved model
        data_path: Path to HDF5 dataset
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        threshold: Classification threshold
        plot_style: Plot style ('default', 'presentation', or 'poster')
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
    plot_confusion_matrix(y_true, y_pred, 
                         output_dir / 'confusion_matrix.png',
                         style=plot_style)
    
    # Plot example spectrograms
    plot_example_spectrograms(
        test_dataset,
        y_pred,
        num_examples=4,
        save_dir=output_dir / 'examples',
        style=plot_style
    )

    # Plot all false positives
    plot_all_false_positives(
        test_dataset,
        y_pred,
        save_dir=output_dir / 'false_positives',
        style=plot_style
    )

    # Plot all false negatives
    plot_all_false_negatives(
        test_dataset,
        y_pred,
        save_dir=output_dir / 'false_negatives',
        style=plot_style
    )

    # Plot anomaly type metrics
    plot_anomaly_type_metrics(
        anomaly_metrics,
        output_dir / 'anomaly_type_metrics.png',
        style=plot_style
    )

    # Plot co-occurrence matrix
    plot_co_occurrence_matrix(
        test_dataset,
        output_dir / 'co_occurrence_matrix.png',
        style=plot_style
    )

    # Plot anomaly type by hydrophone
    plot_anomaly_type_by_hydrophone(
        test_dataset,
        y_pred,
        threshold,
        min_samples=5,
        save_dir=output_dir / 'anomaly_type_by_hydrophone',
        style=plot_style
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
    parser.add_argument('--plot-style', default='default', choices=['default', 'presentation', 'poster'],
                      help='Plot style to use')
    
    args = parser.parse_args()
    main(**vars(args))