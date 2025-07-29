import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import base64
import cv2
from .colmap_hyd import colmap_hyd_py
import plotly.graph_objects as go
from .caching import image_cache, spectrogram_cache
import os
import logging
import scipy.io as sio
from cachetools import cached

logger = logging.getLogger(__name__)

# Get args from config
from config import FOLDER, TARGET_DIM

# Helper Functions
@cached(spectrogram_cache)
def load_spectrogram_cached(filename):
    logger.info(f"Loading spectrogram for {filename}")
    mat_file = os.path.join(FOLDER, filename)
    EXPECTED_SHAPE = (854, 1000)  # Expected spectrogram shape
    try:
        mat_data = sio.loadmat(mat_file)
        if 'SpectData' in mat_data:
            data = mat_data['SpectData']
            psd = data['PSD'][0, 0]
            freq = data['frequency'][0, 0].flatten()  # Extract frequency data
            time = data['time'][0, 0].flatten()  # Extract time data

            # Check if spectrogram is shortened
            if psd.shape[1] < EXPECTED_SHAPE[1]:
                print(f"\nWarning: {os.path.basename(mat_file)} has shape {psd.shape}, padding to {EXPECTED_SHAPE}")
                padding_width = ((0, 0), (0, EXPECTED_SHAPE[1] - psd.shape[1]))
                psd = np.pad(psd, padding_width, mode='constant', constant_values=0)

            # Create mask for valid data (non-inf)
            valid_mask = (psd != -np.inf)
            
            # Replace -inf with zeros
            psd[~valid_mask] = 0
            
            # Replace NaNs with zeros
            psd = np.nan_to_num(psd, nan=0.0)
            
            return {
                'psd': psd,
                'freq': freq,
                'time': time
            }
    except Exception as e:
        logger.error(f"Error loading {mat_file}: {e}")
    return None

def generate_image_cached(filename, colormap='default', y_axis_scale='linear'):
    # check if in cache first
    cache_key = (filename, colormap, y_axis_scale)
    if cache_key in image_cache:
        return image_cache[cache_key]
    else:
        result = _generate_image(filename, colormap, y_axis_scale)
        image_cache[cache_key] = result
        return result

def _generate_image(filename, colormap='default', y_axis_scale='linear'):
    import time as time_module
    start_time = time_module.time()
    
    spectrogram = load_spectrogram_cached(filename)
    if spectrogram is None:
        return None
    
    fig, ax = plt.subplots(figsize=(1.5, 1.5), facecolor='none')  # smaller figures for thumbnails
    if colormap == 'hydrophone':
        cmap_array = colmap_hyd_py(36, 3)
        cmap = mcolors.ListedColormap(cmap_array)
    else:
        cmap = 'viridis'
    
    psd = spectrogram['psd']
    freq = spectrogram['freq']/1000  # Convert to kHz
    time = spectrogram['time']
    
    # Convert Julian days to minutes for x-axis
    time_minutes = (time - time[0]) * 24 * 60
    
    if y_axis_scale == 'log':
        # OPTIMIZED log scaling: use imshow with pre-transformed data
        valid_freq_mask = freq > 0
        if not np.any(valid_freq_mask):
            # Fallback to linear 
            im = ax.imshow(psd, 
                          extent=[time_minutes[0], time_minutes[-1], freq[0], freq[-1]],
                          aspect='auto', origin='lower', cmap=cmap, vmin=40, vmax=140)
        else:
            # Pre-filter data and use faster rendering
            freq_for_plot = freq[valid_freq_mask]
            psd_for_plot = psd[valid_freq_mask, :]
            min_freq = max(freq_for_plot[0], 0.1)
            max_freq = freq_for_plot[-1]
            
            # Simple approach: use imshow and set log scale (but with reduced complexity)
            im = ax.imshow(psd_for_plot, 
                          extent=[time_minutes[0], time_minutes[-1], min_freq, max_freq],
                          aspect='auto', origin='lower', cmap=cmap, vmin=40, vmax=140)
            ax.set_yscale('log')
            ax.set_ylim(min_freq, max_freq)
    else:
        # Linear scaling - use imshow for consistency with existing behavior
        im = ax.imshow(psd, 
                      extent=[time_minutes[0], time_minutes[-1], freq[0], freq[-1]],
                      aspect='auto', 
                      origin='lower', 
                      cmap=cmap,
                      vmin=40,    # Match zmin from modal
                      vmax=140)   # Match zmax from modal
    
    ax.axis('off')
    # Remove all whitespace/padding around the image
    ax.set_position([0, 0, 1, 1])
    
    buf = BytesIO()
    # ultra low dpi for thumbnails - these are tiny images anyway
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, 
                facecolor='none', edgecolor='none', dpi=72)
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    
    # Performance monitoring removed - optimizations complete
    
    return f"data:image/png;base64,{data}"

def create_spectrogram_figure(spectrogram_data, colormap_value, y_axis_scale='linear'):
    if spectrogram_data is None:
        return go.Figure()
        
    psd = spectrogram_data['psd']
    freq = spectrogram_data['freq']/1000  # Convert to kHz
    time = spectrogram_data['time']

    print(f"Range of psd: {np.min(psd)} to {np.max(psd)}")
    
    # Convert Julian days to minutes
    time_minutes = (time - time[0]) * 24 * 60  # Convert days to minutes
    
    if colormap_value == 'hydrophone':
        cmap_array = colmap_hyd_py(36, 3)
        colorscale = [[i/(len(cmap_array)-1), f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'] 
                      for i, (r, g, b) in enumerate(cmap_array)]
    else:
        colorscale = 'Viridis'

    # Handle y-axis scaling
    if y_axis_scale == 'log':
        # For logarithmic scaling, ensure we don't have zero or negative frequencies
        freq_for_plot = np.maximum(freq, 0.001)  # Set minimum frequency to avoid log(0)
        y_axis_type = 'log'
        y_axis_title = 'Frequency (kHz) - Log Scale'
    else:
        freq_for_plot = freq
        y_axis_type = 'linear'
        y_axis_title = 'Frequency (kHz)'

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=psd,
        x=time_minutes,
        y=freq_for_plot,
        colorscale=colorscale,
        zmin=40,
        zmax=140,
        colorbar=dict(
            title='Power (dB/Hz)',
            tickformat='.1f'
        )
    ))
    
    # Configure y-axis based on scale type
    if y_axis_scale == 'log':
        # For log scale, determine appropriate tick values based on actual data range
        freq_min = np.min(freq_for_plot)
        freq_max = np.max(freq_for_plot)
        
        # Define potential tick values
        all_tick_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        all_tick_labels = ['0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100']
        
        # Filter tick values to only include those within the data range
        valid_ticks = []
        valid_labels = []
        for tick_val, tick_label in zip(all_tick_values, all_tick_labels):
            if freq_min <= tick_val <= freq_max:
                valid_ticks.append(tick_val)
                valid_labels.append(tick_label)
        
        # Ensure we have at least a few ticks by extending the range slightly if needed
        if len(valid_ticks) < 3:
            # Add ticks slightly outside the range
            for tick_val, tick_label in zip(all_tick_values, all_tick_labels):
                if tick_val < freq_min and tick_val >= freq_min * 0.5:
                    valid_ticks.insert(0, tick_val)
                    valid_labels.insert(0, tick_label)
                elif tick_val > freq_max and tick_val <= freq_max * 2:
                    valid_ticks.append(tick_val)
                    valid_labels.append(tick_label)
        
        # Set the range to just slightly beyond the actual data range
        log_min = np.log10(max(freq_min * 0.8, 0.1))
        log_max = np.log10(min(freq_max * 1.2, 100))
        
        yaxis_config = dict(
            title=y_axis_title,
            showgrid=True,
            type=y_axis_type,
            tickmode='array',
            tickvals=valid_ticks,
            ticktext=valid_labels,
            range=[log_min, log_max]
        )
    else:
        # For linear scale, use automatic formatting
        yaxis_config = dict(
            title=y_axis_title,
            showgrid=True,
            tickformat='.0f',
            type=y_axis_type
        )

    fig.update_layout(
        xaxis=dict(
            title='Time (minutes)',
            showgrid=True,
            tickformat='.2f'
        ),
        yaxis=yaxis_config,
        margin=dict(l=50, r=20, t=20, b=50),
        autosize=True,
    )
    return fig