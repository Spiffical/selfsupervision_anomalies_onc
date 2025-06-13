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

@cached(image_cache)
def generate_image_cached(filename, colormap='default'):
    logger.info(f"Generating image for {filename} with {colormap} colormap")
    spectrogram = load_spectrogram_cached(filename)
    if spectrogram is None:
        return None
    fig, ax = plt.subplots(figsize=(2, 2))
    if colormap == 'hydrophone':
        cmap_array = colmap_hyd_py(36, 3)
        cmap = mcolors.ListedColormap(cmap_array)
    else:
        cmap = 'viridis'
    
    ax.imshow(spectrogram['psd'], 
              aspect='auto', 
              origin='lower', 
              cmap=cmap,
              vmin=40,    # Match zmin from modal
              vmax=140)   # Match zmax from modal
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    return f"data:image/png;base64,{data}"

def create_spectrogram_figure(spectrogram_data, colormap_value):
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

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=psd,
        x=time_minutes,
        y=freq,
        colorscale=colorscale,
        zmin=40,
        zmax=140,
        colorbar=dict(
            title='Power (dB/Hz)',
            tickformat='.1f'
        )
    ))
    
    fig.update_layout(
        xaxis=dict(
            title='Time (minutes)',
            showgrid=True,
            tickformat='.2f'
        ),
        yaxis=dict(
            title='Frequency (kHz)',
            showgrid=True,
            tickformat='.0f'
        ),
        margin=dict(l=50, r=20, t=20, b=50),
        autosize=True,
    )
    return fig