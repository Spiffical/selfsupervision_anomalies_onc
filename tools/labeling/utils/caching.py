from cachetools import LRUCache
import os
import logging
logger = logging.getLogger(__name__)

spectrogram_cache = LRUCache(maxsize=400)
image_cache = LRUCache(maxsize=800)  # Increased to handle 4 versions per image (2 colormaps × 2 y-axis scales)

from config import SPECS_PER_PAGE

def preload_page_images(mat_files, start_idx, end_idx, y_axis_scale='linear'):
    from .image_processing import generate_image_cached
    for file in mat_files[start_idx:end_idx]:
        filename = os.path.basename(file)
        # Preload all combinations of colormap and y-axis scaling
        for colormap in ['default', 'hydrophone']:
            for y_scale in ['linear', 'log']:
                generate_image_cached(filename, colormap, y_scale)

def preload_next_page_images(current_page, total_pages, mat_files):
    next_page = current_page + 1
    if next_page >= total_pages:
        return
    start_idx = next_page * SPECS_PER_PAGE
    end_idx = min(start_idx + SPECS_PER_PAGE, len(mat_files))
    preload_page_images(mat_files, start_idx, end_idx)

def log_cache_usage():
    logger.info(f"Spectrogram cache: {len(spectrogram_cache)}/{spectrogram_cache.maxsize}")
    logger.info(f"Image cache: {len(image_cache)}/{image_cache.maxsize}")