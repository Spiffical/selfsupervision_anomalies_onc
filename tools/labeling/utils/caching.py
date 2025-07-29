from cachetools import LRUCache
import os
import logging
logger = logging.getLogger(__name__)

spectrogram_cache = LRUCache(maxsize=400)
image_cache = LRUCache(maxsize=800)  # Increased to handle 4 versions per image (2 colormaps × 2 y-axis scales)

from config import SPECS_PER_PAGE

def preload_page_images(mat_files, start_idx, end_idx, y_axis_scale='linear'):
    import time as time_module
    start_time = time_module.time()
    from .image_processing import generate_image_cached
    
    total_files = end_idx - start_idx
    print(f"🚀 Preloading {total_files} files...")
    
    for file in mat_files[start_idx:end_idx]:
        filename = os.path.basename(file)
        # Preload all combinations of colormap and y-axis scaling
        for colormap in ['default', 'hydrophone']:
            for y_scale in ['linear', 'log']:
                generate_image_cached(filename, colormap, y_scale)
    
    end_time = time_module.time()
    preload_time = (end_time - start_time) * 1000
    print(f"✅ Preload completed in {preload_time:.1f}ms for {total_files} files ({total_files * 4} images total)")

def preload_next_page_images(current_page, total_pages, mat_files):
    next_page = current_page + 1
    if next_page >= total_pages:
        print(f"🔚 No next page to preload (current: {current_page}, total: {total_pages})")
        return
    start_idx = next_page * SPECS_PER_PAGE
    end_idx = min(start_idx + SPECS_PER_PAGE, len(mat_files))
    print(f"🔄 Background preloading next page {next_page + 1}")
    preload_page_images(mat_files, start_idx, end_idx)

def log_cache_usage():
    print(f"📊 Cache Stats:")
    print(f"   Spectrogram cache: {len(spectrogram_cache)}/{spectrogram_cache.maxsize}")
    print(f"   Image cache: {len(image_cache)}/{image_cache.maxsize}")
    logger.info(f"Spectrogram cache: {len(spectrogram_cache)}/{spectrogram_cache.maxsize}")
    logger.info(f"Image cache: {len(image_cache)}/{image_cache.maxsize}")