import argparse
import yaml
import os

def resolve_path(path):
    """
    Resolve a path that might be relative to the current working directory.
    If the path is already absolute, return it as-is.
    """
    if not path or os.path.isabs(path):
        return path
    
    # Resolve relative to current working directory (where user runs the command)
    return os.path.abspath(path)

def load_config_file(config_path='config.yaml'):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file) or {}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Spectrogram Labeling Tool")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to configuration file")
    parser.add_argument('--folder', type=str, help="Path to the folder containing spectrogram files")
    parser.add_argument('--audio_folder', type=str, help="Path to the folder containing audio files (.flac)")
    parser.add_argument('--output_file', type=str, help="Path to the output file for saving labeled filenames")
    parser.add_argument('--target_dim', type=int, nargs=2, help="Target dimensions (height, width) for reshaping the data")
    parser.add_argument('--specs_per_page', type=int, help="Number of spectrograms to display per page")
    parser.add_argument('--available_labels', type=str, nargs='+', help="Labels available for selection")
    parser.add_argument('--enable_audio', action='store_true', help="Enable audio playback")
    parser.add_argument('--disable_audio', action='store_true', help="Disable audio playback")
    
    return parser.parse_args()

def get_config():
    """Get configuration by merging config file and command line arguments"""
    args = parse_arguments()
    
    # Load config file
    config = load_config_file(args.config)
    
    # Extract values with command line override priority
    folder = args.folder or config.get('data', {}).get('folder')
    audio_folder = args.audio_folder or config.get('data', {}).get('audio_folder')
    output_file = args.output_file or config.get('data', {}).get('output_file')
    
    target_dim = args.target_dim or config.get('display', {}).get('target_dim', [512, 512])
    specs_per_page = args.specs_per_page or config.get('display', {}).get('specs_per_page', 25)
    
    available_labels = args.available_labels or config.get('labels', {}).get('available', ["Rain", "Engine Noise", "Unknown Features"])
    
    # Audio settings
    enable_audio = config.get('audio', {}).get('enable', True)
    if args.enable_audio:
        enable_audio = True
    elif args.disable_audio:
        enable_audio = False
    
    auto_match_audio = config.get('audio', {}).get('auto_match', True)
    
    # Cache settings
    cache_max_size = config.get('cache', {}).get('max_size', 400)
    preload_next_page = config.get('cache', {}).get('preload_next_page', True)
    
    # Validation
    if not folder:
        raise ValueError("Spectrogram folder must be specified in config file or command line")
    if not output_file:
        raise ValueError("Output file must be specified in config file or command line")
    
    # Resolve paths relative to current working directory
    folder = resolve_path(folder)
    audio_folder = resolve_path(audio_folder) if audio_folder else None
    output_file = resolve_path(output_file)
    
    return {
        'folder': folder,
        'audio_folder': audio_folder,
        'output_file': output_file,
        'target_dim': tuple(target_dim),
        'specs_per_page': specs_per_page,
        'available_labels': available_labels,
        'enable_audio': enable_audio,
        'auto_match_audio': auto_match_audio,
        'cache_max_size': cache_max_size,
        'preload_next_page': preload_next_page
    }

# Get configuration
ARGS = get_config()

# Backwards compatibility
FOLDER = ARGS['folder']
AUDIO_FOLDER = ARGS['audio_folder']
OUTPUT_FILE = ARGS['output_file']
TARGET_DIM = ARGS['target_dim']
SPECS_PER_PAGE = ARGS['specs_per_page']
AVAILABLE_LABELS = ARGS['available_labels']
ENABLE_AUDIO = ARGS['enable_audio']
AUTO_MATCH_AUDIO = ARGS['auto_match_audio']
CACHE_MAX_SIZE = ARGS['cache_max_size']
PRELOAD_NEXT_PAGE = ARGS['preload_next_page']