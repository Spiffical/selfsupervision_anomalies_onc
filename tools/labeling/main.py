import dash
import dash_bootstrap_components as dbc
from layout import create_layout
from callbacks import register_callbacks
from flask import send_file, abort
import os
from config import AUDIO_FOLDER, ENABLE_AUDIO
from utils.audio_matching import create_audio_spectrogram_mapping, get_representative_audio_file

app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
    ]
)

# Global variable to store audio mapping
audio_mapping = {}

def get_repo_root():
    """Find the repository root by looking for setup.py or .git"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Walk up the directory tree
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        # Check for common repo indicators
        if any(os.path.exists(os.path.join(current_dir, indicator)) 
               for indicator in ['setup.py', '.git', 'README.md']):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # Fallback to current working directory if repo root not found
    return os.getcwd()

def resolve_path(path):
    """
    Resolve a path that might be relative to the repository root.
    If the path is already absolute, return it as-is.
    """
    if not path or os.path.isabs(path):
        return path
    
    # Resolve relative to repository root
    repo_root = get_repo_root()
    return os.path.join(repo_root, path)

def create_app(args):
    global audio_mapping
    
    app.layout = create_layout(args)
    register_callbacks(app)
    
    # Create audio-spectrogram mapping if audio is enabled
    if ENABLE_AUDIO and AUDIO_FOLDER:
        print("Creating audio-spectrogram mapping...")
        folder = args['folder'] if isinstance(args, dict) else args.folder
        audio_folder = args['audio_folder'] if isinstance(args, dict) else getattr(args, 'audio_folder', AUDIO_FOLDER)
        
        if audio_folder:
            # Resolve paths relative to repository root
            resolved_folder = resolve_path(folder)
            resolved_audio_folder = resolve_path(audio_folder)
            
            audio_mapping = create_audio_spectrogram_mapping(resolved_folder, resolved_audio_folder)
            print(f"Found audio mappings for {len(audio_mapping)} spectrograms")
            print(f"Spectrogram folder: {resolved_folder}")
            print(f"Audio folder: {resolved_audio_folder}")
    
    # Add route for serving audio files
    @app.server.route('/audio/<filename>')
    def serve_audio(filename):
        """Serve audio files for the audio players"""
        if not ENABLE_AUDIO or not AUDIO_FOLDER:
            abort(404)
        
        # Resolve the audio folder path relative to repository root
        resolved_audio_folder = resolve_path(AUDIO_FOLDER)
        
        # Find the full path to the audio file
        audio_file_path = os.path.join(resolved_audio_folder, filename)
        
        if not os.path.exists(audio_file_path):
            # Try to find the file in subdirectories (if organized differently)
            for root, dirs, files in os.walk(resolved_audio_folder):
                if filename in files:
                    audio_file_path = os.path.join(root, filename)
                    break
            else:
                print(f"Audio file not found: {filename}")
                print(f"Searched in: {resolved_audio_folder}")
                abort(404)
        
        try:
            return send_file(audio_file_path, as_attachment=False)
        except Exception as e:
            print(f"Error serving audio file {filename}: {e}")
            abort(500)
    
    return app

def get_audio_mapping():
    """Get the global audio mapping"""
    return audio_mapping