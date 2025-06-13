from dash import html, dcc
import dash_bootstrap_components as dbc
import base64
import os
from typing import Optional

def create_audio_player(audio_file_path: Optional[str], spectrogram_filename: str, player_id: str = None) -> html.Div:
    """
    Create an audio player component for a given audio file.
    
    Args:
        audio_file_path: Path to the audio file (.flac)
        spectrogram_filename: Name of the spectrogram file (for labeling)
        player_id: Unique ID for the audio player
    
    Returns:
        Dash HTML component with audio player
    """
    if not audio_file_path or not os.path.exists(audio_file_path):
        return html.Div([
            html.Small("No audio available", style={'color': 'gray', 'font-style': 'italic'})
        ], style={'text-align': 'center', 'padding': '5px'})
    
    # Generate unique ID if not provided
    if player_id is None:
        # Use filename hash for unique ID
        player_id = f"audio-{hash(spectrogram_filename) % 10000}"
    
    # Create audio element with controls
    # Note: For FLAC files, we might need to convert to a web-compatible format
    # For now, we'll use HTML5 audio (browser support for FLAC varies)
    
    audio_filename = os.path.basename(audio_file_path)
    
    return html.Div([
        html.Audio(
            id=f'{player_id}-audio',
            src=f'/audio/{audio_filename}',  # This will need a route handler
            controls=True,
            style={'width': '100%', 'height': '32px'}
        ),
        html.Div([
            html.Small(f"🎵 {audio_filename}", 
                      style={'color': '#666', 'font-size': '0.8em'})
        ], style={'text-align': 'center', 'margin-top': '2px'})
    ], style={'padding': '5px'})

def create_audio_player_with_controls(audio_file_path: Optional[str], spectrogram_filename: str, player_id: str = None) -> html.Div:
    """
    Create an enhanced audio player with additional controls.
    This version includes play/pause buttons and time display.
    """
    if not audio_file_path or not os.path.exists(audio_file_path):
        return html.Div([
            html.Small("No audio available", style={'color': 'gray', 'font-style': 'italic'})
        ], style={'text-align': 'center', 'padding': '5px'})
    
    # Generate unique ID if not provided
    if player_id is None:
        player_id = f"audio-{hash(spectrogram_filename) % 10000}"
    
    audio_filename = os.path.basename(audio_file_path)
    
    return html.Div([
        # Audio element (hidden, controlled via JavaScript)
        html.Audio(
            id=f'{player_id}-audio',
            src=f'/audio/{audio_filename}',
            style={'display': 'none'}
        ),
        
        # Custom controls
        html.Div([
            dbc.ButtonGroup([
                dbc.Button("▶", id=f'{player_id}-play', size='sm', color='primary', outline=True),
                dbc.Button("⏸", id=f'{player_id}-pause', size='sm', color='secondary', outline=True),
                dbc.Button("⏹", id=f'{player_id}-stop', size='sm', color='danger', outline=True),
            ], size='sm'),
        ], style={'text-align': 'center', 'margin-bottom': '5px'}),
        
        # Progress bar
        dbc.Progress(
            id=f'{player_id}-progress',
            value=0,
            style={'height': '4px', 'margin-bottom': '5px'}
        ),
        
        # Audio info
        html.Div([
            html.Small(f"🎵 {audio_filename}", 
                      style={'color': '#666', 'font-size': '0.8em'})
        ], style={'text-align': 'center'})
    ], style={'padding': '5px', 'border': '1px solid #ddd', 'border-radius': '4px', 'margin': '2px 0'})

def create_simple_audio_link(audio_file_path: Optional[str], spectrogram_filename: str) -> html.Div:
    """
    Create a simple download link for the audio file as a fallback.
    This is useful when direct audio playback in browser is not supported.
    """
    if not audio_file_path or not os.path.exists(audio_file_path):
        return html.Div([
            html.Small("No audio available", style={'color': 'gray', 'font-style': 'italic'})
        ], style={'text-align': 'center', 'padding': '5px'})
    
    audio_filename = os.path.basename(audio_file_path)
    
    return html.Div([
        html.A([
            html.Small(f"🎵 Download Audio: {audio_filename}", 
                      style={'color': '#007bff', 'text-decoration': 'underline'})
        ], href=f'/audio/{audio_filename}', download=audio_filename, target='_blank')
    ], style={'text-align': 'center', 'padding': '5px'}) 