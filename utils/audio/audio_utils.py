#!/usr/bin/env python3
"""
Audio processing utility functions.
"""

import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List
import logging

logger = logging.getLogger(__name__)

def load_audio(audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    Load audio file with automatic format detection.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    try:
        # Try soundfile first (supports FLAC, WAV, etc.)
        audio_data, sample_rate = sf.read(str(audio_path))
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
    except Exception as e:
        try:
            # Fallback to librosa for other formats
            audio_data, sample_rate = librosa.load(str(audio_path), sr=None)
        except Exception as e2:
            raise RuntimeError(f"Could not load audio file {audio_path}: {e}, {e2}")
    
    return audio_data, sample_rate

def validate_audio_file(audio_path: Union[str, Path]) -> bool:
    """
    Validate if file is a readable audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        True if file can be loaded, False otherwise
    """
    try:
        audio_data, sample_rate = load_audio(audio_path)
        return len(audio_data) > 0 and sample_rate > 0
    except Exception:
        return False

def get_audio_info(audio_path: Union[str, Path]) -> dict:
    """
    Get basic information about an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio file information
    """
    try:
        audio_data, sample_rate = load_audio(audio_path)
        duration = len(audio_data) / sample_rate
        
        return {
            'path': str(audio_path),
            'sample_rate': sample_rate,
            'duration': duration,
            'samples': len(audio_data),
            'channels': 1,  # We convert to mono
            'valid': True
        }
    except Exception as e:
        return {
            'path': str(audio_path),
            'error': str(e),
            'valid': False
        }

def find_audio_files(directory: Union[str, Path], 
                    extensions: List[str] = ['.wav', '.flac', '.mp3', '.m4a']) -> List[Path]:
    """
    Find all audio files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        List of audio file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(directory.glob(f"*{ext}"))
        audio_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(audio_files)

def estimate_processing_time(audio_files: List[Path], 
                           win_dur: float = 1.0,
                           overhead_factor: float = 2.0) -> float:
    """
    Estimate processing time for a list of audio files.
    
    Args:
        audio_files: List of audio file paths
        win_dur: Window duration for spectrogram computation
        overhead_factor: Factor to account for processing overhead
        
    Returns:
        Estimated processing time in seconds
    """
    total_duration = 0
    valid_files = 0
    
    for audio_file in audio_files:
        try:
            info = get_audio_info(audio_file)
            if info['valid']:
                total_duration += info['duration']
                valid_files += 1
        except Exception:
            continue
    
    if valid_files == 0:
        return 0
    
    # Rough estimate: processing time scales with audio duration and window size
    processing_time = (total_duration / win_dur) * overhead_factor / 100  # Rough scaling factor
    
    return max(processing_time, valid_files * 0.1)  # Minimum 0.1s per file 