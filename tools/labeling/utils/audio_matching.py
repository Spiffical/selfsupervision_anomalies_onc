import os
import glob
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Parse timestamp from ONC filename format.
    
    Examples:
    - ICLISTENHF6406_20240523T061507.000Z.flac -> 2024-05-23 06:15:07
    - ICLISTENHF6406_20240523T061507.000Z_20240523T062007.000Z-spect_plotRes.mat -> 2024-05-23 06:15:07
    """
    # Pattern for ONC timestamp format: YYYYMMDDTHHMMSS.sssZ
    timestamp_pattern = r'(\d{8}T\d{6}(?:\.\d{3})?Z)'
    
    matches = re.findall(timestamp_pattern, filename)
    if not matches:
        return None
    
    # Take the first timestamp (start time for spectrograms)
    timestamp_str = matches[0]
    
    # Remove microseconds if present for parsing
    if '.' in timestamp_str:
        timestamp_str = timestamp_str.split('.')[0] + 'Z'
    
    try:
        return datetime.strptime(timestamp_str, '%Y%m%dT%H%M%SZ')
    except ValueError:
        return None

def parse_spectrogram_time_range(filename: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Parse start and end timestamps from spectrogram filename.
    
    Example:
    ICLISTENHF6406_20240523T061507.000Z_20240523T062007.000Z-spect_plotRes.mat
    -> (2024-05-23 06:15:07, 2024-05-23 06:20:07)
    """
    timestamp_pattern = r'(\d{8}T\d{6}(?:\.\d{3})?Z)'
    matches = re.findall(timestamp_pattern, filename)
    
    if len(matches) < 2:
        return None
    
    try:
        start_str = matches[0].split('.')[0] + 'Z' if '.' in matches[0] else matches[0]
        end_str = matches[1].split('.')[0] + 'Z' if '.' in matches[1] else matches[1]
        
        start_time = datetime.strptime(start_str, '%Y%m%dT%H%M%SZ')
        end_time = datetime.strptime(end_str, '%Y%m%dT%H%M%SZ')
        
        return start_time, end_time
    except ValueError:
        return None

def find_matching_audio_files(spectrogram_filename: str, audio_folder: str, tolerance_seconds: int = 300) -> List[str]:
    """
    Find audio files that match the time range of a spectrogram file.
    
    Args:
        spectrogram_filename: Name of the spectrogram file
        audio_folder: Path to folder containing audio files
        tolerance_seconds: How many seconds before/after to search for audio files
    
    Returns:
        List of matching audio file paths
    """
    if not audio_folder or not os.path.exists(audio_folder):
        return []
    
    # Parse spectrogram time range
    time_range = parse_spectrogram_time_range(spectrogram_filename)
    if not time_range:
        return []
    
    start_time, end_time = time_range
    
    # Get all audio files
    audio_files = glob.glob(os.path.join(audio_folder, '*.flac'))
    matching_files = []
    
    # Find audio files within the time range
    for audio_file in audio_files:
        audio_basename = os.path.basename(audio_file)
        audio_timestamp = parse_timestamp_from_filename(audio_basename)
        
        if audio_timestamp:
            # Check if audio timestamp falls within spectrogram time range (with tolerance)
            tolerance_delta = timedelta(seconds=tolerance_seconds)
            if (start_time - tolerance_delta <= audio_timestamp <= end_time + tolerance_delta):
                matching_files.append(audio_file)
    
    # Sort by timestamp
    matching_files.sort(key=lambda f: parse_timestamp_from_filename(os.path.basename(f)))
    return matching_files

def create_audio_spectrogram_mapping(spectrogram_folder: str, audio_folder: str) -> Dict[str, List[str]]:
    """
    Create a mapping of spectrogram filenames to their matching audio files.
    
    Args:
        spectrogram_folder: Path to folder containing spectrogram files
        audio_folder: Path to folder containing audio files
    
    Returns:
        Dictionary mapping spectrogram filename -> list of audio file paths
    """
    if not audio_folder or not os.path.exists(audio_folder):
        return {}
    
    spectrogram_files = glob.glob(os.path.join(spectrogram_folder, '*.mat'))
    mapping = {}
    
    for spec_file in spectrogram_files:
        spec_basename = os.path.basename(spec_file)
        matching_audio = find_matching_audio_files(spec_basename, audio_folder)
        if matching_audio:
            mapping[spec_basename] = matching_audio
    
    return mapping

def get_representative_audio_file(audio_files: List[str]) -> Optional[str]:
    """
    Get a representative audio file from a list of matching files.
    For now, just returns the first file, but could be enhanced to pick
    the best quality or most complete file.
    """
    return audio_files[0] if audio_files else None 