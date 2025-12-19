#!/usr/bin/env python3
"""
Fin Whale Call Analysis and Comparison Tool

This script analyzes the fin whale call library Excel file, downloads corresponding 
ONC audio files, creates custom spectrograms, and downloads ONC PNG spectrograms for comparison.

Features:
- Intelligent sampling from 93k+ fin whale calls
- ONC API integration for downloading specific .wav files
- Custom spectrogram generation optimized for whale calls
- ONC PNG spectrogram download for comparison
- Organized output structure for analysis

Usage Examples:

  # Sample 10 calls and create comparison spectrograms
  python scripts/whale_call_analysis.py --excel-file data/finwhales/FinWhale20Hz_CallLibrary_Rannankari.xlsx --sample-size 10 --output-dir whale_analysis
  
  # Generate ML dataset: .mat files only for entire unfiltered dataset  
  python scripts/whale_call_analysis.py --excel-file data/finwhales/FinWhale20Hz_CallLibrary_Rannankari.xlsx --process-all --mat-only --skip-onc-spectrograms --cleanup-audio
  
  # Focus on specific device and date range
  python scripts/whale_call_analysis.py --excel-file data/finwhales/FinWhale20Hz_CallLibrary_Rannankari.xlsx --device ICLISTENHF1353 --start-date 2018-07-01 --end-date 2018-08-01 --sample-size 20
  
  # High-quality calls only with custom spectrogram parameters
  python scripts/whale_call_analysis.py --excel-file data/finwhales/FinWhale20Hz_CallLibrary_Rannankari.xlsx --sample-size 15 --min-duration 5.0 --freq-range 10 500 --win-dur 1.0
  
  # Generate visualization plots only (no .mat files)
  python scripts/whale_call_analysis.py --excel-file data/finwhales/FinWhale20Hz_CallLibrary_Rannankari.xlsx --png-only --sample-size 50
  
  # Process entire dataset efficiently: MAT files only, cleanup audio after (parallel incremental mode)
  python scripts/whale_call_analysis.py --excel-file data/finwhales/FinWhale20Hz_CallLibrary_Rannankari.xlsx --process-all --mat-only --cleanup-audio
  
  # Use more workers for faster processing (if you have good bandwidth/CPU)
  python scripts/whale_call_analysis.py --excel-file data/finwhales/FinWhale20Hz_CallLibrary_Rannankari.xlsx --process-all --mat-only --cleanup-audio --workers 8
  
  # Use batch mode if you need the old processing method
  python scripts/whale_call_analysis.py --excel-file data/finwhales/FinWhale20Hz_CallLibrary_Rannankari.xlsx --sample-size 50 --batch-mode --cleanup-audio
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import time
import logging
import soundfile as sf
import yaml
import scipy.io
import concurrent.futures
import threading
from dotenv import load_dotenv

# Thread lock for matplotlib operations
_plot_lock = threading.Lock()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from onc_hydrophone_data.data.hydrophone_downloader import HydrophoneDownloader
from onc_hydrophone_data.audio import SpectrogramGenerator
from onc import ONC

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_status(message: str, status: str = "INFO"):
    """Print formatted status messages (thread-safe)"""
    status_symbols = {
        "INFO": "ℹ️",
        "SUCCESS": "✅", 
        "WARNING": "⚠️",
        "ERROR": "❌",
        "PROGRESS": "🔄"
    }
    symbol = status_symbols.get(status, "ℹ️")
    # Thread-safe printing
    print(f"{symbol} {message}", flush=True)

def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

class FinWhaleCallAnalyzer:
    """
    Comprehensive fin whale call analysis tool that integrates Excel data,
    ONC API downloads, and custom spectrogram generation.
    """
    
    def __init__(self, onc_token: str, excel_file: Optional[str] = None, config_path: str = "./config/dataset_config.yaml", excel_files: Optional[List[str]] = None):
        """Initialize the analyzer with ONC credentials and one or more Excel file paths"""
        self.onc = ONC(onc_token)
        # Normalize excel files to a list
        if excel_files and len(excel_files) > 0:
            self.excel_files = [str(Path(p)) for p in excel_files]
        elif excel_file:
            self.excel_files = [str(Path(excel_file))]
        else:
            raise ValueError("At least one Excel file must be provided")
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Load whale call data
        self.whale_data = None
        self.load_whale_data()
        
        # Initialize components (use current directory as base for downloads)
        self.downloader = HydrophoneDownloader(onc_token, ".")
        self.spectrogram_generator = None
    
    def _create_safe_call_id(self, clip_id, call):
        """
        Create a safe call ID, handling NaN values in timing data
        
        Args:
            clip_id: The audio clip ID
            call: The call row from the DataFrame
            
        Returns:
            str: Safe call ID or None if timing data is invalid
        """
        begin_time_raw = call['begin time (s)']
        end_time_raw = call['end time (s)']
        
        # Check for NaN values
        if pd.isna(begin_time_raw) or pd.isna(end_time_raw):
            return None
            
        try:
            begin_time = float(begin_time_raw)
            end_time = float(end_time_raw)
            
            # Validate that times are reasonable (allow zero-length: begin == end)
            if begin_time < 0 or end_time < 0 or begin_time > end_time:
                return None
                
            call_id = f"{clip_id}_{begin_time:.1f}s_{end_time:.1f}s"
            return call_id.replace('.wav', '').replace(':', '-').replace(' ', '_')
        except (ValueError, TypeError):
            return None
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print_status(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            print_status(f"Config file not found: {config_path}, using defaults", "WARNING")
            return {}
        except Exception as e:
            print_status(f"Error loading config: {e}, using defaults", "WARNING")
            return {}
        
    def load_whale_data(self):
        """Load and preprocess fin whale call library data from one or more Excel files"""
        print_status("Loading fin whale call library...")
        # Validate and read all files, then concatenate
        dataframes = []
        for p in self.excel_files:
            path_obj = Path(p)
            if not path_obj.exists():
                raise FileNotFoundError(f"Whale call library not found: {p}")
            try:
                df = pd.read_excel(path_obj)
                # Normalize column names: strip whitespace
                df.columns = [str(c).strip() for c in df.columns]
                # Map 40Hz schema -> unified schema expected by pipeline
                rename_map = {}
                # Clip ID
                if 'Clip ID' not in df.columns and 'Clip_identifier' in df.columns:
                    rename_map['Clip_identifier'] = 'Clip ID'
                # Date (UTC)
                if 'Date (UTC)' not in df.columns and 'Date_UTC' in df.columns:
                    rename_map['Date_UTC'] = 'Date (UTC)'
                # Begin/end seconds
                if 'begin time (s)' not in df.columns and 'begin_ time_s' in df.columns:
                    rename_map['begin_ time_s'] = 'begin time (s)'
                if 'begin time (s)' not in df.columns and 'begin_time_s' in df.columns:
                    rename_map['begin_time_s'] = 'begin time (s)'
                if 'end time (s)' not in df.columns and 'end_time_s' in df.columns:
                    rename_map['end_time_s'] = 'end time (s)'
                # Frequencies
                if 'low freq' not in df.columns and 'low_freq_Hz' in df.columns:
                    rename_map['low_freq_Hz'] = 'low freq'
                if 'high freq' not in df.columns and 'high_freq_Hz' in df.columns:
                    rename_map['high_freq_Hz'] = 'high freq'
                # Handle odd peak name variants
                if 'peak freq' not in df.columns:
                    if 'peak_ freq_Hz' in df.columns:
                        rename_map['peak_ freq_Hz'] = 'peak freq'
                    elif 'peak_freq_Hz' in df.columns:
                        rename_map['peak_freq_Hz'] = 'peak freq'
                # Robust, case/format-insensitive fallbacks for freq columns
                def _norm(name: str) -> str:
                    return str(name).lower().replace(' ', '').replace('_', '').replace('-', '')
                norm_to_orig = { _norm(c): c for c in df.columns }
                # Low frequency variants
                if 'low freq' not in df.columns:
                    for cand in ['lowfreqhz', 'lowfreq', 'lowfrehz']:
                        if cand in norm_to_orig:
                            rename_map[norm_to_orig[cand]] = 'low freq'
                            break
                # High frequency variants
                if 'high freq' not in df.columns:
                    for cand in ['highfreqhz', 'highfreq']:
                        if cand in norm_to_orig:
                            rename_map[norm_to_orig[cand]] = 'high freq'
                            break
                # Peak frequency variants
                if 'peak freq' not in df.columns:
                    for cand in ['peakfreqhz', 'peakfreq']:
                        if cand in norm_to_orig:
                            rename_map[norm_to_orig[cand]] = 'peak freq'
                            break
                if rename_map:
                    df = df.rename(columns=rename_map)
                # Ensure Duration (s) exists (compute if missing)
                if 'Duration (s)' not in df.columns and {'begin time (s)', 'end time (s)'} <= set(df.columns):
                    with pd.option_context('mode.use_inf_as_na', True):
                        b = pd.to_numeric(df['begin time (s)'], errors='coerce')
                        e = pd.to_numeric(df['end time (s)'], errors='coerce')
                        df['Duration (s)'] = e - b
                df['__source_file__'] = str(path_obj)
                dataframes.append(df)
                print_status(f"Loaded annotations from {path_obj}")
            except Exception as e:
                raise RuntimeError(f"Failed reading Excel file {p}: {e}")
        if not dataframes:
            raise ValueError("No annotation data loaded from provided Excel files")
        self.whale_data = pd.concat(dataframes, ignore_index=True, sort=False)
        
        # Clean and preprocess data
        print_status(f"Loaded {len(self.whale_data)} whale call records across {len(self.excel_files)} files")
        
        # Derive source band (20Hz/40Hz) from source file path
        self.whale_data['source_band'] = self.whale_data['__source_file__'].apply(self._classify_source_file) if '__source_file__' in self.whale_data.columns else 'unknown'

        # Extract device codes from clip IDs
        self.whale_data['device_code'] = self.whale_data['Clip ID'].str.extract(r'(ICLISTENHF\d+)')
        
        # Convert dates and times
        self.whale_data['Date (UTC)'] = pd.to_datetime(self.whale_data['Date (UTC)'])
        
        def convert_time_to_seconds(series):
            """Convert time format (MM:SS.ms) to seconds"""
            # First try direct numeric conversion
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            # For failed conversions, try time format conversion
            failed_mask = numeric_series.isna()
            if failed_mask.any():
                failed_indices = series[failed_mask].index
                for idx in failed_indices:
                    val = series.loc[idx]
                    try:
                        if hasattr(val, 'hour'):
                            # Convert time object: treat hour as minutes, minute as seconds, second as centiseconds
                            # So 3:54:24 = 3 minutes + 54 seconds + 24/100 seconds = 234.24 seconds
                            total_seconds = val.hour * 60 + val.minute + val.second / 100.0
                            if hasattr(val, 'microsecond'):
                                total_seconds += val.microsecond / 1e6
                            numeric_series.loc[idx] = total_seconds
                        elif isinstance(val, str):
                            # Try parsing as time format MM:SS:ms
                            time_parts = val.split(':')
                            if len(time_parts) == 3:
                                minutes, seconds, milliseconds = map(float, time_parts)
                                total_seconds = minutes * 60 + seconds + milliseconds / 100.0
                                numeric_series.loc[idx] = total_seconds
                    except:
                        pass  # Keep as NaN if conversion fails
            
            return numeric_series
        
        # Clean duration and timing columns with proper time conversion
        timing_cols = ['begin time (s)', 'end time (s)', 'Duration (s)']
        for col in timing_cols:
            if col in self.whale_data.columns:
                self.whale_data[col] = convert_time_to_seconds(self.whale_data[col])
        
        # Clean frequency columns normally
        freq_cols = ['low freq', 'high freq', 'peak freq']
        for col in freq_cols:
            if col in self.whale_data.columns:
                self.whale_data[col] = pd.to_numeric(self.whale_data[col], errors='coerce')
        
        # Filter for valid calls (handle whitespace in Call Category)
        mask = (
            (self.whale_data['device_code'].notna()) &
            (self.whale_data['Clip ID'].str.endswith('.wav')) &
            (self.whale_data['begin time (s)'].notna()) &
            (self.whale_data['end time (s)'].notna()) &
            (self.whale_data['begin time (s)'] >= 0) &
            (self.whale_data['end time (s)'] >= 0) &
            (self.whale_data['begin time (s)'] <= self.whale_data['end time (s)'])
        )
        
        invalid_timing_count = len(self.whale_data) - mask.sum()
        if invalid_timing_count > 0:
            print_status(f"Filtered out {invalid_timing_count} calls with invalid timing data", "WARNING")
        
        self.whale_data = self.whale_data[mask].copy()
        print_status(f"Filtered to {len(self.whale_data)} valid fin whale calls")
        
        # Show summary statistics
        self.print_data_summary()
    
    @staticmethod
    def _classify_source_file(path_str: str) -> str:
        p = (str(path_str) or "").lower()
        if "40hz" in p:
            return "40Hz"
        if "20hz" in p:
            return "20Hz"
        return "unknown"

    def print_data_summary(self):
        """Print summary statistics of the whale call data"""
        print_header("WHALE CALL LIBRARY SUMMARY")
        
        print(f"📊 Total valid calls: {len(self.whale_data):,}")
        print(f"🎛️  Unique devices: {self.whale_data['device_code'].nunique()}")
        print(f"📅 Date range: {self.whale_data['Date (UTC)'].min().date()} to {self.whale_data['Date (UTC)'].max().date()}")
        print(f"⏱️  Duration range: {self.whale_data['Duration (s)'].min():.1f}s to {self.whale_data['Duration (s)'].max():.1f}s")
        print(f"🔊 Frequency range: {self.whale_data['low freq'].min():.0f} to {self.whale_data['high freq'].max():.0f} Hz")
        
        print("\n🎛️  Top devices by call count:")
        device_counts = self.whale_data['device_code'].value_counts().head()
        for device, count in device_counts.items():
            print(f"   {device}: {count:,} calls")
    
    def sample_calls(self, 
                    sample_size: Optional[int] = 20,
                    device_filter: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    min_duration: float = 1.0,
                    max_duration: float = 30.0,
                    freq_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Intelligently sample whale calls based on specified criteria.
        
        Args:
            sample_size: Number of calls to sample
            device_filter: Specific device code to filter by
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            min_duration: Minimum call duration in seconds
            max_duration: Maximum call duration in seconds  
            freq_range: Frequency range filter (min_freq, max_freq) in Hz
            
        Returns:
            DataFrame with sampled whale calls
        """
        print_header("SAMPLING WHALE CALLS")
        
        # Start with all valid data
        filtered_data = self.whale_data.copy()
        
        # Apply filters
        if device_filter:
            filtered_data = filtered_data[filtered_data['device_code'] == device_filter]
            print_status(f"Filtered by device {device_filter}: {len(filtered_data)} calls")
            
        if start_date:
            start_dt = pd.to_datetime(start_date)
            filtered_data = filtered_data[filtered_data['Date (UTC)'] >= start_dt]
            print_status(f"Filtered by start date {start_date}: {len(filtered_data)} calls")
            
        if end_date:
            end_dt = pd.to_datetime(end_date)
            filtered_data = filtered_data[filtered_data['Date (UTC)'] <= end_dt]
            print_status(f"Filtered by end date {end_date}: {len(filtered_data)} calls")
            
        if min_duration or max_duration:
            duration_mask = (
                (filtered_data['Duration (s)'] >= min_duration) &
                (filtered_data['Duration (s)'] <= max_duration)
            )
            filtered_data = filtered_data[duration_mask]
            print_status(f"Filtered by duration {min_duration}-{max_duration}s: {len(filtered_data)} calls")
            
        if freq_range:
            min_freq, max_freq = freq_range
            freq_mask = (
                (filtered_data['low freq'] >= min_freq) &
                (filtered_data['high freq'] <= max_freq)
            )
            filtered_data = filtered_data[freq_mask]
            print_status(f"Filtered by frequency {min_freq}-{max_freq} Hz: {len(filtered_data)} calls")
        
        if len(filtered_data) == 0:
            raise ValueError("No calls match the specified criteria")
        
        # Show filtered data summary
        if len(filtered_data) != len(self.whale_data):
            print_status(f"📊 After filtering: {len(filtered_data):,} calls")
            if len(filtered_data) > 0:
                print_status(f"📅 Filtered date range: {filtered_data['Date (UTC)'].min().date()} to {filtered_data['Date (UTC)'].max().date()}")
                print_status(f"🎛️ Devices in filtered data: {filtered_data['device_code'].nunique()}")
        elif sample_size is None:
            print_status(f"📊 No filters applied: processing entire dataset ({len(filtered_data):,} calls)")
        
        # Handle sample size
        if sample_size is None:
            # Process entire dataset
            sample_size = len(filtered_data)
            if len(filtered_data) == len(self.whale_data):
                print_status(f"🚀 Processing entire unfiltered dataset: {sample_size:,} calls", "INFO")
            else:
                print_status(f"🚀 Processing entire filtered dataset: {sample_size:,} calls", "INFO")
        elif len(filtered_data) < sample_size:
            print_status(f"Only {len(filtered_data)} calls available, sampling all", "WARNING")
            sample_size = len(filtered_data)
        
        # Intelligent sampling strategy:
        # 1. Diversify by device and date
        # 2. Prefer calls with good frequency characteristics
        # 3. Include various duration ranges
        
        sampled_calls = []
        remaining_sample_size = sample_size
        
        # Sample by device to ensure diversity
        devices = filtered_data['device_code'].unique()
        calls_per_device = max(1, remaining_sample_size // len(devices))
        
        for device in devices:
            if remaining_sample_size <= 0:
                break
                
            device_data = filtered_data[filtered_data['device_code'] == device]
            
            # Sample from this device, diversifying by date
            device_sample_size = min(calls_per_device, remaining_sample_size, len(device_data))
            
            if device_sample_size > 0:
                # Sort by date and sample evenly across time range
                device_data_sorted = device_data.sort_values('Date (UTC)')
                indices = np.linspace(0, len(device_data_sorted)-1, device_sample_size, dtype=int)
                device_sample = device_data_sorted.iloc[indices]
                
                sampled_calls.append(device_sample)
                remaining_sample_size -= device_sample_size
        
        # If we still need more samples, fill randomly from remaining data
        if remaining_sample_size > 0:
            used_indices = pd.concat(sampled_calls).index if sampled_calls else pd.Index([])
            remaining_data = filtered_data.drop(used_indices)
            
            if len(remaining_data) > 0:
                additional_sample_size = min(remaining_sample_size, len(remaining_data))
                additional_sample = remaining_data.sample(n=additional_sample_size, random_state=42)
                sampled_calls.append(additional_sample)
        
        # Combine all samples
        final_sample = pd.concat(sampled_calls, ignore_index=True) if sampled_calls else pd.DataFrame()
        
        print_status(f"Sampled {len(final_sample)} whale calls for analysis", "SUCCESS")
        return final_sample
    
    def download_whale_call_audio(self, whale_calls: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
        """
        Download the specific .wav files containing sampled whale calls.
        
        Args:
            whale_calls: DataFrame with sampled calls
            output_dir: Directory to save downloaded files
            
        Returns:
            Dictionary mapping clip IDs to downloaded file paths
        """
        print_header("DOWNLOADING WHALE CALL AUDIO FILES")
        
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique clip IDs
        unique_clips = whale_calls['Clip ID'].unique()
        print_status(f"Need to download {len(unique_clips)} unique audio files")
        
        downloaded_files = {}
        
        # Temporarily set ONC output path
        original_output_path = self.onc.outPath
        self.onc.outPath = str(audio_dir)
        
        try:
            for i, clip_id in enumerate(unique_clips, 1):
                print_status(f"Downloading {i}/{len(unique_clips)}: {clip_id}", "PROGRESS")
                
                try:
                    # Check if file already exists
                    downloaded_path = audio_dir / clip_id
                    if downloaded_path.exists():
                        downloaded_files[clip_id] = str(downloaded_path)
                        print_status(f"✓ Already exists: {clip_id}")
                        continue
                        
                    # Extract device and date info from filename for search
                    # Format: ICLISTENHF1353_20180710T183452.152Z.wav
                    device_match = clip_id.split('_')[0] if '_' in clip_id else None
                    
                    if device_match:
                        # Try to download the specific file
                        result = self.onc.getFile(clip_id)
                        
                        if result:
                            if downloaded_path.exists():
                                downloaded_files[clip_id] = str(downloaded_path)
                                print_status(f"✓ Downloaded: {clip_id}")
                            else:
                                print_status(f"⚠️  Download reported success but file not found: {clip_id}", "WARNING")
                        else:
                            print_status(f"❌ Failed to download: {clip_id}", "WARNING")
                            
                except Exception as e:
                    print_status(f"❌ Error downloading {clip_id}: {e}", "WARNING")
                    continue
                    
        finally:
            # Restore original output path
            self.onc.outPath = original_output_path
            
        print_status(f"Successfully downloaded {len(downloaded_files)}/{len(unique_clips)} audio files", "SUCCESS")
        return downloaded_files

    def _stitch_audio_files(self, call: pd.Series, desired_start: float, desired_end: float, context_duration: float, audio_dir: Path) -> Optional[np.ndarray]:
        """
        Stitch audio files when context window spans multiple files.
        
        Args:
            call: Row from calls DataFrame
            desired_start: Desired start time (can be negative)
            desired_end: Desired end time (can exceed current file)
            context_duration: Total context duration needed
            
        Returns:
            Stitched audio array or None if stitching fails
        """
        # Parse current filename to understand the temporal structure
        current_filename = call['Clip ID']
        device_code = call['device_code']
        
        # Extract timestamp from filename (format: ICLISTENHF1353_20180725T063328.510Z.wav)
        import re
        timestamp_match = re.search(r'(\d{8}T\d{6}\.\d{3}Z)', current_filename)
        if not timestamp_match:
            print_status(f"❌ Could not parse timestamp from: {current_filename}", "ERROR")
            return None
            
        current_timestamp_str = timestamp_match.group(1)
        try:
            current_timestamp = pd.to_datetime(current_timestamp_str, format='%Y%m%dT%H%M%S.%fZ')
            print_status(f"📅 Parsed timestamp: {current_timestamp}", "INFO")
        except Exception as e:
            print_status(f"❌ Could not parse timestamp {current_timestamp_str}: {e}", "ERROR")
            return None
        
        # Load current file
        current_path = audio_dir / current_filename
        current_audio, sample_rate = sf.read(current_path)
        current_duration = len(current_audio) / sample_rate
        
        stitched_audio = []
        
        # Handle previous file if needed
        if desired_start < 0:
            prev_timestamp = current_timestamp - pd.Timedelta(seconds=300)  # ONC files are 5min (300s)
            prev_filename = f"{device_code}_{prev_timestamp.strftime('%Y%m%dT%H%M%S.%f')[:-3]}Z.wav"
            prev_path = audio_dir / prev_filename
            
            if prev_path.exists():
                prev_audio, _ = sf.read(prev_path)
                prev_duration = len(prev_audio) / sample_rate
                
                # Calculate how much we need from previous file
                needed_from_prev = -desired_start
                start_in_prev = max(0, prev_duration - needed_from_prev)
                
                prev_segment = prev_audio[int(start_in_prev * sample_rate):]
                stitched_audio.append(prev_segment)
                print_status(f"📎 Stitched {len(prev_segment)/sample_rate:.1f}s from previous file: {prev_filename}", "INFO")
            else:
                # Download previous file if it doesn't exist
                if not self._download_adjacent_file(device_code, prev_timestamp, audio_dir):
                    return None
                
                if prev_path.exists():
                    prev_audio, _ = sf.read(prev_path)
                    prev_duration = len(prev_audio) / sample_rate
                    needed_from_prev = -desired_start
                    start_in_prev = max(0, prev_duration - needed_from_prev)
                    prev_segment = prev_audio[int(start_in_prev * sample_rate):]
                    stitched_audio.append(prev_segment)
                    print_status(f"📎 Downloaded & stitched {len(prev_segment)/sample_rate:.1f}s from previous file: {prev_filename}", "INFO")
                else:
                    return None
        
        # Add current file segment
        current_start = max(0, desired_start)
        current_end = min(current_duration, desired_end)
        current_start_sample = int(current_start * sample_rate)
        current_end_sample = int(current_end * sample_rate)
        current_segment = current_audio[current_start_sample:current_end_sample]
        stitched_audio.append(current_segment)
        
        # Handle next file if needed
        if desired_end > current_duration:
            next_timestamp = current_timestamp + pd.Timedelta(seconds=300)  # ONC files are 5min (300s)
            next_filename = f"{device_code}_{next_timestamp.strftime('%Y%m%dT%H%M%S.%f')[:-3]}Z.wav"
            next_path = audio_dir / next_filename
            
            if next_path.exists():
                next_audio, _ = sf.read(next_path)
                
                # Calculate how much we need from next file
                needed_from_next = desired_end - current_duration
                end_in_next = min(len(next_audio) / sample_rate, needed_from_next)
                
                next_segment = next_audio[:int(end_in_next * sample_rate)]
                stitched_audio.append(next_segment)
                print_status(f"📎 Stitched {len(next_segment)/sample_rate:.1f}s from next file: {next_filename}", "INFO")
            else:
                # Download next file if it doesn't exist
                if not self._download_adjacent_file(device_code, next_timestamp, audio_dir):
                    return None
                
                if next_path.exists():
                    next_audio, _ = sf.read(next_path)
                    needed_from_next = desired_end - current_duration
                    end_in_next = min(len(next_audio) / sample_rate, needed_from_next)
                    next_segment = next_audio[:int(end_in_next * sample_rate)]
                    stitched_audio.append(next_segment)
                    print_status(f"📎 Downloaded & stitched {len(next_segment)/sample_rate:.1f}s from next file: {next_filename}", "INFO")
                else:
                    return None
        
        # Concatenate all segments
        if stitched_audio:
            final_audio = np.concatenate(stitched_audio)
            
            # Ensure exact duration
            target_samples = int(context_duration * sample_rate)
            if len(final_audio) > target_samples:
                final_audio = final_audio[:target_samples]
            elif len(final_audio) < target_samples:
                padding_needed = target_samples - len(final_audio)
                final_audio = np.pad(final_audio, (0, padding_needed), mode='constant')
                
            return final_audio
        
        return None
    
    def _stitch_audio_files_from_clip(self, clip_id: str, device_code: str, desired_start: float, desired_end: float, context_duration: float, audio_dir: Path) -> Optional[np.ndarray]:
        """
        Stitch audio when the desired window spans multiple files, using only clip_id and device code.
        """
        import re
        timestamp_match = re.search(r'(\d{8}T\d{6}\.\d{3}Z)', clip_id)
        if not timestamp_match:
            print_status(f"❌ Could not parse timestamp from: {clip_id}", "ERROR")
            return None
        current_timestamp_str = timestamp_match.group(1)
        try:
            current_timestamp = pd.to_datetime(current_timestamp_str, format='%Y%m%dT%H%M%S.%fZ')
        except Exception as e:
            print_status(f"❌ Could not parse timestamp {current_timestamp_str}: {e}", "ERROR")
            return None
        # Load current file
        current_path = audio_dir / clip_id
        current_audio, sample_rate = sf.read(current_path)
        current_duration = len(current_audio) / sample_rate
        stitched_audio = []
        # Prev file if needed
        if desired_start < 0:
            prev_timestamp = current_timestamp - pd.Timedelta(seconds=300)
            prev_filename = f"{device_code}_{prev_timestamp.strftime('%Y%m%dT%H%M%S.%f')[:-3]}Z.wav"
            prev_path = audio_dir / prev_filename
            if not prev_path.exists():
                if not self._download_adjacent_file(device_code, prev_timestamp, audio_dir):
                    return None
            if prev_path.exists():
                prev_audio, _ = sf.read(prev_path)
                prev_duration = len(prev_audio) / sample_rate
                needed_from_prev = -desired_start
                start_in_prev = max(0, prev_duration - needed_from_prev)
                prev_segment = prev_audio[int(start_in_prev * sample_rate):]
                stitched_audio.append(prev_segment)
        # Current segment
        current_start = max(0, desired_start)
        current_end = min(current_duration, desired_end)
        current_segment = current_audio[int(current_start * sample_rate):int(current_end * sample_rate)]
        stitched_audio.append(current_segment)
        # Next file if needed
        if desired_end > current_duration:
            next_timestamp = current_timestamp + pd.Timedelta(seconds=300)
            next_filename = f"{device_code}_{next_timestamp.strftime('%Y%m%dT%H%M%S.%f')[:-3]}Z.wav"
            next_path = audio_dir / next_filename
            if not next_path.exists():
                if not self._download_adjacent_file(device_code, next_timestamp, audio_dir):
                    return None
            if next_path.exists():
                next_audio, _ = sf.read(next_path)
                needed_from_next = desired_end - current_duration
                next_segment = next_audio[:int(min(len(next_audio) / sample_rate, needed_from_next) * sample_rate)]
                stitched_audio.append(next_segment)
        if stitched_audio:
            final_audio = np.concatenate(stitched_audio)
            target_samples = int(context_duration * sample_rate)
            if len(final_audio) > target_samples:
                final_audio = final_audio[:target_samples]
            elif len(final_audio) < target_samples:
                final_audio = np.pad(final_audio, (0, target_samples - len(final_audio)), mode='constant')
            return final_audio
        return None
    
    def _compute_free_intervals(self, occupied_intervals: List[Tuple[float, float]], file_duration: float, margin: float = 0.0) -> List[Tuple[float, float]]:
        """Compute free intervals in [0, file_duration] given occupied intervals, with optional margins around calls."""
        if not occupied_intervals:
            return [(0.0, file_duration)]
        # Expand by margin and clamp
        expanded = []
        for b, e in occupied_intervals:
            b2 = max(0.0, b - margin)
            e2 = min(file_duration, e + margin)
            expanded.append((b2, e2))
        # Merge overlaps
        expanded.sort()
        merged = []
        for b, e in expanded:
            if not merged or b > merged[-1][1]:
                merged.append([b, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        merged = [(b, e) for b, e in merged]
        # Free intervals are gaps
        free = []
        cursor = 0.0
        for b, e in merged:
            if b > cursor:
                free.append((cursor, b))
            cursor = max(cursor, e)
        if cursor < file_duration:
            free.append((cursor, file_duration))
        return free
    
    def _largest_prefix_free(self, free_intervals: List[Tuple[float, float]]) -> float:
        """Return length of free region contiguous from start (t=0)."""
        total = 0.0
        cursor = 0.0
        for b, e in free_intervals:
            if b > cursor:
                break
            if b <= cursor < e:
                total += e - cursor
                cursor = e
        return total
    
    def _largest_suffix_free(self, free_intervals: List[Tuple[float, float]], file_duration: float) -> float:
        """Return length of free region contiguous to end (t=file_duration)."""
        total = 0.0
        cursor = file_duration
        for b, e in reversed(free_intervals):
            if e < cursor:
                break
            if b < cursor <= e:
                total += cursor - b
                cursor = b
        return total
    
    def _get_prev_next_filenames(self, clip_id: str, device_code: str) -> Tuple[str, str]:
        import re
        timestamp_match = re.search(r'(\d{8}T\d{6}\.\d{3}Z)', clip_id)
        if not timestamp_match:
            return None, None
        ts = pd.to_datetime(timestamp_match.group(1), format='%Y%m%dT%H%M%S.%fZ')
        prev_ts = ts - pd.Timedelta(seconds=300)
        next_ts = ts + pd.Timedelta(seconds=300)
        prev_filename = f"{device_code}_{prev_ts.strftime('%Y%m%dT%H%M%S.%f')[:-3]}Z.wav"
        next_filename = f"{device_code}_{next_ts.strftime('%Y%m%dT%H%M%S.%f')[:-3]}Z.wav"
        return prev_filename, next_filename
    
    def _create_file_overview_spectrogram(self,
                                          clip_id: str,
                                          device_code: str,
                                          audio_file_path: Path,
                                          output_dir: Path,
                                          intervals_20: List[Tuple[float, float]],
                                          intervals_40: List[Tuple[float, float]],
                                          neg_windows: List[Tuple[float, float]],
                                          win_dur: float,
                                          overlap: float,
                                          freq_range: Tuple[float, float]) -> Optional[str]:
        """Create a full-file spectrogram PNG with overlays for calls (20Hz/40Hz) and negative windows."""
        try:
            audio_data, sample_rate = self.spectrogram_generator.load_audio(str(audio_file_path)) if self.spectrogram_generator else sf.read(audio_file_path)
            if isinstance(audio_data, tuple):
                audio_data, sample_rate = audio_data  # handle sf.read branch
        except Exception as e:
            print_status(f"❌ Failed to load audio for overview {clip_id}: {e}", "WARNING")
            return None
        # Resolve plotting params similar to positives
        config_spectrograms = self.config.get('custom_spectrograms', {})
        resolved_win_dur = win_dur if win_dur != 0.1 else config_spectrograms.get('window_duration', 2.0)
        resolved_overlap = overlap if overlap != 0.9 else config_spectrograms.get('overlap', 0.985)
        if freq_range != (5, 100):
            resolved_freq_lims = freq_range
        else:
            config_freq = config_spectrograms.get('frequency_limits', {})
            resolved_freq_lims = (config_freq.get('min', 5), config_freq.get('max', 100))
        colormap = config_spectrograms.get('colormap', 'viridis')
        config_clim = config_spectrograms.get('color_limits', {})
        clim = (config_clim.get('min', -40), config_clim.get('max', 0))
        log_freq = config_spectrograms.get('log_frequency', False)
        # Ensure generator exists with these params
        if self.spectrogram_generator is None:
            self.spectrogram_generator = SpectrogramGenerator(
                win_dur=resolved_win_dur,
                overlap=resolved_overlap,
                freq_lims=resolved_freq_lims,
                colormap=colormap,
                clim=clim,
                log_freq=log_freq,
                max_duration=None
            )
        frequencies, times, _, power_db_norm = self.spectrogram_generator.compute_spectrogram(audio_data, sample_rate)
        # Crop to whale band for consistency
        freq_min, freq_max = 5, 100
        freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
        frequencies_c = frequencies[freq_mask]
        power_c = power_db_norm[freq_mask, :]
        fig = self.spectrogram_generator.plot_spectrogram(
            frequencies_c, times, power_c,
            title=f"File Overview - {device_code} | {clip_id}\nCalls (20/40Hz) and Negative Windows"
        )
        try:
            ax = fig.axes[0]
        except Exception:
            ax = plt.gca()
        # Overlays: calls and negatives
        import matplotlib.patches as patches
        # Calls overlays as translucent spans
        for b, e in intervals_20:
            ax.axvspan(b, e, color='tab:blue', alpha=0.25, label='20Hz call')
        for b, e in intervals_40:
            ax.axvspan(b, e, color='tab:red', alpha=0.25, label='40Hz call')
        # Negative windows in green
        for b, e in neg_windows:
            ax.axvspan(b, e, color='tab:green', alpha=0.35, label='Negative window')
        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq_h.append(h)
                uniq_l.append(l)
                seen.add(l)
        if uniq_h:
            ax.legend(uniq_h, uniq_l, loc='upper right')
        # Save
        out_dir = output_dir / "file_overview_png"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{clip_id.replace('.wav','')}_overview.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print_status(f"✓ Created file overview PNG: {out_path.name}")
        return str(out_path)

    def _sample_negative_windows_for_file(self,
                                          clip_id: str,
                                          device_code: str,
                                          file_duration: float,
                                          context_duration: float,
                                          calls_by_file: Dict[str, List[Tuple[float, float]]],
                                          max_windows: int,
                                          margin: float = 0.0) -> List[Tuple[float, float]]:
        """Return up to max_windows [start, end] pairs (relative to current file) that avoid any calls (with margin), sampled randomly within-file.
        Limits heavy overlap between negatives while allowing small overlaps.
        """
        def overlap_amount(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

        windows: List[Tuple[float, float]] = []
        occupied = sorted(calls_by_file.get(clip_id, []))
        free = self._compute_free_intervals(occupied, file_duration, margin)
        # Keep only intervals that can fit the context exactly
        candidate_intervals = [(b, e) for (b, e) in free if (e - b) >= context_duration]
        if not candidate_intervals or max_windows <= 0:
            return windows
        rng = np.random.default_rng()
        max_overlap_ratio = 0.3  # allow up to 30% overlap between negatives
        allowed_overlap = context_duration * max_overlap_ratio
        # Attempt random sampling across intervals
        attempts = 0
        max_attempts = max(50, max_windows * 20)
        while len(windows) < max_windows and attempts < max_attempts:
            attempts += 1
            # Randomly choose an interval (weighted by available start range could be added later)
            b, e = candidate_intervals[rng.integers(0, len(candidate_intervals))]
            start_min = b
            start_max = e - context_duration
            if start_max < start_min:
                continue
            start = float(rng.uniform(start_min, start_max))
            end = start + context_duration
            # Bounds check
            if start < 0 or end > file_duration or end > e:
                continue
            cand = (start, end)
            # Overlap check vs already chosen windows
            too_much = False
            for w in windows:
                if overlap_amount(cand, w) > allowed_overlap:
                    too_much = True
                    break
            if too_much:
                continue
            windows.append(cand)
        return windows
    
    def create_negative_spectrograms(self,
                                     clip_id: str,
                                     device_code: str,
                                     windows: List[Tuple[float, float]],
                                     output_dir: Path,
                                     win_dur: float,
                                     overlap: float,
                                     freq_range: Tuple[float, float],
                                     context_duration: float,
                                     calls_in_file_df: Optional[pd.DataFrame] = None,
                                     generate_overview: bool = False) -> Tuple[Dict[str, str], List[Dict]]:
        """Create negative spectrograms for provided windows for a single clip."""
        # Output format settings from config
        config_spectrograms = self.config.get('custom_spectrograms', {})
        config_formats = config_spectrograms.get('output_formats', {})
        save_matlab = config_formats.get('matlab', False)
        save_plots = config_formats.get('plots', True)
        # Directories
        if save_matlab and save_plots:
            mat_dir = output_dir / "neg_mat_files"
            png_dir = output_dir / "neg_png_files"
        elif save_matlab:
            mat_dir = output_dir / "neg_mat_files"
            png_dir = None
        elif save_plots:
            mat_dir = None
            png_dir = output_dir / "neg_png_files"
        else:
            mat_dir = png_dir = output_dir / "negative_spectrograms"
        if mat_dir:
            mat_dir.mkdir(parents=True, exist_ok=True)
        if png_dir:
            png_dir.mkdir(parents=True, exist_ok=True)
        # Resolve spectrogram params same as positives
        # Use command line args if provided, otherwise use config values with whale-optimized fallbacks
        resolved_win_dur = win_dur if win_dur != 0.1 else config_spectrograms.get('window_duration', 2.0)
        resolved_overlap = overlap if overlap != 0.9 else config_spectrograms.get('overlap', 0.985)
        if freq_range != (5, 100):
            resolved_freq_lims = freq_range
        else:
            config_freq = config_spectrograms.get('frequency_limits', {})
            resolved_freq_lims = (config_freq.get('min', 5), config_freq.get('max', 100))
        # Visual params
        colormap = config_spectrograms.get('colormap', 'viridis')
        config_clim = config_spectrograms.get('color_limits', {})
        clim = (config_clim.get('min', -40), config_clim.get('max', 0))
        log_freq = config_spectrograms.get('log_frequency', False)
        # Initialize generator if needed
        if self.spectrogram_generator is None:
            self.spectrogram_generator = SpectrogramGenerator(
                win_dur=resolved_win_dur,
                overlap=resolved_overlap,
                freq_lims=resolved_freq_lims,
                colormap=colormap,
                clim=clim,
                log_freq=log_freq,
                max_duration=None
            )
        spectrogram_files: Dict[str, str] = {}
        failed: List[Dict] = []
        # Audio dir
        audio_dir = output_dir / "audio"
        audio_path = audio_dir / clip_id
        # Load current audio to get sample rate and duration (if fully inside)
        try:
            audio_data, sample_rate = self.spectrogram_generator.load_audio(str(audio_path))
            current_duration = len(audio_data) / sample_rate
        except Exception as e:
            for start, end in windows:
                failed.append({
                    'clip_id': clip_id,
                    'call_id': f"{clip_id}_neg_{start:.1f}s_{end:.1f}s",
                    'reason': f'Negative audio load error: {str(e)}'
                })
            return spectrogram_files, failed
        # Prepare call summaries for this file (20 Hz vs 40 Hz) if provided
        intervals_20: List[Tuple[float, float]] = []
        intervals_40: List[Tuple[float, float]] = []
        if calls_in_file_df is not None and not calls_in_file_df.empty:
            def classify_source(path_str: str) -> str:
                p = (path_str or "").lower()
                if "40hz" in p:
                    return "40Hz"
                if "20hz" in p:
                    return "20Hz"
                return "unknown"
            for _, row in calls_in_file_df.iterrows():
                try:
                    b = float(row['begin time (s)'])
                    e = float(row['end time (s)'])
                    if not (b < e):
                        continue
                    src = classify_source(row.get('__source_file__', ''))
                    if src == "40Hz":
                        intervals_40.append((b, e))
                    elif src == "20Hz":
                        intervals_20.append((b, e))
                except Exception:
                    continue
            intervals_20.sort()
            intervals_40.sort()

        # Process windows
        overview_generated = False
        neg_windows_done: List[Tuple[float, float]] = []
        for start, end in windows:
            # Build a clean base id without "neg"; we'll append _neg to filenames only
            call_id = f"{clip_id}_{start:.1f}s_{end:.1f}s".replace('.wav', '').replace(':', '-').replace(' ', '_')
            try:
                # Within-file only for negatives; require exact size (no edge padding)
                if start < 0 or end > current_duration:
                    failed.append({'clip_id': clip_id, 'call_id': call_id, 'reason': 'Negative window touches file boundary; skipped to maintain exact size'})
                    continue
                target_samples = int(round(context_duration * sample_rate))
                start_sample = int(round(start * sample_rate))
                end_sample = start_sample + target_samples
                if end_sample > len(audio_data):
                    failed.append({'clip_id': clip_id, 'call_id': call_id, 'reason': 'Negative window exceeds file samples after rounding; skipped'})
                    continue
                call_audio = audio_data[start_sample:end_sample]
                if len(call_audio) != target_samples:
                    failed.append({'clip_id': clip_id, 'call_id': call_id, 'reason': 'Negative window sample length mismatch; skipped'})
                    continue
                neg_windows_done.append((start, end))
                # Print verification summary for this negative window
                if intervals_20 or intervals_40:
                    def fmt_intervals(ints: List[Tuple[float, float]]) -> str:
                        return ", ".join([f"{b:.1f}-{e:.1f}s" for b, e in ints]) if ints else "none"
                    print_status(
                        f"NEG window {clip_id} @ {start:.1f}-{end:.1f}s | 20Hz calls: {len(intervals_20)} [{fmt_intervals(intervals_20)}] | 40Hz calls: {len(intervals_40)} [{fmt_intervals(intervals_40)}]",
                        "INFO"
                    )
                # Compute spectrogram
                frequencies, times, power, power_db_norm = self.spectrogram_generator.compute_spectrogram(call_audio, sample_rate)
                # Crop to whale range (keep consistent dims with positives)
                freq_min, freq_max = 5, 100
                freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
                frequencies = frequencies[freq_mask]
                power_db_norm = power_db_norm[freq_mask, :]
                # Save PNG
                if save_plots:
                    neg_src_tag = 'unknown'
                    if calls_in_file_df is not None and not calls_in_file_df.empty:
                        try:
                            if 'source_band' in calls_in_file_df.columns:
                                neg_src_tag = str(calls_in_file_df.iloc[0]['source_band'])
                            else:
                                neg_src_tag = self._classify_source_file(calls_in_file_df.iloc[0].get('__source_file__', ''))
                        except Exception:
                            neg_src_tag = 'unknown'
                    fig = self.spectrogram_generator.plot_spectrogram(
                        frequencies, times, power_db_norm,
                        title=f"No Fin Whale (negative; {neg_src_tag}) - {device_code} | Window: {start:.1f}s-{end:.1f}s"
                    )
                    out_png = png_dir / f"{call_id}_{neg_src_tag}_neg.png" if png_dir else None
                    if out_png:
                        fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
                        plt.close(fig)
                # Save MAT
                if save_matlab:
                    out_mat = mat_dir / f"{call_id}_{neg_src_tag}_neg.mat"
                    scipy.io.savemat(out_mat, {
                        'spectrogram': power_db_norm,
                        'frequencies': frequencies,
                        'times': times,
                        'negative_window': {
                            'start_s': float(start),
                            'end_s': float(end),
                            'context_duration_s': float(context_duration),
                        },
                        'clip_id': clip_id,
                        'device_code': device_code,
                        'source_band': neg_src_tag
                    })
                # Register
                if save_matlab:
                    spectrogram_files[call_id] = str(out_mat)
                elif save_plots and png_dir:
                    spectrogram_files[call_id] = str(out_png)
                else:
                    spectrogram_files[call_id] = f"processed_{call_id}"
                print_status(f"✓ Created negative spectrogram: {call_id}")
            except Exception as e:
                failed.append({'clip_id': clip_id, 'call_id': call_id, 'reason': f'Negative processing error: {str(e)}'})
                print_status(f"❌ Error creating negative spectrogram for {call_id}: {e}", "WARNING")
        # Create overview once per file after negatives, if any
        try:
            if generate_overview and neg_windows_done and (intervals_20 or intervals_40):
                audio_dir = output_dir / "audio"
                audio_path = audio_dir / clip_id
                self._create_file_overview_spectrogram(
                    clip_id, device_code, audio_path, output_dir,
                    intervals_20, intervals_40, neg_windows_done,
                    win_dur, overlap, freq_range
                )
        except Exception as e:
            print_status(f"⚠️ Failed to create overview for {clip_id}: {e}", "WARNING")
        return spectrogram_files, failed

    def _download_adjacent_file(self, device_code: str, timestamp: pd.Timestamp, audio_dir: Path) -> bool:
        """Download an adjacent audio file if needed."""
        filename = f"{device_code}_{timestamp.strftime('%Y%m%dT%H%M%S.%f')[:-3]}Z.wav"
        
        # Set output path to the provided audio directory
        original_output_path = self.onc.outPath
        audio_dir.mkdir(parents=True, exist_ok=True)
        self.onc.outPath = str(audio_dir)
        
        try:
            # Check if file already exists
            file_path = audio_dir / filename
            if file_path.exists():
                print_status(f"✓ Adjacent file already exists: {filename}", "INFO")
                return True
                
            print_status(f"🔄 Downloading adjacent file: {filename}", "INFO")
            result = self.onc.getFile(
                filename=filename,
                overwrite=False
            )
            if result and 'downloadResults' in result:
                for download_result in result['downloadResults']:
                    if download_result.get('status') == 'complete':
                        return True
            elif file_path.exists():
                # Sometimes the download succeeds but doesn't return proper status
                print_status(f"✓ Downloaded adjacent file: {filename}", "INFO")
                return True
        except Exception as e:
            error_msg = str(e)
            if "No file could be found" in error_msg:
                print_status(f"⚠️ Adjacent file doesn't exist in archive: {filename}", "WARNING")
            else:
                print_status(f"❌ Failed to download {filename}: {error_msg}", "ERROR")
        finally:
            # Restore original output path
            self.onc.outPath = original_output_path
        
        return False

    def create_custom_spectrograms(self, 
                                 whale_calls: pd.DataFrame,
                                 downloaded_files: Dict[str, str],
                                 output_dir: Path,
                                 win_dur: float = 2.0,
                                 overlap: float = 0.5,
                                 freq_range: Tuple[float, float] = (10, 1000),
                                 ml_context: Optional[float] = None) -> Tuple[Dict[str, str], List[Dict]]:
        """
        Create custom spectrograms focused on whale call timing and frequency.
        
        Args:
            whale_calls: DataFrame with call data
            downloaded_files: Mapping of clip IDs to audio file paths
            output_dir: Output directory
            win_dur: Window duration in seconds
            overlap: Overlap ratio for spectrogram
            freq_range: Frequency range for analysis
            
        Returns:
            Dictionary mapping call IDs to spectrogram file paths
        """
        print_header("GENERATING CUSTOM SPECTROGRAMS")
        
        # Get spectrogram parameters from config with command line overrides
        config_spectrograms = self.config.get('custom_spectrograms', {})
        
        # Use command line args if provided, otherwise use config values with whale-optimized fallbacks
        win_dur = win_dur if win_dur != 0.1 else config_spectrograms.get('window_duration', 2.0)
        overlap = overlap if overlap != 0.9 else config_spectrograms.get('overlap', 0.985)
        
        # Handle frequency range - command line overrides config
        if freq_range != (5, 100):  # If command line freq_range was explicitly changed
            freq_lims = freq_range
        else:
            # Use config frequency limits, with whale-optimized fallback
            config_freq = config_spectrograms.get('frequency_limits', {})
            freq_lims = (config_freq.get('min', 5), config_freq.get('max', 100))
        
        # Visual parameters - use whale-optimized defaults, config can override
        colormap = config_spectrograms.get('colormap', 'viridis')  # Good for whale calls
        config_clim = config_spectrograms.get('color_limits', {})
        clim = (config_clim.get('min', -40), config_clim.get('max', 0))  # Whale-optimized dynamic range
        log_freq = config_spectrograms.get('log_frequency', False)  # Linear freq for whale calls
        
        # Output format settings
        config_formats = config_spectrograms.get('output_formats', {})
        save_matlab = config_formats.get('matlab', False)
        save_plots = config_formats.get('plots', True)
        
        # Log the parameters being used
        print_status(f"Spectrogram parameters: win_dur={win_dur}s, overlap={overlap}, freq_lims={freq_lims} Hz")
        print_status(f"Visual parameters: colormap={colormap}, clim={clim} dB, log_freq={log_freq}")
        print_status(f"Output formats: matlab={save_matlab}, plots={save_plots}")
        
        # Create separate directories for different output formats
        if save_matlab and save_plots:
            mat_dir = output_dir / "mat_files"
            png_dir = output_dir / "png_files"
        elif save_matlab:
            mat_dir = output_dir / "mat_files"
            png_dir = None
        elif save_plots:
            png_dir = output_dir / "png_files"
            mat_dir = None
        else:
            # Fallback to combined directory
            mat_dir = png_dir = output_dir / "custom_spectrograms"
        
        if mat_dir:
            mat_dir.mkdir(parents=True, exist_ok=True)
        if png_dir:
            png_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize spectrogram generator with config/override values
        self.spectrogram_generator = SpectrogramGenerator(
            win_dur=win_dur,         # FFT window duration from config/args
            overlap=overlap,         # Overlap ratio from config/args  
            freq_lims=freq_lims,     # Frequency limits from config/args
            colormap=colormap,       # Colormap from config
            clim=clim,              # Color limits from config
            log_freq=log_freq,      # Log frequency scale from config
            max_duration=None       # We'll handle timing manually
        )
        
        spectrogram_files = {}
        failed_calls = []
        actual_dimensions = None  # Track actual dimensions from first successful spectrogram
        
        for idx, call in whale_calls.iterrows():
            clip_id = call['Clip ID']
            
            if clip_id not in downloaded_files:
                print_status(f"⚠️  Audio file not available for {clip_id}", "WARNING")
                continue
                
            audio_path = downloaded_files[clip_id]
            
            try:
                # Validate timing data first
                begin_time_raw = call['begin time (s)']
                end_time_raw = call['end time (s)']
                
                # Skip calls with invalid timing data
                if pd.isna(begin_time_raw) or pd.isna(end_time_raw):
                    print_status(f"⚠️ Skipping call with invalid timing data: begin={begin_time_raw}, end={end_time_raw}", "WARNING")
                    failed_calls.append({
                        'clip_id': clip_id,
                        'call_id': f"{clip_id}_invalid_timing",
                        'reason': f'Invalid timing data: begin={begin_time_raw}, end={end_time_raw}'
                    })
                    continue
                
                begin_time = float(begin_time_raw)
                end_time = float(end_time_raw)
                
                # Validate that times are reasonable (allow begin == end)
                if begin_time < 0 or end_time < 0 or begin_time > end_time:
                    print_status(f"⚠️ Skipping call with unreasonable timing: begin={begin_time:.1f}s, end={end_time:.1f}s", "WARNING")
                    failed_calls.append({
                        'clip_id': clip_id,
                        'call_id': f"{clip_id}_invalid_timing",
                        'reason': f'Unreasonable timing: begin={begin_time:.1f}s, end={end_time:.1f}s'
                    })
                    continue
                
                # Create output filename with call timing info
                call_id = f"{clip_id}_{begin_time:.1f}s_{end_time:.1f}s"
                call_id = call_id.replace('.wav', '').replace(':', '-').replace(' ', '_')
                # Determine source tag for title (20Hz/40Hz if known)
                try:
                    src_tag = call.get('source_band') if 'source_band' in call else (self._classify_source_file(call.get('__source_file__', '')) if '__source_file__' in call else 'unknown')
                except Exception:
                    src_tag = 'unknown'
                
                # Set output file paths for different formats (embed source band)
                if png_dir:
                    output_file = png_dir / f"{call_id}_{src_tag}_custom.png"
                else:
                    output_file = None
                
                print_status(f"Generating spectrogram for call: {call_id}", "PROGRESS")
                
                # Load audio and extract call segment
                audio_data, sample_rate = self.spectrogram_generator.load_audio(audio_path)
                call_duration = end_time - begin_time
                
                # ML Context: Use command line override, then config, then default
                if ml_context is not None:
                    context_duration = ml_context
                else:
                    context_duration = self.config.get('temporal_context', {}).get('context_duration', 40.0)
                call_center = (begin_time + end_time) / 2  # Center of the call
                padding = context_duration / 2  # 20s padding on each side
                
                # Calculate desired segment boundaries
                desired_start = call_center - padding
                desired_end = call_center + padding
                
                # Check if we need multi-file stitching
                audio_duration = len(audio_data) / sample_rate
                needs_prev_file = desired_start < 0
                needs_next_file = desired_end > audio_duration
                
                if needs_prev_file or needs_next_file:
                    print_status(f"Need stitching: prev={needs_prev_file}, next={needs_next_file}, start={desired_start:.1f}s, end={desired_end:.1f}s, audio_dur={audio_duration:.1f}s", "INFO")
                    # Derive audio directory from the downloaded audio file path
                    audio_dir = Path(audio_path).parent
                    call_audio = self._stitch_audio_files(call, desired_start, desired_end, context_duration, audio_dir)
                    if call_audio is None:
                        print_status(f"⚠️ Skipping call: unable to stitch adjacent files", "WARNING")
                        failed_calls.append({
                            'clip_id': clip_id,
                            'call_id': call_id,
                            'reason': 'Failed to stitch adjacent files'
                        })
                        continue
                else:
                    # Simple case: extract from single file
                    start_sample = int(desired_start * sample_rate)
                    end_sample = int(desired_end * sample_rate)
                    call_audio = audio_data[start_sample:end_sample]
                
                # Log the actual segment being processed
                actual_duration = len(call_audio) / sample_rate
                print_status(f"ML Context: {actual_duration:.1f}s segment (call: {call_duration:.1f}s, centered with {padding:.1f}s padding each side)", "INFO")
                
                # Generate spectrogram
                frequencies, times, power, power_db_norm = self.spectrogram_generator.compute_spectrogram(
                    call_audio, sample_rate
                )
                
                # Crop to whale call frequency range for ML training
                freq_min, freq_max = 5, 100  # Whale call frequency range
                freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
                frequencies = frequencies[freq_mask]
                power_db_norm = power_db_norm[freq_mask, :]
                
                print_status(f"Cropped to whale frequencies: {frequencies.shape[0]} freq bins ({freq_min}-{freq_max} Hz)", "INFO")
                
                # Capture actual dimensions from first successful spectrogram
                if actual_dimensions is None:
                    actual_dimensions = power_db_norm.shape
                    print_status(f"Captured actual dimensions: {actual_dimensions[0]} x {actual_dimensions[1]} (freq x time)", "INFO")
                
                # Create and save PNG plot if requested
                if save_plots:
                    fig = self.spectrogram_generator.plot_spectrogram(
                        frequencies, times, power_db_norm,
                        title=f"Fin Whale Call ({src_tag}) - {call['device_code']} - {call['Date (UTC)'].strftime('%Y-%m-%d')}\nCall: {begin_time:.1f}s-{end_time:.1f}s ({call_duration:.1f}s) | ML Context: {actual_duration:.1f}s (centered)"
                    )
                    
                    # Save with ML-optimized settings
                    fig.savefig(output_file, dpi=150, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    plt.close(fig)
                    print_status(f"✓ Created PNG: {output_file.name}")
                else:
                    print_status(f"⏭️ Skipped PNG generation (plots=false in config)")
                
                # Log spectrogram dimensions for ML reference
                print_status(f"Spectrogram shape: {power_db_norm.shape} (freq x time) for ML preprocessing", "INFO")
                
                # Save .mat file if requested
                if save_matlab:
                    mat_file = mat_dir / f"{call_id}_{src_tag}_custom.mat"
                    
                    # Prepare data for MATLAB format
                    mat_data = {
                        'spectrogram': power_db_norm,
                        'frequencies': frequencies,
                        'times': times,
                        'call_info': {
                            'device_code': call['device_code'],
                            'date': call['Date (UTC)'].strftime('%Y-%m-%d'),
                            'begin_time_s': float(call['begin time (s)']),
                            'end_time_s': float(call['end time (s)']),
                            'duration_s': float(call['Duration (s)']),
                            'low_freq_hz': float(call['low freq']),
                            'high_freq_hz': float(call['high freq']),
                            'ml_context_duration_s': actual_duration
                        },
                        'processing_params': {
                            'win_dur': win_dur,
                            'overlap': overlap,
                            'freq_lims': freq_lims,
                            'colormap': colormap,
                            'clim': clim,
                            'log_freq': log_freq
                        }
                    }
                    
                    mat_data['call_info']['source_band'] = src_tag
                    scipy.io.savemat(mat_file, mat_data)
                    print_status(f"✓ Saved .mat file: {mat_file.name}")
                
                # Store the output file path (prefer .mat for ML, fallback to PNG)
                if save_matlab:
                    spectrogram_files[call_id] = str(mat_file)
                elif save_plots:
                    spectrogram_files[call_id] = str(output_file)
                else:
                    spectrogram_files[call_id] = f"processed_{call_id}"
                
            except Exception as e:
                print_status(f"❌ Error creating spectrogram for {clip_id}: {e}", "WARNING")
                failed_calls.append({
                    'clip_id': clip_id,
                    'call_id': call_id if 'call_id' in locals() else 'unknown',
                    'reason': f'Processing error: {str(e)}'
                })
                continue
        
        print_status(f"Generated {len(spectrogram_files)} custom spectrograms", "SUCCESS")
        
        # Report failed calls
        if failed_calls:
            print_status(f"⚠️ Failed to create {len(failed_calls)} spectrograms:", "WARNING")
            for failed in failed_calls:
                print_status(f"  - {failed['call_id']}: {failed['reason']}", "WARNING")
        
        return spectrogram_files, failed_calls, actual_dimensions
    
    def download_onc_spectrograms(self,
                                whale_calls: pd.DataFrame,
                                output_dir: Path) -> Dict[str, str]:
        """
        Download corresponding ONC PNG spectrograms for comparison.
        
        Args:
            whale_calls: DataFrame with call data  
            output_dir: Output directory
            
        Returns:
            Dictionary mapping clip IDs to ONC spectrogram paths
        """
        print_header("DOWNLOADING ONC SPECTROGRAMS")
        
        onc_spectrograms_dir = output_dir / "onc_spectrograms"
        onc_spectrograms_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporarily set ONC output path
        original_output_path = self.onc.outPath
        self.onc.outPath = str(onc_spectrograms_dir)
        
        onc_files = {}
        
        try:
            # Process each whale call individually to find relevant spectrograms
            for idx, call in whale_calls.iterrows():
                device = call['device_code']
                call_date = call['Date (UTC)']
                clip_id = call['Clip ID']
                
                print_status(f"Searching for ONC spectrograms for {clip_id}", "PROGRESS")
                
                # Create a 1-hour window around the call time for spectrogram search
                start_time = call_date - timedelta(minutes=30)
                end_time = call_date + timedelta(minutes=30)
                
                try:
                    # Search for PNG spectrogram files in the specific time window
                    filters = {
                        'deviceCode': device,
                        'dateFrom': start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                        'dateTo': end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                        'extension': 'png'
                    }
                    
                    result = self.onc.getListByDevice(filters, allPages=True)
                    
                    if 'files' in result and result['files']:
                        png_files = [f for f in result['files'] if 'spect' in f.lower()]
                        
                        # Limit to 3-5 most relevant spectrograms per call
                        png_files = png_files[:5]
                        
                        for png_file in png_files:
                            try:
                                # Check if already downloaded
                                downloaded_path = onc_spectrograms_dir / png_file
                                if downloaded_path.exists():
                                    onc_files[png_file] = str(downloaded_path)
                                    print_status(f"✓ Already exists: {png_file}")
                                    continue
                                    
                                self.onc.getFile(png_file)
                                if downloaded_path.exists():
                                    onc_files[png_file] = str(downloaded_path)
                                    print_status(f"✓ Downloaded ONC spectrogram: {png_file}")
                            except Exception as e:
                                print_status(f"❌ Error downloading {png_file}: {e}", "WARNING")
                    else:
                        print_status(f"No PNG spectrograms found for {clip_id}", "WARNING")
                        
                except Exception as e:
                    print_status(f"❌ Error searching spectrograms for {clip_id}: {e}", "WARNING")
                        
        finally:
            # Restore original output path
            self.onc.outPath = original_output_path
            
        print_status(f"Downloaded {len(onc_files)} ONC spectrograms", "SUCCESS")
        return onc_files
    
    def cleanup_audio_files(self, downloaded_files: Dict[str, str]) -> int:
        """
        Clean up downloaded audio files to save disk space.
        
        Args:
            downloaded_files: Dictionary mapping clip IDs to file paths
            
        Returns:
            Number of files successfully deleted
        """
        print_header("CLEANING UP AUDIO FILES")
        
        deleted_count = 0
        total_size_mb = 0
        
        for clip_id, file_path in downloaded_files.items():
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    # Get file size before deletion
                    file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
                    total_size_mb += file_size_mb
                    
                    # Delete the file
                    file_path_obj.unlink()
                    deleted_count += 1
                    print_status(f"✓ Deleted: {clip_id} ({file_size_mb:.1f} MB)")
                else:
                    print_status(f"⚠️ File not found: {clip_id}", "WARNING")
            except Exception as e:
                print_status(f"❌ Error deleting {clip_id}: {e}", "ERROR")
        
        print_status(f"🗑️ Cleaned up {deleted_count} audio files, freed {total_size_mb:.1f} MB", "SUCCESS")
        return deleted_count
    
    def _process_single_file_group(self, 
                                 clip_id: str, 
                                 calls_in_file: pd.DataFrame,
                                 output_dir: Path,
                                 audio_dir: Path,
                                 win_dur: float,
                                 overlap: float,
                                 freq_range: Tuple[float, float],
                                 ml_context: Optional[float],
                                 cleanup_audio: bool,
                                 file_num: int,
                                 total_files: int,
                                 generate_positives: bool,
                                 generate_negatives: bool,
                                 generate_overview: bool,
                                 negatives_per_call: int,
                                 neg_context: Optional[float],
                                 calls_by_file: Dict[str, List[Tuple[float, float]]],
                                 neg_margin: float) -> Tuple[Dict[str, str], List[Dict], Optional[Tuple[int, int]], float]:
        """
        Process a single audio file and its associated calls.
        
        Returns:
            Tuple of (spectrogram_files, failed_calls, actual_dimensions, file_size_cleaned_mb)
        """
        thread_id = threading.current_thread().name
        print_status(f"🔄 [{thread_id}] Processing file {file_num}/{total_files}: {clip_id} ({len(calls_in_file)} calls)", "PROGRESS")
        
        spectrogram_files = {}
        failed_calls = []
        actual_dimensions = None
        file_size_cleaned_mb = 0
        
        # Check if file already exists
        audio_file_path = audio_dir / clip_id
        file_already_existed = audio_file_path.exists()
        
        # Download if needed (with thread-safe ONC client handling)
        if not file_already_existed:
            download_success = False
            max_retries = 2
            
            for attempt in range(max_retries + 1):
                try:
                    # Create a temporary ONC client for this thread to avoid conflicts
                    temp_onc = ONC(self.onc.token)
                    temp_onc.outPath = str(audio_dir)
                    
                    if attempt > 0:
                        print_status(f"🔄 [{thread_id}] Retry {attempt}/{max_retries} downloading: {clip_id}", "INFO")
                    
                    result = temp_onc.getFile(clip_id)
                    if not audio_file_path.exists():
                        if attempt == max_retries:
                            print_status(f"❌ [{thread_id}] Failed to download after {max_retries + 1} attempts: {clip_id}", "WARNING")
                        continue
                    
                    # Validate the downloaded file
                    try:
                        import soundfile as sf
                        with sf.SoundFile(audio_file_path) as f:
                            duration = len(f) / f.samplerate
                        print_status(f"✓ [{thread_id}] Downloaded: {clip_id} ({duration:.1f}s, {f.samplerate}Hz)")
                        download_success = True
                        break
                    except Exception as e:
                        print_status(f"❌ [{thread_id}] Downloaded file is corrupted: {clip_id} - {e}", "WARNING")
                        # Delete corrupted file
                        try:
                            audio_file_path.unlink()
                        except:
                            pass
                        if attempt == max_retries:
                            print_status(f"❌ [{thread_id}] File corrupted after {max_retries + 1} attempts: {clip_id}", "WARNING")
                        continue
                        
                except Exception as e:
                    print_status(f"❌ [{thread_id}] Error downloading {clip_id} (attempt {attempt + 1}): {e}", "WARNING")
                    if attempt == max_retries:
                        break
                    continue
            
            if not download_success:
                # Mark all calls in this file as failed
                for _, call in calls_in_file.iterrows():
                    call_id = self._create_safe_call_id(clip_id, call)
                    if call_id is None:
                        call_id = f"{clip_id}_invalid_timing"
                    failed_calls.append({
                        'clip_id': clip_id,
                        'call_id': call_id,
                        'reason': 'Audio file download/validation failed after retries'
                    })
                # Ensure cleanup if a partial file exists
                try:
                    if cleanup_audio and audio_file_path.exists():
                        file_size_mb = audio_file_path.stat().st_size / (1024 * 1024)
                        audio_file_path.unlink()
                        print_status(f"🗑️ [{thread_id}] Cleaned up (failed download): {clip_id} ({file_size_mb:.1f} MB)")
                except Exception as e:
                    print_status(f"⚠️ [{thread_id}] Failed to cleanup after download failure {clip_id}: {e}", "WARNING")
                return spectrogram_files, failed_calls, actual_dimensions, file_size_cleaned_mb
        else:
            # Validate existing file
            try:
                import soundfile as sf
                with sf.SoundFile(audio_file_path) as f:
                    duration = len(f) / f.samplerate
                print_status(f"✓ [{thread_id}] Using existing: {clip_id} ({duration:.1f}s, {f.samplerate}Hz)")
            except Exception as e:
                print_status(f"❌ [{thread_id}] Existing file is corrupted: {clip_id} - {e}", "WARNING")
                # Mark all calls in this file as failed
                for _, call in calls_in_file.iterrows():
                    call_id = self._create_safe_call_id(clip_id, call)
                    if call_id is None:
                        call_id = f"{clip_id}_invalid_timing"
                    failed_calls.append({
                        'clip_id': clip_id,
                        'call_id': call_id,
                        'reason': f'Existing file corrupted: {str(e)}'
                    })
                # Delete corrupted file
                try:
                    if audio_file_path.exists():
                        audio_file_path.unlink()
                except Exception:
                    pass
                return spectrogram_files, failed_calls, actual_dimensions, file_size_cleaned_mb
        
        # Process all calls in this audio file
        try:
            # Generate positives if requested
            if generate_positives:
                file_spectrograms, file_failed, file_dimensions = self.create_custom_spectrograms(
                    calls_in_file, {clip_id: str(audio_file_path)}, output_dir,
                    win_dur=win_dur, overlap=overlap, freq_range=freq_range, ml_context=ml_context
                )
                # Collect results
                spectrogram_files.update(file_spectrograms)
                failed_calls.extend(file_failed)
                if file_dimensions is not None:
                    actual_dimensions = file_dimensions
            # Generate negatives if requested
            if generate_negatives:
                # Compute file duration and sample rate
                try:
                    with sf.SoundFile(audio_file_path) as f:
                        duration = len(f) / f.samplerate
                        sample_rate = f.samplerate
                except Exception as e:
                    print_status(f"❌ [{thread_id}] Unable to read duration for negatives: {e}", "WARNING")
                    duration = 300.0
                device_code = calls_in_file.iloc[0]['device_code'] if not calls_in_file.empty else clip_id.split('_')[0]
                desired_neg = max(0, negatives_per_call * len(calls_in_file))
                if desired_neg > 0:
                    context_duration = neg_context if neg_context is not None else (ml_context if ml_context is not None else 40.0)
                    neg_windows = self._sample_negative_windows_for_file(
                        clip_id, device_code, duration, context_duration, calls_by_file, desired_neg, neg_margin
                    )
                    if neg_windows:
                        # Subset of combined whale_data for this clip for verification summary
                        calls_in_file_df = calls_in_file.copy()
                        neg_specs, neg_failed = self.create_negative_spectrograms(
                            clip_id, device_code, neg_windows, output_dir,
                            win_dur=win_dur, overlap=overlap, freq_range=freq_range, context_duration=context_duration,
                            calls_in_file_df=calls_in_file_df,
                            generate_overview=generate_overview
                        )
                        spectrogram_files.update(neg_specs)
                        failed_calls.extend(neg_failed)
                
        except Exception as e:
            print_status(f"❌ [{thread_id}] Error processing spectrograms for {clip_id}: {e}", "WARNING")
            # Mark all calls as failed
            for _, call in calls_in_file.iterrows():
                call_id = f"{clip_id}_{call['begin time (s)']:.1f}s_{call['end time (s)']:.1f}s".replace('.wav', '').replace(':', '-').replace(' ', '_')
                failed_calls.append({
                    'clip_id': clip_id,
                    'call_id': call_id,
                    'reason': f'Spectrogram processing error: {str(e)}'
                })
        
        finally:
            # Always cleanup audio file if requested, regardless of processing success/failure
            if cleanup_audio and audio_file_path.exists():
                try:
                    file_size_mb = audio_file_path.stat().st_size / (1024 * 1024)
                    audio_file_path.unlink()
                    file_size_cleaned_mb = file_size_mb
                    print_status(f"🗑️ [{thread_id}] Cleaned up: {clip_id} ({file_size_mb:.1f} MB)")
                except Exception as e:
                    print_status(f"⚠️ [{thread_id}] Failed to cleanup {clip_id}: {e}", "WARNING")
        
        return spectrogram_files, failed_calls, actual_dimensions, file_size_cleaned_mb
    
    def process_calls_incrementally(self,
                                  whale_calls: pd.DataFrame,
                                  output_dir: Path,
                                  win_dur: float = 2.0,
                                  overlap: float = 0.5,
                                  freq_range: Tuple[float, float] = (10, 1000),
                                  ml_context: Optional[float] = None,
                                  cleanup_audio: bool = False,
                                  max_workers: int = 2,
                                  generate_positives: bool = True,
                                  generate_negatives: bool = False,
                                  generate_overview: bool = False,
                                  negatives_per_call: int = 0,
                                  neg_context: Optional[float] = None,
                                  neg_margin: float = 0.0) -> Tuple[Dict[str, str], List[Dict], Optional[Tuple[int, int]]]:
        """
        Process whale calls incrementally with parallel processing.
        Downloads, processes, and cleans up audio files in parallel for efficiency.
        
        Returns:
            Tuple of (spectrogram_files, failed_calls, actual_dimensions)
        """
        print_header("PROCESSING CALLS INCREMENTALLY (PARALLEL)")
        
        # Group calls by audio file to process efficiently
        file_groups = list(whale_calls.groupby('Clip ID'))
        total_files = len(file_groups)
        total_calls = len(whale_calls)
        
        print_status(f"📊 Processing {total_calls:,} calls across {total_files:,} audio files")
        print_status(f"⚡ Using {max_workers} parallel workers")
        if cleanup_audio:
            print_status("🧹 Audio cleanup enabled: files will be deleted after processing")
        
        spectrogram_files = {}
        failed_calls = []
        actual_dimensions = None
        total_size_cleaned = 0
        
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Build calls-by-file map for negative sampling and adjacency checks
        calls_by_file: Dict[str, List[Tuple[float, float]]] = {}
        for clip, df in file_groups:
            intervals: List[Tuple[float, float]] = []
            for _, row in df.iterrows():
                try:
                    b = float(row['begin time (s)'])
                    e = float(row['end time (s)'])
                    if b < e:
                        intervals.append((b, e))
                except Exception:
                    continue
            calls_by_file[clip] = intervals
        # Use ThreadPoolExecutor for parallel processing
        # Note: Each audio file is processed by exactly one worker to avoid race conditions
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Worker") as executor:
            # Submit all file processing tasks (one worker per audio file)
            future_to_file = {}
            for i, (clip_id, calls_in_file) in enumerate(file_groups, 1):
                future = executor.submit(
                    self._process_single_file_group,
                    clip_id, calls_in_file, output_dir, audio_dir,
                    win_dur, overlap, freq_range, ml_context, cleanup_audio,
                    i, total_files,
                    generate_positives,
                    generate_negatives, generate_overview, negatives_per_call, neg_context,
                    calls_by_file, neg_margin
                )
                future_to_file[future] = (clip_id, i)
            
            # Collect results as they complete
            completed_files = 0
            for future in concurrent.futures.as_completed(future_to_file):
                clip_id, file_num = future_to_file[future]
                completed_files += 1
                
                try:
                    file_spectrograms, file_failed, file_dimensions, file_size_cleaned_mb = future.result()
                    
                    # Collect results
                    spectrogram_files.update(file_spectrograms)
                    failed_calls.extend(file_failed)
                    if actual_dimensions is None and file_dimensions is not None:
                        actual_dimensions = file_dimensions
                    total_size_cleaned += file_size_cleaned_mb
                    
                    # Progress update
                    calls_completed = sum(len(sg) for sg in file_spectrograms.values() if sg)
                    print_status(f"📈 Progress: {completed_files}/{total_files} files completed ({completed_files/total_files*100:.1f}%)")
                    
                except Exception as e:
                    print_status(f"❌ Error processing file group for {clip_id}: {e}", "ERROR")
                    # The individual file handler should have already marked calls as failed
        
        # Summary
        print_status(f"✅ Processed {len(spectrogram_files)} spectrograms successfully")
        if failed_calls:
            print_status(f"⚠️ Failed to process {len(failed_calls)} calls", "WARNING")
        if cleanup_audio and total_size_cleaned > 0:
            print_status(f"🧹 Total space cleaned: {total_size_cleaned:.1f} MB", "SUCCESS")
        
        return spectrogram_files, failed_calls, actual_dimensions
    
    def create_analysis_report(self,
                             whale_calls: pd.DataFrame,
                             downloaded_files: Dict[str, str],
                             custom_spectrograms: Dict[str, str],
                             onc_spectrograms: Dict[str, str],
                             output_dir: Path,
                             failed_calls: List[Dict] = None,
                             actual_dimensions: Optional[Tuple[int, int]] = None,
                             audio_cleaned_up: bool = False):
        """Create a comprehensive analysis report"""
        print_header("CREATING ANALYSIS REPORT")
        
        # Separate positive/negative counts
        total_specs = len(custom_spectrograms)
        negative_count = sum(1 for k in custom_spectrograms.keys() if "_neg_" in k)
        positive_count = total_specs - negative_count

        report = {
            "dataset_metadata": {
                "creation_date": datetime.now().isoformat(),
                "source_libraries": list(self.excel_files),
                "total_calls_analyzed": len(whale_calls),
                "successful_spectrograms": total_specs,
                "positive_spectrograms": positive_count,
                "negative_spectrograms": negative_count,
                "failed_spectrograms": len(failed_calls) if failed_calls else 0,
                "unique_audio_files": len(downloaded_files),
                "onc_spectrograms_downloaded": len(onc_spectrograms)
            },
            "processing_parameters": {
                "spectrogram_generation": {
                    "window_duration_s": self.spectrogram_generator.win_dur,
                    "overlap_ratio": self.spectrogram_generator.overlap,
                    "frequency_limits_hz": {
                        "min": self.spectrogram_generator.freq_lims[0],
                        "max": self.spectrogram_generator.freq_lims[1]
                    },
                    "colormap": self.spectrogram_generator.colormap,
                    "color_limits_db": {
                        "min": self.spectrogram_generator.clim[0],
                        "max": self.spectrogram_generator.clim[1]
                    },
                    "log_frequency_scale": self.spectrogram_generator.log_freq,
                    "fft_method": "scipy.signal.spectrogram with Hann window",
                    "scaling": "power spectral density (PSD)",
                    "normalization": "10*log10(abs(P/max(P)))"
                },
                "temporal_context": {
                    "context_duration_s": self.config.get('temporal_context', {}).get('context_duration', 40.0),
                    "padding_method": self.config.get('temporal_context', {}).get('padding_method', 'centered'),
                    "multi_file_stitching": self.config.get('temporal_context', {}).get('multi_file_stitching', True),
                    "exact_duration_enforcement": self.config.get('temporal_context', {}).get('exact_duration_enforcement', True)
                },
                "frequency_filtering": {
                    "whale_call_range_hz": [5, 100],
                    "post_processing_crop": "applied after spectrogram generation",
                    "actual_freq_bins": f"{actual_dimensions[0]} bins" if actual_dimensions else "varies per spectrogram",
                    "actual_freq_resolution_hz": f"~{95/actual_dimensions[0]:.2f} Hz per bin" if actual_dimensions else "varies per spectrogram"
                },
                "spectrogram_dimensions": {
                    "actual_dimensions": f"{actual_dimensions[0]} x {actual_dimensions[1]} (freq x time)" if actual_dimensions else "varies per spectrogram",
                    "actual_time_resolution_ms": f"~{40000/actual_dimensions[1]:.1f} ms per bin" if actual_dimensions else "varies per spectrogram",
                    "frequency_range_hz": [5, 100],
                    "temporal_context_s": 40.0,
                    "augmentation_ready": "centered context allows sliding window cropping"
                }
            },
            "technical_specifications": {
                "audio_format": "WAV files from Ocean Networks Canada",
                "sample_rate_hz": "varies by file (typically 64kHz)",
                "bit_depth": "varies by file",
                "file_duration_s": 300,
                "device_codes": list(set(whale_calls['device_code'].tolist())),
                "date_range": {
                    "start": whale_calls['Date (UTC)'].min().isoformat(),
                    "end": whale_calls['Date (UTC)'].max().isoformat()
                }
            },
            "output_locations": {
                "audio_directory": "whale_call_analysis/audio/" if not audio_cleaned_up else "whale_call_analysis/audio/ (cleaned up)",
                "mat_files_directory": "whale_call_analysis/mat_files/" if self.config.get('custom_spectrograms', {}).get('output_formats', {}).get('matlab', False) else None,
                "png_files_directory": "whale_call_analysis/png_files/" if self.config.get('custom_spectrograms', {}).get('output_formats', {}).get('plots', True) else None,
                "neg_mat_files_directory": "whale_call_analysis/neg_mat_files/" if self.config.get('custom_spectrograms', {}).get('output_formats', {}).get('matlab', False) else None,
                "neg_png_files_directory": "whale_call_analysis/neg_png_files/" if self.config.get('custom_spectrograms', {}).get('output_formats', {}).get('plots', True) else None,
                "onc_spectrograms_directory": "whale_call_analysis/onc_spectrograms/",
                "note": "Directories created based on output format settings",
                "audio_files_cleaned_up": audio_cleaned_up
            },
            "reproduction_instructions": {
                "required_libraries": [
                    "pandas", "numpy", "matplotlib", "scipy", "soundfile", "onc"
                ],
                "key_parameters": {
                    "win_dur": self.spectrogram_generator.win_dur,
                    "overlap": self.spectrogram_generator.overlap,
                    "freq_crop": [5, 100],
                    "context_duration": self.config.get('temporal_context', {}).get('context_duration', 40.0),
                    "normalization": "10*log10(abs(spectrogram/max(spectrogram)))"
                },
                "notes": [
                    "Spectrograms are cropped to whale call frequencies (5-100 Hz) after generation",
                    "40-second temporal context is centered on each call",
                    "Multi-file stitching is used when context extends beyond file boundaries",
                    "Failed stitching cases are documented in separate failed_spectrograms.json",
                    "Negative (no-call) windows are sampled from free intervals avoiding all annotated calls"
                ]
            }
        }
        
        # No detailed call listings needed in main documentation
        
        # Save main dataset report
        dataset_report_file = output_dir / "dataset_documentation.json"
        with open(dataset_report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print_status(f"Dataset documentation saved: {dataset_report_file}", "SUCCESS")
        
        # Save failed calls report (only if there are failures)
        if failed_calls:
            failed_report = {
                "failed_spectrograms": {
                    "total_failed": len(failed_calls),
                    "analysis_date": datetime.now().isoformat(),
                    "failures": []
                }
            }
            
            for failed in failed_calls:
                failure_entry = {
                    "call_id": failed["call_id"],
                    "clip_id": failed["clip_id"],
                    "failure_reason": failed["reason"]
                }
                failed_report["failed_spectrograms"]["failures"].append(failure_entry)
            
            failed_file = output_dir / "failed_spectrograms.json"
            with open(failed_file, 'w') as f:
                json.dump(failed_report, f, indent=2, default=str)
                
            print_status(f"Failed spectrograms report saved: {failed_file}", "SUCCESS")
        
        # Print summary
        print_status(f"📊 Analyzed {len(whale_calls)} fin whale calls")
        print_status(f"🎵 Downloaded {len(downloaded_files)} audio files")
        print_status(f"📈 Created {len(custom_spectrograms)} custom spectrograms")
        print_status(f"🌊 Downloaded {len(onc_spectrograms)} ONC spectrograms")
        print_status(f"📁 Results saved to: {output_dir}")

def main():
    """Main function with argument parsing and orchestration"""
    parser = argparse.ArgumentParser(
        description="Fin Whale Call Analysis and Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Sampling parameters
    parser.add_argument('--sample-size', type=int, default=20,
                       help='Number of whale calls to analyze (default: 20). Use --process-all to process entire dataset')
    parser.add_argument('--process-all', action='store_true',
                       help='Process the entire unfiltered dataset (ignores all filtering arguments and --sample-size)')
    parser.add_argument('--device', type=str,
                       help='Filter by specific device code (e.g., ICLISTENHF1353)')
    parser.add_argument('--start-date', type=str,
                       help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--min-duration', type=float, default=1.0,
                       help='Minimum call duration in seconds (default: 1.0)')
    parser.add_argument('--max-duration', type=float, default=30.0,
                       help='Maximum call duration in seconds (default: 30.0)')
    parser.add_argument('--freq-range', type=float, nargs=2, default=[5, 100],
                       help='Frequency range for analysis [min max] in Hz (default: from config, fallback: 5 100)')
    
    # Spectrogram parameters
    parser.add_argument('--win-dur', type=float, default=0.1,
                       help='Window duration for spectrograms in seconds (default: from config, fallback: 0.1)')
    parser.add_argument('--overlap', type=float, default=0.9,
                       help='Overlap ratio for spectrograms (default: from config, fallback: 0.9)')
    
    # Input/Output options
    group_excel = parser.add_mutually_exclusive_group(required=True)
    group_excel.add_argument('--excel-file', type=str,
                       help='Path to a single Excel file containing whale call library (e.g., data/finwhales/FinWhale20Hz_CallLibrary_Rannankari.xlsx)')
    group_excel.add_argument('--excel-files', type=str, nargs='+',
                       help='Paths to multiple Excel files (e.g., 20 Hz and 40 Hz annotations)')
    parser.add_argument('--output-dir', type=str, default='whale_call_analysis',
                       help='Output directory for results (default: whale_call_analysis)')
    parser.add_argument('--config', type=str, default='./config/dataset_config.yaml',
                       help='Path to configuration file (default: ./config/dataset_config.yaml)')
    parser.add_argument('--mat-only', action='store_true',
                       help='Generate only .mat files for ML (no PNG plots)')
    parser.add_argument('--png-only', action='store_true',
                       help='Generate only PNG plots (no .mat files)')
    
    # ML-specific options
    parser.add_argument('--ml-context', type=float, default=40.0,
                       help='Minimum time context for ML augmentation in seconds (default: from config, fallback: 40.0)')
    parser.add_argument('--target-size', type=str, default='512x512',
                       help='Target spectrogram size for ML (default: 512x512)')
    # Negative (no-call) dataset options
    parser.add_argument('--generate-negatives', action='store_true',
                       help='Also generate negative spectrograms with no fin whale calls')
    parser.add_argument('--negatives-only', action='store_true',
                       help='Generate only negative spectrograms (skip positive/call spectrograms)')
    parser.add_argument('--negatives-per-call', type=int, default=1,
                       help='Number of negative windows to sample per annotated call (default: 1)')
    parser.add_argument('--neg-context', type=float,
                       help='Context duration for negative windows in seconds (default: use --ml-context)')
    parser.add_argument('--neg-margin', type=float, default=2.0,
                       help='Seconds to keep as safety margin around annotated calls when sampling negatives (default: 2.0)')
    parser.add_argument('--overview', action='store_true',
                       help='Generate per-file overview spectrograms with call and negative overlays')
    
    # Processing options
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip audio file download (use existing files)')
    parser.add_argument('--skip-onc-spectrograms', action='store_true',
                       help='Skip ONC spectrogram download')
    parser.add_argument('--cleanup-audio', action='store_true',
                       help='Delete WAV files after processing to save disk space')
    parser.add_argument('--batch-mode', action='store_true',
                       help='Use old batch processing method (download all, then process all)')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of parallel workers for processing (default: 2, increase if you have good bandwidth/CPU)')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    onc_token = os.getenv('ONC_TOKEN')
    
    if not onc_token:
        raise ValueError("ONC_TOKEN not found in environment variables. Please set it in .env file.")
    
    try:
        print_header("FIN WHALE CALL ANALYSIS TOOL")
        
        # Validate Excel file(s) exist
        if args.excel_files and len(args.excel_files) > 0:
            missing = [p for p in args.excel_files if not Path(p).exists()]
            if missing:
                raise FileNotFoundError(f"Excel files not found: {missing}")
        else:
            if not Path(args.excel_file).exists():
                raise FileNotFoundError(f"Excel file not found: {args.excel_file}")
        
        # Initialize analyzer with one or more Excel file paths
        analyzer = FinWhaleCallAnalyzer(onc_token, args.excel_file, args.config, excel_files=args.excel_files)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle --process-all option
        if args.process_all:
            print_status("🚀 Processing entire dataset (--process-all enabled, ignoring all filters)", "INFO")
            whale_calls = analyzer.sample_calls(
                sample_size=None,
                device_filter=None,
                start_date=None,
                end_date=None,
                min_duration=0.0,
                max_duration=float('inf'),
                freq_range=None
            )
        else:
            # Sample whale calls with specified filters
            whale_calls = analyzer.sample_calls(
                sample_size=args.sample_size,
                device_filter=args.device,
                start_date=args.start_date,
                end_date=args.end_date,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                freq_range=tuple(args.freq_range)
            )
        
        # Save sampled calls
        calls_file = output_dir / "sampled_calls.csv"
        whale_calls.to_csv(calls_file, index=False)
        print_status(f"Sampled calls saved: {calls_file}")
        
        # Override output formats if specified
        original_formats = None
        if args.mat_only:
            print_status("🔬 MAT-only mode: Generating .mat files for ML", "INFO")
            # Temporarily override config
            original_formats = analyzer.config.get('custom_spectrograms', {}).get('output_formats', {})
            analyzer.config.setdefault('custom_spectrograms', {})['output_formats'] = {'matlab': True, 'plots': False}
        elif args.png_only:
            print_status("🖼️ PNG-only mode: Generating visualization plots", "INFO")
            original_formats = analyzer.config.get('custom_spectrograms', {}).get('output_formats', {})
            analyzer.config.setdefault('custom_spectrograms', {})['output_formats'] = {'matlab': False, 'plots': True}
        
        # Choose processing method
        custom_spectrograms = {}
        failed_calls = []
        actual_dimensions = None
        downloaded_files = {}
        
        if not args.skip_download:
            if args.batch_mode and not (args.generate_negatives or args.negatives_only):
                # Old batch method: download all, then process all
                print_status("📦 Using batch processing mode", "INFO")
                downloaded_files = analyzer.download_whale_call_audio(whale_calls, output_dir)
                if downloaded_files:
                    custom_spectrograms, failed_calls, actual_dimensions = analyzer.create_custom_spectrograms(
                        whale_calls, downloaded_files, output_dir,
                        win_dur=args.win_dur,
                        overlap=args.overlap,
                        freq_range=tuple(args.freq_range),
                        ml_context=args.ml_context if args.ml_context != 40.0 else None
                    )
                    # Clean up after batch processing if requested
                    if args.cleanup_audio:
                        deleted_count = analyzer.cleanup_audio_files(downloaded_files)
                        print_status(f"🧹 Audio cleanup: {deleted_count} files deleted", "INFO")
            else:
                # New incremental method: download -> process -> cleanup as we go
                print_status("🔄 Using incremental processing mode (memory efficient)", "INFO")
                custom_spectrograms, failed_calls, actual_dimensions = analyzer.process_calls_incrementally(
                    whale_calls, output_dir,
                    win_dur=args.win_dur,
                    overlap=args.overlap,
                    freq_range=tuple(args.freq_range),
                    ml_context=args.ml_context if args.ml_context != 40.0 else None,
                    cleanup_audio=args.cleanup_audio,
                    max_workers=args.workers,
                    generate_positives=(not args.negatives_only),
                    generate_negatives=(args.generate_negatives or args.negatives_only),
                    generate_overview=args.overview,
                    negatives_per_call=args.negatives_per_call,
                    neg_context=args.neg_context,
                    neg_margin=args.neg_margin
                )
                # For report compatibility, simulate downloaded_files
                unique_clips = whale_calls['Clip ID'].unique()
                downloaded_files = {clip_id: f"processed_{clip_id}" for clip_id in unique_clips}
        
        # Restore original formats if overridden
        if original_formats is not None:
            analyzer.config.setdefault('custom_spectrograms', {})['output_formats'] = original_formats
        
        # Download ONC spectrograms for comparison
        onc_spectrograms = {}
        if not args.skip_onc_spectrograms and not args.negatives_only:
            onc_spectrograms = analyzer.download_onc_spectrograms(whale_calls, output_dir)
        
        # Create analysis report
        analyzer.create_analysis_report(
            whale_calls, downloaded_files, custom_spectrograms, onc_spectrograms, output_dir, failed_calls, actual_dimensions, args.cleanup_audio
        )
        
        print_header("ANALYSIS COMPLETE")
        print_status("Fin whale call analysis completed successfully!", "SUCCESS")
        
    except Exception as e:
        print_status(f"Error: {e}", "ERROR")
        raise

if __name__ == "__main__":
    main() 