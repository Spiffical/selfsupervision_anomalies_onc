#!/usr/bin/env python3
"""
Script to download spectrograms from Ocean Networks Canada (ONC) using the SpectrogramDownloader class.

This script provides multiple download modes:
1. Sampling schedule mode: Downloads spectrograms based on a sampling schedule
2. Specific times mode: Downloads spectrograms for specific timestamps
3. Date range mode: Downloads all available spectrograms in a date range

Usage:
    python download_spectrograms.py --mode sampling --device ICLISTENHF6020 --start-date 2020 10 2 --threshold 1000
    python download_spectrograms.py --mode specific --config specific_times.json
    python download_spectrograms.py --mode range --device ICLISTENHF6020 --start-date 2020 10 2 --end-date 2020 10 5
"""

import os
import sys
import argparse
import json
import warnings
import glob
from datetime import datetime, date
from dotenv import load_dotenv

# Add the utils directory to the path so we can import the SpectrogramDownloader
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data.spectrogram_downloader import SpectrogramDownloader


class VerboseSpectrogramDownloader:
    """Wrapper around SpectrogramDownloader with better messaging"""
    
    def __init__(self, onc_token, parent_dir, verbose=False):
        self.downloader = SpectrogramDownloader(onc_token, parent_dir)
        self.verbose = verbose
        
    def download_spectrograms_with_sampling_schedule(self, deviceCode, start_date, threshold_num, num_days=None, filetype='png'):
        """Download with better progress tracking"""
        print_status("Setting up directories...", "PROGRESS")
        self.downloader.setup_directories(filetype)
        
        print_status("Calculating sampling schedule...", "PROGRESS")
        year, month, day = start_date
        
        # Suppress warnings unless verbose mode
        if not self.verbose:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                date_object_list, sample_time_per_day = self.downloader.sampling_schedule(
                    deviceCode, threshold_num, year, month, day, num_days=num_days
                )
        else:
            date_object_list, sample_time_per_day = self.downloader.sampling_schedule(
                deviceCode, threshold_num, year, month, day, num_days=num_days
            )
        
        print_status(f"Found {len(date_object_list)} time slots to download", "SUCCESS")
        
        if len(date_object_list) == 0:
            print_status("No new files to download (all already exist)", "INFO")
            return
        
        print_section("Downloading Files")
        
        for i, start_date_object in enumerate(date_object_list, 1):
            # Check current progress
            num_files_downloaded = len(glob.glob(os.path.join(self.downloader.processed_path, f'*.{filetype}')))
            
            print_status(f"Progress: {num_files_downloaded}/{threshold_num} files downloaded", "PROGRESS")
            
            if num_files_downloaded >= threshold_num:
                print_status("Target number of files reached!", "SUCCESS")
                break
            
            print_status(f"Downloading batch {i}/{len(date_object_list)}: {start_date_object.strftime('%Y-%m-%d %H:%M:%S')}", "PROGRESS")
            
            # Download files
            if not self.verbose:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.downloader.download_MAT_or_PNG(deviceCode, start_date_object, filetype=filetype, data_length_seconds=sample_time_per_day)
            else:
                self.downloader.download_MAT_or_PNG(deviceCode, start_date_object, filetype=filetype, data_length_seconds=sample_time_per_day)
            
            # Process files
            print_status("Processing downloaded files...", "PROGRESS")
            self.downloader.process_spectrograms(filetype)
            
            # Show updated progress
            num_files_after = len(glob.glob(os.path.join(self.downloader.processed_path, f'*.{filetype}')))
            new_files = num_files_after - num_files_downloaded
            if new_files > 0:
                print_status(f"Added {new_files} new files", "SUCCESS")
            else:
                print_status("No new files added (may be duplicates or anomalies)", "WARNING")
    
    def download_specific_spectrograms(self, device_times_dict, filetype='png'):
        """Download specific spectrograms with progress tracking"""
        total_downloads = sum(len(times) for times in device_times_dict.values())
        current_download = 0
        
        for device_id, times in device_times_dict.items():
            print_status(f"Processing device: {device_id}", "PROGRESS")
            
            for time_tuple in times:
                current_download += 1
                year, month, day, hour, minute, second = time_tuple
                start_date_object = datetime(year, month, day, hour, minute, second)
                
                print_status(f"Download {current_download}/{total_downloads}: {start_date_object.strftime('%Y-%m-%d %H:%M:%S')}", "PROGRESS")
                
                # Setup directories
                self.downloader.setup_directories(filetype)
                
                # Download specific spectrogram
                if not self.verbose:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.downloader.download_MAT_or_PNG(device_id, start_date_object, filetype=filetype, data_length_seconds=300)
                else:
                    self.downloader.download_MAT_or_PNG(device_id, start_date_object, filetype=filetype, data_length_seconds=300)
                
                # Process the spectrograms
                self.downloader.process_spectrograms(filetype)


def load_config():
    """Load configuration from .env file"""
    load_dotenv()
    
    onc_token = os.getenv('ONC_TOKEN')
    data_dir = os.getenv('DATA_DIR', './data')
    
    if not onc_token or onc_token == 'your_onc_api_token_here':
        raise ValueError("Please set your ONC_TOKEN in the .env file")
    
    return onc_token, data_dir


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_section(title):
    """Print a formatted section header"""
    print(f"\n--- {title} ---")


def print_status(message, level="INFO"):
    """Print a status message with level indicator"""
    prefix = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅ ",
        "WARNING": "⚠️ ",
        "ERROR": "❌ ",
        "PROGRESS": "🔄 "
    }.get(level, "")
    print(f"{prefix}{message}")


def download_with_sampling_schedule(args, downloader):
    """Download spectrograms using sampling schedule mode"""
    print_header("SAMPLING SCHEDULE MODE")
    
    print_status(f"Device Code: {args.device}")
    print_status(f"Start Date: {'-'.join(map(str, args.start_date))}")
    print_status(f"Target Files: {args.threshold}")
    print_status(f"File Type: {args.filetype.upper()}")
    
    if args.num_days:
        print_status(f"Days to Consider: {args.num_days}")
    
    print_section("Starting Download Process")
    
    try:
        downloader.download_spectrograms_with_sampling_schedule(
            deviceCode=args.device,
            start_date=args.start_date,
            threshold_num=args.threshold,
            num_days=args.num_days,
            filetype=args.filetype
        )
    except Exception as e:
        if "restricted" in str(e).lower():
            print_status("Some data may be restricted. Check ONC permissions if downloads fail.", "WARNING")
        raise


def download_specific_times(args, downloader):
    """Download spectrograms for specific timestamps"""
    print_header("SPECIFIC TIMES MODE")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        device_times_dict = json.load(f)
    
    print_status(f"Configuration File: {args.config}")
    print_status(f"Devices Found: {', '.join(device_times_dict.keys())}")
    print_status(f"File Type: {args.filetype.upper()}")
    
    total_downloads = sum(len(times) for times in device_times_dict.values())
    print_status(f"Total Downloads Planned: {total_downloads}")
    
    print_section("Starting Download Process")
    
    try:
        downloader.download_specific_spectrograms(
            device_times_dict=device_times_dict,
            filetype=args.filetype
        )
    except Exception as e:
        if "restricted" in str(e).lower():
            print_status("Some data may be restricted. Check ONC permissions if downloads fail.", "WARNING")
        raise


def download_date_range(args, downloader):
    """Download spectrograms for a date range (simplified approach)"""
    print_header("DATE RANGE MODE")
    
    print_status(f"Device Code: {args.device}")
    print_status(f"Start Date: {'-'.join(map(str, args.start_date))}")
    print_status(f"End Date: {'-'.join(map(str, args.end_date))}")
    print_status(f"File Type: {args.filetype.upper()}")
    
    # Calculate number of days between start and end date
    start_date_obj = date(*args.start_date)
    end_date_obj = date(*args.end_date)
    num_days = (end_date_obj - start_date_obj).days + 1
    
    print_status(f"Date Range: {num_days} days")
    
    # Use a high threshold to get all available data
    threshold = 10000
    print_status(f"Maximum Files: {threshold} (will download all available)")
    
    print_section("Starting Download Process")
    
    try:
        downloader.download_spectrograms_with_sampling_schedule(
            deviceCode=args.device,
            start_date=args.start_date,
            threshold_num=threshold,
            num_days=num_days,
            filetype=args.filetype
        )
    except Exception as e:
        if "restricted" in str(e).lower():
            print_status("Some data may be restricted. Check ONC permissions if downloads fail.", "WARNING")
        raise


def create_example_config():
    """Create an example configuration file for specific times mode"""
    print_header("CREATING EXAMPLE CONFIGURATION")
    
    example_config = {
        "ICLISTENHF6020": [
            [2020, 10, 2, 12, 0, 0],  # Year, Month, Day, Hour, Minute, Second
            [2020, 10, 2, 18, 30, 0],
            [2020, 10, 3, 6, 15, 0]
        ],
        "ANOTHER_DEVICE": [
            [2020, 10, 5, 9, 0, 0],
            [2020, 10, 5, 15, 45, 0]
        ]
    }
    
    config_file = "example_specific_times.json"
    with open(config_file, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print_status(f"Example configuration created: {config_file}", "SUCCESS")
    print_status("Edit this file with your desired device codes and timestamps.")
    print_status("Each timestamp format: [Year, Month, Day, Hour, Minute, Second]")


def main():
    parser = argparse.ArgumentParser(
        description="Download spectrograms from Ocean Networks Canada (ONC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 1000 spectrograms using sampling schedule
  python download_spectrograms.py --mode sampling --device ICLISTENHF6020 --start-date 2020 10 2 --threshold 1000

  # Download spectrograms for specific times (requires config file)
  python download_spectrograms.py --mode specific --config specific_times.json

  # Download all spectrograms in a date range
  python download_spectrograms.py --mode range --device ICLISTENHF6020 --start-date 2020 10 2 --end-date 2020 10 5

  # Create example configuration file
  python download_spectrograms.py --create-example-config
        """
    )
    
    parser.add_argument('--mode', choices=['sampling', 'specific', 'range'],
                       help='Download mode: sampling (with schedule), specific (timestamps), or range (date range)')
    
    parser.add_argument('--device', type=str,
                       help='Device code (e.g., ICLISTENHF6020)')
    
    parser.add_argument('--start-date', nargs=3, type=int, metavar=('YEAR', 'MONTH', 'DAY'),
                       help='Start date as three integers: YEAR MONTH DAY')
    
    parser.add_argument('--end-date', nargs=3, type=int, metavar=('YEAR', 'MONTH', 'DAY'),
                       help='End date as three integers: YEAR MONTH DAY (for range mode)')
    
    parser.add_argument('--threshold', type=int, default=1000,
                       help='Number of spectrograms to download (for sampling mode)')
    
    parser.add_argument('--num-days', type=int,
                       help='Number of days to consider (optional, for sampling mode)')
    
    parser.add_argument('--filetype', choices=['png', 'mat'], default='mat',
                       help='File type to download: png or mat')
    
    parser.add_argument('--config', type=str,
                       help='JSON configuration file for specific times mode')
    
    parser.add_argument('--create-example-config', action='store_true',
                       help='Create an example configuration file and exit')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed ONC API messages (including warnings)')
    
    args = parser.parse_args()
    
    # Handle example config creation
    if args.create_example_config:
        create_example_config()
        return
    
    # Validate arguments
    if not args.mode:
        parser.error("Mode is required. Use --mode sampling, --mode specific, or --mode range")
    
    if args.mode in ['sampling', 'range'] and not args.device:
        parser.error(f"Device code is required for {args.mode} mode")
    
    if args.mode in ['sampling', 'range'] and not args.start_date:
        parser.error(f"Start date is required for {args.mode} mode")
    
    if args.mode == 'range' and not args.end_date:
        parser.error("End date is required for range mode")
    
    if args.mode == 'specific' and not args.config:
        parser.error("Configuration file is required for specific mode")
    
    try:
        print_header("ONC SPECTROGRAM DOWNLOADER")
        
        # Load configuration
        print_section("Loading Configuration")
        onc_token, data_dir = load_config()
        print_status(f"Data Directory: {data_dir}")
        print_status("ONC Token: ✓ Loaded", "SUCCESS")
        
        if not args.verbose:
            print_status("Verbose mode OFF - ONC warnings suppressed for cleaner output")
            print_status("Use --verbose flag to see detailed API messages")
        
        # Initialize downloader
        print_section("Initializing Downloader")
        downloader = VerboseSpectrogramDownloader(onc_token, data_dir, verbose=args.verbose)
        print_status("SpectrogramDownloader initialized", "SUCCESS")
        
        # Execute based on mode
        if args.mode == 'sampling':
            download_with_sampling_schedule(args, downloader)
        elif args.mode == 'specific':
            download_specific_times(args, downloader)
        elif args.mode == 'range':
            download_date_range(args, downloader)
        
        print_section("Download Complete")
        print_status("All downloads completed successfully!", "SUCCESS")
        print_status(f"Check your data directory: {data_dir}")
        print_status(f"Processed files: {data_dir}/{args.filetype}/processed/")
        print_status(f"Rejected files: {data_dir}/{args.filetype}/rejects/")
        
    except ValueError as e:
        print_status(f"Configuration Error: {e}", "ERROR")
        sys.exit(1)
    except FileNotFoundError as e:
        print_status(f"File Error: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        print_status(f"Unexpected Error: {e}", "ERROR")
        if "restricted" in str(e).lower():
            print_status("This may be due to data access restrictions.", "WARNING")
            print_status("Contact datastewardship@oceannetworks.ca for access permissions.", "INFO")
        sys.exit(1)


if __name__ == "__main__":
    main() 