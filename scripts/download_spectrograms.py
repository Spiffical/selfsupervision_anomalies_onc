#!/usr/bin/env python3
"""
Script to download spectrograms from Ocean Networks Canada (ONC) using the SpectrogramDownloader class.

This script provides multiple download modes with automatic interactive prompting for missing parameters:

1. Sampling schedule mode: Downloads spectrograms based on a sampling schedule within a date range
2. Specific times mode: Downloads spectrograms for specific timestamps (requires config file)
3. Date range mode: Downloads all available spectrograms in a date range
4. Check deployments mode: Show available deployments for planning downloads

The script automatically becomes interactive when required parameters are missing, making it easy to use
for both beginners (who can be guided through the process) and power users (who can specify all parameters).

Key Features:
- Automatic deployment checking and validation
- Interactive device selection from available hydrophones
- Deployment-aware date selection with validation
- Smart parameter collection for missing arguments
- Timezone-aware datetime handling
- Efficient API call caching to avoid redundant requests

Usage Examples:

  # Fully specified (non-interactive)
  python download_spectrograms.py --mode sampling --device ICLISTENHF6020 --start-date 2020 10 2 --threshold 1000
  
  # Interactive - will prompt for missing parameters
  python download_spectrograms.py --mode sampling  # Will ask for device, dates, threshold
  python download_spectrograms.py  # Will ask for device, dates, threshold (defaults to sampling mode)
  
  # With deployment checking enabled
  python download_spectrograms.py --mode sampling --device ICLISTENHF6020 --start-date 2020 10 2 --threshold 1000 --check-deployments
  
  # Date range downloads
  python download_spectrograms.py --mode range --device ICLISTENHF6020 --start-date 2020 10 2 --end-date 2020 10 5
  
  # Check available deployments
  python download_spectrograms.py --mode check-deployments --device ICLISTENHF6020 --start-date 2020 10 2 --end-date 2020 10 5
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
        
    def download_spectrograms_with_sampling_schedule(self, deviceCode, start_date, threshold_num, num_days=None, filetype='png', check_deployments=False, auto_select_deployment=False, duration_seconds=None):
        """Download spectrograms using sampling schedule with optional deployment checking"""
        
        if check_deployments:
            print_status("Using deployment-aware download mode", "INFO")
            return self._download_with_deployment_check(deviceCode, start_date, threshold_num, num_days, filetype, auto_select_deployment, duration_seconds)
        
        print_status("Setting up directories...", "PROGRESS")
        
        # Calculate end date for directory naming
        year, month, day = start_date
        if num_days:
            from datetime import date, timedelta
            start_date_obj = date(year, month, day)
            end_date_obj = start_date_obj + timedelta(days=num_days)
            end_date_tuple = (end_date_obj.year, end_date_obj.month, end_date_obj.day)
        else:
            end_date_tuple = None
        
        # Use provided duration or default
        actual_duration = duration_seconds if duration_seconds else 1799
        
        self.downloader.setup_directories(filetype, deviceCode, 'sampling', start_date, end_date_tuple, actual_duration)
        
        print_status("Calculating sampling schedule...", "PROGRESS")
        year, month, day = start_date
        
        # Use custom duration if provided, otherwise use calculated sampling duration
        if duration_seconds:
            sample_time_per_day = duration_seconds
        
        # Suppress warnings unless verbose mode
        if not self.verbose:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if duration_seconds:
                    # Custom duration: create simple date list
                    date_object_list = self.downloader.sampling_schedule(
                        deviceCode, threshold_num, year, month, day, num_days=num_days
                    )[0]  # Only get date list, ignore calculated sample_time
                else:
                    # Default behavior: calculate optimal sampling
                    date_object_list, sample_time_per_day = self.downloader.sampling_schedule(
                        deviceCode, threshold_num, year, month, day, num_days=num_days
                    )
        else:
            if duration_seconds:
                # Custom duration: create simple date list
                date_object_list = self.downloader.sampling_schedule(
                    deviceCode, threshold_num, year, month, day, num_days=num_days
                )[0]  # Only get date list, ignore calculated sample_time
            else:
                # Default behavior: calculate optimal sampling
                date_object_list, sample_time_per_day = self.downloader.sampling_schedule(
                    deviceCode, threshold_num, year, month, day, num_days=num_days
                )
        
        print_status(f"Found {len(date_object_list)} time slots to download", "SUCCESS")
        
        if len(date_object_list) == 0:
            print_status("No new files to download (all already exist)", "INFO")
            return
        
        print_section("Downloading Files")
        
        for i, start_date_object in enumerate(date_object_list, 1):
            # Check current progress for this specific device
            num_files_downloaded = len(glob.glob(os.path.join(self.downloader.processed_path, f'{deviceCode}_*.{filetype}')))
            
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
            
            # Show updated progress for this specific device
            num_files_after = len(glob.glob(os.path.join(self.downloader.processed_path, f'{deviceCode}_*.{filetype}')))
            new_files = num_files_after - num_files_downloaded
            if new_files > 0:
                print_status(f"Added {new_files} new files", "SUCCESS")
            else:
                print_status("No new files added (may be duplicates or anomalies)", "WARNING")
    
    def _download_with_deployment_check(self, deviceCode, start_date, threshold_num, num_days=None, filetype='png', auto_select_deployment=False, duration_seconds=None):
        """Internal method for deployment-aware downloads"""
        print_status("Using deployment-aware download with validation", "INFO")
        
        try:
            if not self.verbose:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.downloader.download_spectrograms_with_deployment_check(
                        deviceCode, start_date, threshold_num, num_days=num_days, 
                        filetype=filetype, auto_select_deployment=auto_select_deployment,
                        duration_seconds=duration_seconds
                    )
            else:
                self.downloader.download_spectrograms_with_deployment_check(
                    deviceCode, start_date, threshold_num, num_days=num_days, 
                    filetype=filetype, auto_select_deployment=auto_select_deployment,
                    duration_seconds=duration_seconds
                )
        except Exception as e:
            print_status(f"Deployment-aware download failed: {e}", "ERROR")
            raise
    
    def show_deployments(self, device_code, start_date, end_date):
        """Show available deployments for a device in a date range"""
        from datetime import datetime
        
        start_date_obj = datetime(*start_date)
        end_date_obj = datetime(*end_date)
        
        print_status(f"Checking deployments for {device_code}...", "PROGRESS")
        
        deployments = self.downloader.show_available_deployments(
            device_code, start_date_obj, end_date_obj, check_data_availability=True
        )
        
        if deployments:
            print_status(f"Found {len(deployments)} deployment(s)", "SUCCESS")
        else:
            print_status("No deployments found in the specified range", "WARNING")
        
        return deployments
    
    def interactive_download(self, device_code, filetype='png'):
        """Launch interactive download process"""
        print_status("Starting interactive download process...", "PROGRESS")
        
        try:
            if not self.verbose:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.downloader.interactive_download_with_deployments(device_code, filetype=filetype)
            else:
                self.downloader.interactive_download_with_deployments(device_code, filetype=filetype)
        except Exception as e:
            print_status(f"Interactive download failed: {e}", "ERROR")
            raise
    
    def download_specific_spectrograms(self, device_times_dict, filetype='png', duration_seconds=300):
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
                self.downloader.setup_directories(filetype, device_id, 'specific_times')
                
                # Download specific spectrogram with custom duration
                if not self.verbose:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.downloader.download_MAT_or_PNG(device_id, start_date_object, filetype=filetype, data_length_seconds=duration_seconds)
                else:
                    self.downloader.download_MAT_or_PNG(device_id, start_date_object, filetype=filetype, data_length_seconds=duration_seconds)
                
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
    
    # Calculate num_days from end_date if provided
    if args.end_date:
        start_date_obj = date(*args.start_date)
        end_date_obj = date(*args.end_date)
        calculated_num_days = (end_date_obj - start_date_obj).days
        # Use the calculated num_days, but allow override if --num-days was explicitly provided
        if not args.num_days:
            args.num_days = calculated_num_days
        print_status(f"End Date: {'-'.join(map(str, args.end_date))}")
        print_status(f"Date Range: {args.num_days} days")
    
    print_status(f"Target Files: {args.threshold}")
    print_status(f"File Type: {args.filetype.upper()}")
    
    if args.num_days:
        print_status(f"Days to Consider: {args.num_days}")
    
    if args.check_deployments:
        print_status("Deployment Checking: ENABLED", "SUCCESS")
        if args.auto_select_deployment:
            print_status("Auto-select Deployment: ENABLED", "SUCCESS")
    else:
        print_status("Deployment Checking: DISABLED", "WARNING")
    
    print_section("Starting Download Process")
    
    try:
        downloader.download_spectrograms_with_sampling_schedule(
            deviceCode=args.device,
            start_date=args.start_date,
            threshold_num=args.threshold,
            num_days=args.num_days,
            filetype=args.filetype,
            check_deployments=args.check_deployments,
            auto_select_deployment=args.auto_select_deployment,
            duration_seconds=args.duration
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
            filetype=args.filetype,
            duration_seconds=args.duration
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
    
    if args.check_deployments:
        print_status("Deployment Checking: ENABLED", "SUCCESS")
        if args.auto_select_deployment:
            print_status("Auto-select Deployment: ENABLED", "SUCCESS")
    else:
        print_status("Deployment Checking: DISABLED", "WARNING")
    
    print_section("Starting Download Process")
    
    # Set up directories for date range mode
    print_status("Setting up directories...", "PROGRESS")
    downloader.downloader.setup_directories(args.filetype, args.device, 'date_range', args.start_date, args.end_date, args.duration)
    
    try:
        if args.check_deployments:
            downloader._download_with_deployment_check(
                args.device, args.start_date, threshold, num_days, 
                args.filetype, args.auto_select_deployment
            )
        else:
            # Use the internal sampling method but with date_range directory
            downloader.downloader.download_spectrograms_with_sampling_schedule(
                args.device, args.start_date, threshold, num_days=num_days, filetype=args.filetype
            )
    except Exception as e:
        if "restricted" in str(e).lower():
            print_status("Some data may be restricted. Check ONC permissions if downloads fail.", "WARNING")
        raise


def check_deployments_mode(args, downloader):
    """Check and display available deployments"""
    print_header("CHECK DEPLOYMENTS MODE")
    
    print_status(f"Device Code: {args.device}")
    print_status(f"Start Date: {'-'.join(map(str, args.start_date))}")
    print_status(f"End Date: {'-'.join(map(str, args.end_date))}")
    
    print_section("Checking Deployments")
    
    try:
        deployments = downloader.show_deployments(args.device, args.start_date, args.end_date)
        
        if deployments:
            print_section("Deployment Summary")
            print_status("Use these deployment periods to plan your downloads", "INFO")
            print_status("You can use --check-deployments flag with sampling or range mode for automatic validation", "INFO")
        else:
            print_status("No deployments found for the specified device and date range", "WARNING")
            print_status("Try expanding your date range or check the device code", "INFO")
            
    except Exception as e:
        print_status(f"Failed to check deployments: {e}", "ERROR")
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
    """Main function to handle command line arguments and run appropriate download mode"""
    
    parser = argparse.ArgumentParser(
        description='Download spectrograms from Ocean Networks Canada (ONC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for beginners)
  python %(prog)s
  
  # Sampling mode with all parameters
  python %(prog)s --mode sampling --device ICLISTENHF6020 --start-date 2020 10 2 --threshold 1000 --check-deployments
  
  # Date range mode
  python %(prog)s --mode range --device ICLISTENHF6020 --start-date 2020 10 2 --end-date 2020 10 5
  
  # Specific times with custom config
  python %(prog)s --mode specific --config my_times.json
  
  # Check deployments
  python %(prog)s --mode check-deployments --device ICLISTENHF6020 --start-date 2020 10 2 --end-date 2020 10 5

  # Custom duration examples
  python %(prog)s --mode sampling --device ICLISTENHF6020 --start-date 2020 10 2 --threshold 100 --duration 600  # 10 minutes
  python %(prog)s --mode specific --config my_times.json --duration 1800  # 30 minutes
        """
    )
    
    # Global options
    parser.add_argument('--mode', 
                        choices=['sampling', 'range', 'specific', 'check-deployments'], 
                        default='sampling',
                        help='Download mode (default: sampling)')
    
    parser.add_argument('--device', 
                        help='Device code (e.g., ICLISTENHF6020). Will prompt if not provided.')
    
    parser.add_argument('--filetype', 
                        choices=['png', 'mat'], 
                        default='mat',
                        help='File type to download (default: mat)')
    
    parser.add_argument('--check-deployments', 
                        action='store_true',
                        help='Enable deployment validation (recommended)')
    
    parser.add_argument('--auto-select-deployment', 
                        action='store_true',
                        help='Auto-select best deployment for multiple options')
    
    parser.add_argument('--verbose', 
                        action='store_true',
                        help='Show detailed ONC API messages')
    
    # Duration parameter - NEW
    parser.add_argument('--duration', 
                        type=int,
                        help='Duration of each spectrogram in seconds (default: 300 = 5 minutes). ' +
                             'Examples: 300=5min, 600=10min, 1800=30min, 3600=1hour')
    
    # Date/time options
    parser.add_argument('--start-date', 
                        nargs=3, 
                        type=int, 
                        metavar=('YEAR', 'MONTH', 'DAY'),
                        help='Start date as YEAR MONTH DAY (e.g., 2020 10 2). Will prompt if not provided.')
    
    parser.add_argument('--end-date', 
                        nargs=3, 
                        type=int, 
                        metavar=('YEAR', 'MONTH', 'DAY'),
                        help='End date as YEAR MONTH DAY (required for range and check-deployments modes)')
    
    # Mode-specific options
    parser.add_argument('--threshold', 
                        type=int,
                        help='Number of spectrograms to download (for sampling mode). Will prompt if not provided.')
    
    parser.add_argument('--num-days', 
                        type=int,
                        help='Override calculated date range for sampling mode')
    
    parser.add_argument('--config', 
                        help='JSON configuration file for specific times mode')
    
    # Utility options
    parser.add_argument('--create-example-config', 
                        action='store_true',
                        help='Create example configuration file for specific times mode')
    
    args = parser.parse_args()
    
    # Handle --create-example-config
    if args.create_example_config:
        create_example_config()
        return
    
    try:
        # Load configuration and setup
        onc_token, data_dir = load_config()
        downloader = VerboseSpectrogramDownloader(onc_token, data_dir, verbose=args.verbose)
        
        print_header("ONC SPECTROGRAM DOWNLOADER")
        
        # Collect any missing parameters interactively
        args = collect_missing_parameters(args, downloader)
        
        print_status(f"Spectrogram Duration: {args.duration} seconds ({args.duration/60:.1f} minutes)", "INFO")
        
        # Route to appropriate download function
        if args.mode == 'sampling':
            download_with_sampling_schedule(args, downloader)
        elif args.mode == 'range':
            download_date_range(args, downloader)
        elif args.mode == 'specific':
            download_specific_times(args, downloader)
        elif args.mode == 'check-deployments':
            check_deployments_mode(args, downloader)
        
        print_status("Download process completed!", "SUCCESS")
        
    except KeyboardInterrupt:
        print_status("\nDownload interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        print_status(f"Error: {e}", "ERROR")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def collect_missing_parameters(args, downloader):
    """
    Interactively collect missing parameters from the user.
    
    :param args: Parsed command line arguments
    :param downloader: VerboseSpectrogramDownloader instance
    :return: Updated args with collected parameters
    """
    print_section("Parameter Collection")
    
    # Collect device if missing (except for specific mode which uses config)
    if args.mode != 'specific' and not args.device:
        args.device = prompt_for_device(downloader)
        if not args.device:
            print_status("Device selection cancelled", "ERROR")
            sys.exit(1)
    
    # Collect dates if missing - for interactive mode, always ask for both start and end dates
    if args.mode in ['sampling', 'range', 'check-deployments']:
        if not args.start_date:
            args.start_date = prompt_for_dates(args.device, downloader, args.mode)
            if not args.start_date:
                print_status("Date selection cancelled", "ERROR")
                sys.exit(1)
        
        # For interactive usage, ask for end date for all modes to define the time range
        if not args.end_date:
            if args.mode == 'sampling':
                print_status("For sampling mode, you can specify an end date to limit the sampling period", "INFO")
            
            args.end_date = prompt_for_end_date(args.start_date, required=(args.mode in ['range', 'check-deployments']))
            if not args.end_date and args.mode in ['range', 'check-deployments']:
                print_status("End date is required for this mode", "ERROR")
                sys.exit(1)
    
    # Collect threshold for sampling mode
    if args.mode == 'sampling' and not args.threshold:
        args.threshold = prompt_for_threshold()
        if not args.threshold:
            print_status("Threshold selection cancelled", "ERROR")
            sys.exit(1)
    
    # Collect duration if not specified
    if args.duration is None:
        args.duration = prompt_for_duration()
        if args.duration is None:
            print_status("Duration selection cancelled", "ERROR")
            sys.exit(1)
    
    return args


def prompt_for_device(downloader):
    """
    Prompt user to select a device from available hydrophones.
    
    :param downloader: VerboseSpectrogramDownloader instance
    :return: Selected device code or None if cancelled
    """
    print_status("Device not specified. Fetching available hydrophone devices...", "INFO")
    
    try:
        # Get all deployments to find available devices
        all_deployments = downloader.downloader._get_cached_deployments()
        
        # Get unique device codes
        device_codes = sorted(list(set(dep.device_code for dep in all_deployments)))
        
        if not device_codes:
            print_status("No hydrophone devices found", "ERROR")
            return None
        
        print(f"\nAvailable hydrophone devices ({len(device_codes)} found):")
        for i, device_code in enumerate(device_codes, 1):
            # Get deployment count for this device
            device_deployments = [dep for dep in all_deployments if dep.device_code == device_code]
            latest_deployment = max(device_deployments, key=lambda x: x.begin_date)
            
            print(f"  {i:2d}. {device_code}")
            print(f"      Location: {latest_deployment.location_name}")
            print(f"      Latest deployment: {latest_deployment.begin_date.strftime('%Y-%m-%d')} to {latest_deployment.end_date.strftime('%Y-%m-%d') if latest_deployment.end_date else 'ongoing'}")
        
        while True:
            try:
                choice = input(f"\nSelect device (1-{len(device_codes)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(device_codes):
                    selected_device = device_codes[choice_num - 1]
                    print_status(f"Selected device: {selected_device}", "SUCCESS")
                    return selected_device
                else:
                    print_status(f"Please enter a number between 1 and {len(device_codes)}", "WARNING")
            except ValueError:
                print_status("Please enter a valid number or 'q' to quit", "WARNING")
    
    except Exception as e:
        print_status(f"Error fetching devices: {e}", "ERROR")
        return None


def prompt_for_dates(device_code, downloader, mode):
    """
    Prompt user to select dates based on available deployments.
    
    :param device_code: Selected device code
    :param downloader: VerboseSpectrogramDownloader instance
    :param mode: Download mode
    :return: Start date tuple (year, month, day) or None if cancelled
    """
    print_status(f"Dates not specified. Showing available deployments for {device_code}...", "INFO")
    
    try:
        # Get deployments for this device
        all_deployments = downloader.downloader._get_cached_deployments()
        device_deployments = [dep for dep in all_deployments if dep.device_code == device_code]
        
        if not device_deployments:
            print_status(f"No deployments found for device {device_code}", "ERROR")
            return None
        
        print(f"\nAvailable deployment periods for {device_code}:")
        for i, deployment in enumerate(device_deployments, 1):
            end_str = deployment.end_date.strftime('%Y-%m-%d') if deployment.end_date else 'ongoing'
            print(f"  {i}. {deployment.begin_date.strftime('%Y-%m-%d')} to {end_str}")
            print(f"     Location: {deployment.location_name}")
        
        while True:
            try:
                date_input = input(f"\nEnter start date (YYYY-MM-DD) or 'q' to quit: ").strip()
                if date_input.lower() == 'q':
                    return None
                
                # Parse the date
                start_date = datetime.strptime(date_input, '%Y-%m-%d')
                
                # Check if date falls within any deployment
                date_covered = False
                for deployment in device_deployments:
                    dep_start = deployment.begin_date.replace(tzinfo=None)
                    dep_end = (deployment.end_date.replace(tzinfo=None) if deployment.end_date 
                              else datetime.now())
                    
                    if dep_start <= start_date <= dep_end:
                        date_covered = True
                        print_status(f"✅ Date {date_input} is covered by deployment at {deployment.location_name}", "SUCCESS")
                        break
                
                if not date_covered:
                    print_status(f"⚠️ Warning: Date {date_input} may not be covered by any deployment", "WARNING")
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                return (start_date.year, start_date.month, start_date.day)
                
            except ValueError:
                print_status("Invalid date format. Please use YYYY-MM-DD", "WARNING")
    
    except Exception as e:
        print_status(f"Error checking deployments: {e}", "ERROR")
        return None


def prompt_for_end_date(start_date_tuple, required=False):
    """
    Prompt user for end date.
    
    :param start_date_tuple: Start date tuple (year, month, day)
    :param required: Whether end date is required
    :return: End date tuple (year, month, day) or None if cancelled
    """
    start_date_str = f"{start_date_tuple[0]}-{start_date_tuple[1]:02d}-{start_date_tuple[2]:02d}"
    
    if not required:
        prompt_text = f"Enter end date (YYYY-MM-DD) [start: {start_date_str}], 'skip' to use default period, or 'q' to quit: "
    else:
        prompt_text = f"Enter end date (YYYY-MM-DD) [start: {start_date_str}] or 'q' to quit: "
    
    while True:
        try:
            date_input = input(prompt_text).strip()
            if date_input.lower() == 'q':
                return None
            
            if not required and date_input.lower() == 'skip':
                print_status("Skipping end date - will use sampling schedule default", "INFO")
                return None
            
            end_date = datetime.strptime(date_input, '%Y-%m-%d')
            start_date_obj = datetime(*start_date_tuple)
            
            if end_date <= start_date_obj:
                print_status("End date must be after start date", "WARNING")
                continue
            
            # Calculate and show the date range
            date_range = (end_date - start_date_obj).days
            print_status(f"Selected date range: {date_range} days", "SUCCESS")
            
            return (end_date.year, end_date.month, end_date.day)
            
        except ValueError:
            print_status("Invalid date format. Please use YYYY-MM-DD", "WARNING")


def prompt_for_threshold():
    """
    Prompt user for number of spectrograms to download.
    
    :return: Number of spectrograms or None if cancelled
    """
    while True:
        try:
            threshold_input = input("How many spectrograms do you want to download? (or 'q' to quit): ").strip()
            if threshold_input.lower() == 'q':
                return None
            
            threshold = int(threshold_input)
            if threshold <= 0:
                print_status("Number must be positive", "WARNING")
                continue
            
            return threshold
            
        except ValueError:
            print_status("Please enter a valid number", "WARNING")


def prompt_for_duration():
    """
    Prompt user for spectrogram duration.
    
    :return: Duration in seconds or None if cancelled
    """
    print_status("Spectrogram duration not specified", "INFO")
    print("\nCommon duration options:")
    print("  1. 300 seconds (5 minutes) - detailed analysis [DEFAULT]")
    print("  2. 600 seconds (10 minutes) - balanced detail/coverage")
    print("  3. 1800 seconds (30 minutes) - standard for many studies")
    print("  4. 3600 seconds (1 hour) - long-term patterns")
    print("  5. Custom duration")
    
    while True:
        try:
            choice = input("\nSelect duration option (1-5, or press Enter for default) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            # Default to option 1 if user just presses Enter
            if choice == '':
                choice = '1'
            
            if choice == '1':
                duration = 300
            elif choice == '2':
                duration = 600
            elif choice == '3':
                duration = 1800
            elif choice == '4':
                duration = 3600
            elif choice == '5':
                # Custom duration
                while True:
                    try:
                        custom_input = input("Enter custom duration in seconds (or 'q' to quit): ").strip()
                        if custom_input.lower() == 'q':
                            return None
                        
                        duration = int(custom_input)
                        if duration <= 0:
                            print_status("Duration must be positive", "WARNING")
                            continue
                        if duration > 86400:  # 24 hours
                            print_status("Duration should not exceed 24 hours (86400 seconds)", "WARNING")
                            confirm = input("Continue anyway? (y/n): ").strip().lower()
                            if confirm != 'y':
                                continue
                        break
                    except ValueError:
                        print_status("Please enter a valid number", "WARNING")
            else:
                print_status("Please select 1-5 or press Enter for default", "WARNING")
                continue
            
            minutes = duration / 60
            if minutes < 60:
                duration_str = f"{minutes:.1f} minutes"
            else:
                hours = minutes / 60
                duration_str = f"{hours:.1f} hours"
            
            print_status(f"Selected duration: {duration} seconds ({duration_str})", "SUCCESS")
            return duration
            
        except ValueError:
            print_status("Please enter a valid option", "WARNING")


if __name__ == "__main__":
    main() 