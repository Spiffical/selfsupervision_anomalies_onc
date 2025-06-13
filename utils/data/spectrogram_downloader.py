import os
import numpy as np
import datetime as dt
from datetime import date, datetime, timedelta, timezone
from onc.onc import ONC
import random
from .trim_image import crop_image
from .segment import segment2
from .deployment_checker import HydrophoneDeploymentChecker
from PIL import Image
import glob
import shutil
import scipy.io
import concurrent.futures
from typing import List, Dict, Any
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper function to ensure datetime is timezone-aware
def ensure_timezone_aware(dt_obj, tz=timezone.utc):
    """Convert timezone-naive datetime to timezone-aware datetime."""
    if dt_obj.tzinfo is None:
        return dt_obj.replace(tzinfo=tz)
    return dt_obj

class SpectrogramDownloader:
    def __init__(self, ONC_token, parent_dir):
        self.onc = ONC(ONC_token)
        self.parent_dir = parent_dir
        self.delim = '/' if os.name == 'posix' else '\\'
        # Initialize deployment checker
        self.deployment_checker = HydrophoneDeploymentChecker(ONC_token)
        # Cache for deployment data to avoid redundant API calls
        self._deployment_cache = None
        self._cache_timestamp = None
        # Maximum number of parallel downloads
        self.max_workers = 4
        # Batch size for API requests
        self.batch_size = 10
        
    def setup_directories(self, filetype, device_code=None, download_method=None, start_date=None, end_date=None, duration_seconds=None):
        """Setup directory structure with optional device, method, and date organization"""
        if device_code and download_method:
            # Create method folder name with date information
            method_folder = self._create_method_folder_name(download_method, start_date, end_date, duration_seconds)
            
            # New organized structure: data/DEVICE/METHOD_DATES/
            base_path = os.path.join(self.parent_dir, device_code, method_folder)
            
            # Create mat directory with processed and rejects subdirectories
            self.input_path = os.path.join(base_path, filetype, '')
            self.processed_path = os.path.join(self.input_path, 'processed', '')
            self.anom_path = os.path.join(self.input_path, 'rejects', '')
            
            # Create flac directory at the same level as mat
            self.flac_path = os.path.join(base_path, 'flac', '')
        else:
            # Legacy structure for backwards compatibility
            self.input_path = os.path.join(self.parent_dir, filetype, '')
            self.processed_path = os.path.join(self.input_path, 'processed', '')
            self.anom_path = os.path.join(self.input_path, 'rejects', '')
            self.flac_path = os.path.join(self.parent_dir, 'flac', '')
        
        self.onc.outPath = self.input_path

        # Create all necessary directories
        for folder_path in [self.parent_dir, self.input_path, self.processed_path, self.anom_path, self.flac_path]:
            os.makedirs(folder_path, exist_ok=True)
    
    def _create_method_folder_name(self, download_method, start_date=None, end_date=None, duration_seconds=None):
        """Create a descriptive folder name including method and dates"""
        folder_name = download_method
        
        # Add date range information
        if start_date:
            if isinstance(start_date, (list, tuple)):
                # Handle tuple format (year, month, day)
                start_str = f"{start_date[0]}-{start_date[1]:02d}-{start_date[2]:02d}"
            elif hasattr(start_date, 'strftime'):
                # Handle datetime object
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)
            
            folder_name += f"_{start_str}"
            
            if end_date:
                if isinstance(end_date, (list, tuple)):
                    # Handle tuple format (year, month, day)
                    end_str = f"{end_date[0]}-{end_date[1]:02d}-{end_date[2]:02d}"
                elif hasattr(end_date, 'strftime'):
                    # Handle datetime object
                    end_str = end_date.strftime('%Y-%m-%d')
                else:
                    end_str = str(end_date)
                
                folder_name += f"_to_{end_str}"
        
        return folder_name

    def start_and_end_strings(self, start_date_object, time_delta):
        start_time_str = start_date_object.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        end_object = start_date_object + time_delta
        end_time_str = end_object.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        return start_time_str, end_time_str
    
    # Function to check if a file for a given date already exists
    def check_existing_files(self, device_code, date_list):
        print(f'Checking for existing files for {device_code} in {self.processed_path}')

        # Get all files in the directory
        existing_files = glob.glob(os.path.join(self.processed_path, f"{device_code}_*.mat"))
        
        # Extract dates from filenames
        existing_dates = set()
        for file in existing_files:
            filename = os.path.basename(file)
            # Extract the date part from the filename
            date_str = filename.split('_')[1][:8]  # Extracts '20201002' from 'ICLISTENHF6020_20201002T000000.000Z...'
            date_obj = dt.datetime.strptime(date_str, '%Y%m%d')
            existing_dates.add(date_obj.date())  # Add date object to the set
        
        # Filter out dates that already have files
        # Handle both datetime and date objects in date_list
        filtered_dates = []
        for d in date_list:
            check_date = d.date() if hasattr(d, 'date') else d
            if check_date not in existing_dates:
                filtered_dates.append(d)

        return filtered_dates

    def sampling_schedule(self, deviceCode, threshold_num, year, month, day, day_interval=None, num_days=None, spectrograms_per_batch=6):
        spect_length = 300
        sample_time_per_day = 1799
        min_per_day = (sample_time_per_day + 1) / spect_length

        start_date = date(year, month, day)
        if num_days is None:
            today = date.today()
            num_days = (today - start_date).days

        time_delta = dt.timedelta(num_days)
        start_time_str, end_time_str = self.start_and_end_strings(start_date, time_delta)

        filters = {
            'deviceCode': deviceCode,
            'dateFrom': start_time_str,
            'dateTo': end_time_str,
            'extension': 'png'
        }

        result = self.onc.getListByDevice(filters, allPages=True)
        spect_png_files = [s for s in result['files'] if "Z-spect.png" in s]

        day_strings = [spect_png_file.split('_')[1] for spect_png_file in spect_png_files]
        days_int = [int(day_str[0:8]) for day_str in day_strings]
        unique_days = np.unique(days_int)
        num_days_available = len(unique_days)
        print(f'Number of days available: {num_days_available}')

        if day_interval == 1:
            sample_time_per_day = 86400 - 1
            num_per_day = 86400 / spect_length
        else:
            if day_interval is None:
                day_interval = num_days_available / (threshold_num * 1.1 / min_per_day)
                if day_interval > 1:
                    day_interval = int(np.round(day_interval))
                else:
                    day_interval = 1

            if len(np.arange(0, num_days_available, day_interval)) * min_per_day < threshold_num:
                num_per_day = int(np.ceil(threshold_num * 1.1 / len(np.arange(0, num_days_available, day_interval))))
                sample_time_per_day = spect_length * num_per_day - 1
            else:
                num_per_day = int(min_per_day)

        print(f'Plan is to retrieve {num_per_day} spectrograms per day')

        # Calculate how many requests we need (each request gets exactly spectrograms_per_batch spectrograms)
        total_requests_needed = int(np.ceil(threshold_num / spectrograms_per_batch))
        actual_spectrograms_to_download = total_requests_needed * spectrograms_per_batch
        
        print(f'Target: {threshold_num} spectrograms')
        print(f'Each request gets {spectrograms_per_batch} spectrograms')
        print(f'Therefore need {total_requests_needed} requests')
        print(f'This will download {actual_spectrograms_to_download} spectrograms total')
        
        # Generate sampling schedule - distribute requests evenly across the FULL requested time range
        date_list = []
        
        # Calculate how many days we'll sample from
        requests_per_day = max(1, int(np.ceil(total_requests_needed / min(total_requests_needed, num_days_available))))
        num_sampling_days = int(np.ceil(total_requests_needed / requests_per_day))
        
        print(f'Will make {requests_per_day} requests per day across {num_sampling_days} days')
        print(f'Sampling across full requested range of {num_days} days')
        
        for day_idx in range(num_sampling_days):
            if len(date_list) >= total_requests_needed:
                break
                
            # Calculate day offset - spread across the FULL requested date range (num_days)
            # This ensures we sample from start to end of the requested period
            if num_sampling_days > 1:
                day_offset = day_idx * (num_days - 1) // (num_sampling_days - 1)
            else:
                day_offset = 0
                
            # Ensure we don't exceed the requested date range
            if day_offset >= num_days:
                day_offset = num_days - 1
                
            # Calculate the actual date for this day offset
            sample_date = start_date + timedelta(days=day_offset)
            
            # Add the specified number of requests for this day
            for request_in_day in range(requests_per_day):
                if len(date_list) >= total_requests_needed:
                    break
                    
                # Distribute hours across the day for multiple requests
                # OR use random sampling for better temporal diversity
                if requests_per_day > 1:
                    # Multiple requests per day - distribute hours evenly within the day
                    hour_offset = request_in_day * (24 // requests_per_day)
                else:
                    # One request per day - use random hour for maximum temporal diversity
                    # Use day_idx as seed for reproducible but varied sampling
                    random.seed(day_idx + hash(str(sample_date)))  # Reproducible but varied
                    hour_offset = random.randint(0, 23)
                    
                # Convert date to datetime and add random minutes for even better diversity
                # Use same seed for reproducible minute selection
                minute_offset = random.randint(0, 59)
                sample_datetime = datetime.combine(sample_date, datetime.min.time()) + timedelta(hours=hour_offset, minutes=minute_offset)
                
                date_list.append(sample_datetime)
                
        print(f'✅ Generated {len(date_list)} requests across {num_sampling_days} days')
        print(f'This will download exactly {len(date_list) * spectrograms_per_batch} spectrograms total')

        return date_list, sample_time_per_day

    def download_MAT_or_PNG(self, deviceCode, start_date_object, filetype='png', spectrograms_per_batch=6, download_flac=False):
        """
        Download MAT or PNG files for a given time period.
        
        :param deviceCode: ONC device code
        :param start_date_object: Start date and time
        :param filetype: Type of file to download ('png' or 'mat')
        :param spectrograms_per_batch: Number of 5-minute spectrograms to download per batch
        :param download_flac: Whether to download corresponding FLAC files
        """
        # Calculate duration based on number of spectrograms (each is 5 minutes = 300 seconds)
        # Use exact duration to get precisely the requested number of spectrograms
        data_length_seconds = (spectrograms_per_batch - 1) * 300
        
        time_delta = dt.timedelta(0, data_length_seconds)
        start_time, end_time = self.start_and_end_strings(start_date_object, time_delta)

        if filetype == 'mat':
            # Format the date nicely for logging
            date_str = start_date_object.strftime('%Y-%m-%d')
            time_str = start_date_object.strftime('%H:%M:%S')
            day_name = start_date_object.strftime('%A')
            print(f'📅 Downloading data for {day_name}, {date_str} at {time_str} (requesting {spectrograms_per_batch} spectrograms)')
            dataProductCode = 'HSD'
            filters = {
                'dataProductCode': dataProductCode,
                'deviceCode': deviceCode,
                'dateFrom': start_time,
                'dateTo': end_time,
                'extension': 'mat',
                'dpo_hydrophoneDataDiversionMode': 'OD',
                'dpo_spectralDataDownsample': 2
            }
            
            # Request data product
            result = self.onc.requestDataProduct(filters)
            logger.info(f"Request Id: {result['dpRequestId']}")
            logger.info(f"Estimated files: {spectrograms_per_batch} spectrograms + 1 metadata = {spectrograms_per_batch + 1} files")
            
            # Run data product and wait for completion
            run_start = time.time()
            run_data = self.onc.runDataProduct(result['dpRequestId'], waitComplete=True)
            logger.info(f"Data product run completed in {time.time() - run_start:.2f}s")
            
            # Download all files from the run
            if 'runIds' in run_data and run_data['runIds']:
                logger.info("Downloading files...")
                download_start = time.time()
                self.onc.downloadDataProduct(run_data['runIds'][0])
                logger.info(f"Files downloaded successfully in {time.time() - download_start:.2f}s")
                
                # Download FLAC files if requested
                if download_flac:
                    flac_start = time.time()
                    self.download_flac_files(deviceCode, start_time, end_time)
                    logger.info(f"FLAC files downloaded in {time.time() - flac_start:.2f}s")
                
                # Process downloaded files
                process_start = time.time()
                self.process_spectrograms(filetype)
                logger.info(f"Files processed in {time.time() - process_start:.2f}s")
                
                # Log progress
                num_files = len(glob.glob(os.path.join(self.processed_path, f'*.{filetype}')))
                logger.info(f"Progress: {num_files} files downloaded")
                
        elif filetype == 'png':
            # Format the date nicely for logging
            date_str = start_date_object.strftime('%Y-%m-%d')
            time_str = start_date_object.strftime('%H:%M:%S')
            day_name = start_date_object.strftime('%A')
            print(f'📅 Downloading data for {day_name}, {date_str} at {time_str} (requesting {spectrograms_per_batch} spectrograms)')
            filters = {
                'deviceCode': deviceCode,
                'dateFrom': start_time,
                'dateTo': end_time,
                'extension': 'png'
            }
            result = self.onc.getListByDevice(filters, allPages=True)
            spect_png_files = [s for s in result['files'] if "Z-spect.png" in s]
            
            logger.info(f"Found {len(spect_png_files)} PNG files")
            
            # Download all PNG files in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.onc.getFile, png_file) for png_file in spect_png_files]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error downloading PNG file: {e}")
            
            # Download FLAC files if requested
            if download_flac:
                flac_start = time.time()
                self.download_flac_files(deviceCode, start_time, end_time)
                logger.info(f"FLAC files downloaded in {time.time() - flac_start:.2f}s")
            
            # Process downloaded files
            self.process_spectrograms(filetype)

    def download_flac_files(self, deviceCode, start_time, end_time):
        """
        Download FLAC audio files corresponding to the same time window as spectrograms.
        Uses parallel downloads for better performance.
        
        :param deviceCode: ONC device code
        :param start_time: Start time string in ISO format
        :param end_time: End time string in ISO format
        """
        logger.info(f'Finding FLAC audio files for {deviceCode} from {start_time} to {end_time}')
        
        # Store original path to ensure it's always restored
        original_output_path = self.onc.outPath
        
        try:
            # Search for FLAC files using archive file API
            search_start = time.time()
            filters = {
                'deviceCode': deviceCode,
                'dateFrom': start_time,
                'dateTo': end_time,
                'extension': 'flac'
            }
            
            result = self.onc.getListByDevice(filters, allPages=True)
            logger.info(f"FLAC file search completed in {time.time() - search_start:.2f}s")
            
            if 'files' in result and result['files']:
                flac_files = [f for f in result['files'] if f.lower().endswith('.flac')]
                
                if flac_files:
                    # Temporarily set output path to flac directory
                    self.onc.outPath = self.flac_path
                    
                    logger.info(f'Found {len(flac_files)} FLAC file(s)')
                    
                    # Download files in parallel using ThreadPoolExecutor
                    download_start = time.time()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        # Create a list of futures
                        futures = []
                        for flac_file in flac_files:
                            logger.info(f'Queuing FLAC: {flac_file}')
                            futures.append(executor.submit(self._download_flac_with_retry, flac_file))
                        
                        # Wait for all downloads to complete
                        concurrent.futures.wait(futures)
                    logger.info(f"FLAC files downloaded in {time.time() - download_start:.2f}s")
                else:
                    logger.info('No FLAC files found in the specified time range')
            else:
                logger.info('No files found or error in API response for FLAC search')
                
        except Exception as e:
            logger.error(f'Error searching for FLAC files: {e}')
        finally:
            # Always restore original output path
            self.onc.outPath = original_output_path

    def _download_flac_with_retry(self, flac_file: str, max_retries: int = 3, retry_delay: int = 5) -> bool:
        """
        Download a single FLAC file with retry logic.
        
        :param flac_file: Name of the FLAC file to download
        :param max_retries: Maximum number of retry attempts
        :param retry_delay: Delay between retries in seconds
        :return: True if download successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                self.onc.getFile(flac_file)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f'Error downloading {flac_file} (attempt {attempt + 1}/{max_retries}): {e}')
                    time.sleep(retry_delay)
                else:
                    print(f'Failed to download {flac_file} after {max_retries} attempts: {e}')
                    return False
        return False

    def check_for_anomalies(self, file_path, file1, file2):
        try:
            image_obj = None
            if file_path.lower().endswith('.png'):
                image_obj = Image.open(file_path)
                image_obj = np.transpose(image_obj, [1, 0, 2])
            elif file_path.lower().endswith('.mat'):
                mat_data = scipy.io.loadmat(file_path)
                if 'SpectData' in mat_data:
                    image_obj = mat_data['SpectData']['PSD'][0,0]
                else:
                    raise ValueError('No "SpectData" key found in .mat file')

            if image_obj is not None:
                s = np.zeros([np.shape(image_obj)[0], 1])

                anomaly_found = 0
                anom_indices_black = []
                anom_indices_white = []
                for ii in np.arange(0, np.shape(image_obj)[0]):
                    s[ii] = np.sum(image_obj[ii])
                    if s[ii] < 500:
                        anomaly_found = 1
                        anom_indices_black.append(ii)
                    elif s[ii] > 568000:
                        anomaly_found = 2
                        anom_indices_white.append(ii)

                if anomaly_found > 0:
                    file1.write(file_path + "\n")

                    if len(anom_indices_black) > 0:
                        seg = segment2(anom_indices_black)
                        num_segments = seg.shape[0]
                        summary_string = f"{num_segments} black segment(s) found, with entries [{', '.join(' to '.join(map(str, row)) for row in seg)}]"
                        print(f'{summary_string}: {os.path.basename(file_path)}')
                        file2.write(f'{summary_string}: {os.path.basename(file_path)}\n')

                    if len(anom_indices_white) > 0:
                        seg = segment2(anom_indices_white)
                        num_segments = seg.shape[0]
                        summary_string = f"{num_segments} white segment(s) found, with entries [{', '.join(' to '.join(map(str, row)) for row in seg)}]"
                        print(f'{summary_string}: {os.path.basename(file_path)}')
                        file2.write(f'{summary_string}: {os.path.basename(file_path)}\n')

                    # Move file to the rejects folder, check if it's already there
                    if os.path.exists(os.path.join(self.anom_path, os.path.basename(file_path))):
                        # Remove the file from the input folder
                        print(f'File {file_path} already exists in the rejects folder. Removing from the input folder.')
                        os.remove(file_path)
                    else:
                        shutil.move(file_path, self.anom_path)

        except Exception as e:
            print(f'Error encountered for: {file_path}, {e}')
            file1.write(file_path + "\n")
            file2.write(f'Error encountered for: {file_path}\n')
            # Move file to the rejects folder, check if it's already there
            if os.path.exists(os.path.join(self.anom_path, os.path.basename(file_path))):
                # Remove the file from the input folder
                print(f'File {file_path} already exists in the rejects folder. Removing from the input folder.')
                os.remove(file_path)
            else:
                shutil.move(file_path, self.anom_path)

    def process_spectrograms(self, filetype='png'):
        process_start = time.time()
        logger.info("Starting spectrogram processing")
        
        with open(os.path.join(self.processed_path, 'anomalous_files.txt'), 'w') as file1, \
            open(os.path.join(self.processed_path, 'anomalous_file_summary.txt'), 'w') as file2:

            if filetype == 'png':
                input_image_paths = glob.glob(os.path.join(self.input_path, '*.png'))
                logger.info(f"Found {len(input_image_paths)} PNG files to process")

                for input_image in input_image_paths:
                    image_area = (107, 67, 1042, 810)
                    crop_image(input_image, self.processed_path, image_area)

                    image_name = os.path.basename(input_image)
                    trimmed_path = os.path.join(self.processed_path, image_name)

                    self.check_for_anomalies(trimmed_path, file1, file2)

                [os.remove(os.path.join(self.input_path, file_name)) for file_name in os.listdir(self.input_path) if file_name.lower().endswith('.png')]
            elif filetype == 'mat':
                mat_paths = glob.glob(os.path.join(self.input_path, '*.mat'))
                logger.info(f"Found {len(mat_paths)} MAT files to process")
                
                for mat_path in mat_paths:
                    self.check_for_anomalies(mat_path, file1, file2)
                
                # Move files to the processed folder, check if they're already there
                for file_name in os.listdir(self.input_path):
                    if file_name.lower().endswith('.mat'):
                        if os.path.exists(os.path.join(self.processed_path, file_name)):
                            # Remove the file from the input folder
                            logger.info(f'File {file_name} already exists in the processed folder. Removing from the input folder.')
                            os.remove(os.path.join(self.input_path, file_name))
                        else:
                            shutil.move(os.path.join(self.input_path, file_name), os.path.join(self.processed_path, file_name))

        logger.info(f"Spectrogram processing completed in {time.time() - process_start:.2f}s")

    def download_spectrograms_with_sampling_schedule(self, deviceCode, start_date, threshold_num, num_days=None, filetype='png', spectrograms_per_batch=6, download_flac=False):
        """
        Download spectrograms based on a sampling schedule.
        
        :param deviceCode: ONC device code
        :param start_date: Start date for sampling (tuple: year, month, day)
        :param threshold_num: Number of samples to take
        :param num_days: Number of days to sample (optional)
        :param filetype: Type of file to download ('png' or 'mat')
        :param spectrograms_per_batch: Number of 5-minute spectrograms to download per batch
        :param download_flac: Whether to download corresponding FLAC files
        """
        schedule_start = time.time()
        logger.info(f"Starting sampling schedule download for {deviceCode} from {start_date}")
        logger.info(f"Batch size: {spectrograms_per_batch} spectrograms per request")
        
        # Generate sampling schedule first to determine actual date range
        schedule_start_time = time.time()
        year, month, day = start_date
        date_object_list, sample_time_per_day = self.sampling_schedule(
            deviceCode, threshold_num, year, month, day, num_days=num_days, spectrograms_per_batch=spectrograms_per_batch
        )
        logger.info(f"Generated sampling schedule in {time.time() - schedule_start_time:.2f}s")
        
        if not date_object_list:
            logger.error("Failed to generate sampling schedule")
            return

        # Calculate actual date range from the sampling schedule
        actual_start_date = min(date_object_list).date()
        actual_end_date = max(date_object_list).date()
        
        # Convert to tuple format for directory setup
        start_date_tuple = (actual_start_date.year, actual_start_date.month, actual_start_date.day)
        end_date_tuple = (actual_end_date.year, actual_end_date.month, actual_end_date.day)
        
        # Calculate duration for directory setup (used for folder naming)
        duration_seconds = (spectrograms_per_batch * 300) - 1

        # Set up directories with the actual date range
        self.setup_directories(filetype, deviceCode, 'sampling', start_date_tuple, end_date_tuple, duration_seconds)

        # Check for existing files and filter the dates
        filtered_date_list = self.check_existing_files(deviceCode, date_object_list)
        
        # Update the date list to only include files that don't already exist
        date_object_list = filtered_date_list

        # Download files for each request
        total_requests = len(date_object_list)
        logger.info(f"Starting download of {total_requests} requests")
        
        # Show summary of days being downloaded
        unique_dates = sorted(set(ts.date() for ts in date_object_list))
        print(f"📅 Will download data from {len(unique_dates)} unique days:")
        for date in unique_dates:
            day_name = date.strftime('%A')
            date_str = date.strftime('%Y-%m-%d')
            requests_on_day = len([ts for ts in date_object_list if ts.date() == date])
            spectrograms_on_day = requests_on_day * spectrograms_per_batch
            print(f"   • {day_name}, {date_str} ({requests_on_day} requests = {spectrograms_on_day} spectrograms)")
        
        for i, request_time in enumerate(date_object_list, 1):
            request_start = time.time()
            logger.info(f"Processing request {i}/{total_requests}: {request_time}")
            
            # Download files for this request (this will get spectrograms_per_batch + 1 files)
            self.download_MAT_or_PNG(deviceCode, request_time, filetype, spectrograms_per_batch, download_flac)
            
            logger.info(f"Completed request {i}/{total_requests} in {time.time() - request_start:.2f}s")
            logger.info(f"Overall progress: {i}/{total_requests} requests completed")
        
        total_time = time.time() - schedule_start
        logger.info(f"Completed all downloads in {total_time:.2f}s")
        logger.info(f"Average time per request: {total_time/total_requests:.2f}s")

    def download_spectrograms_with_deployment_check(self, deviceCode, start_date, threshold_num, num_days=None, filetype='png', auto_select_deployment=False, spectrograms_per_batch=6, download_flac=False):
        """
        Download spectrograms with deployment checking enabled.
        
        :param deviceCode: ONC device code
        :param start_date: Start date for sampling
        :param threshold_num: Number of samples to take
        :param num_days: Number of days to sample (optional)
        :param filetype: Type of file to download ('png' or 'mat')
        :param auto_select_deployment: Whether to automatically select the best deployment
        :param spectrograms_per_batch: Number of 5-minute spectrograms to download per batch
        :param download_flac: Whether to download corresponding FLAC files
        """
        logger.info(f"Starting deployment-aware download for {deviceCode}")
        logger.info(f"Batch size: {spectrograms_per_batch} spectrograms per request")
        
        # Get deployment information
        deployment_info = self.deployment_checker.get_deployment_info(deviceCode)
        if not deployment_info:
            logger.error(f"Could not get deployment information for {deviceCode}")
            return

        # Generate sampling schedule
        sampling_schedule = self.deployment_checker.generate_sampling_schedule(
            deployment_info, 
            start_date, 
            threshold_num, 
            num_days
        )
        
        if not sampling_schedule:
            logger.error("Failed to generate sampling schedule")
            return

        # Calculate duration for directory setup
        duration_seconds = (spectrograms_per_batch * 300) - 1

        # Set up directories for the download
        self.setup_directories(deviceCode, filetype, start_date, sampling_schedule[-1], duration_seconds)

        # Download files for each time slot with deployment checking
        total_slots = len(sampling_schedule)
        logger.info(f"Starting download of {total_slots} time slots with deployment checking")
        
        for i, time_slot in enumerate(sampling_schedule, 1):
            slot_start = time.time()
            logger.info(f"Processing slot {i}/{total_slots}: {time_slot}")
            
            # Download with deployment check
            success, deployment = self.download_with_deployment_check(
                deviceCode, time_slot, filetype, spectrograms_per_batch, auto_select_deployment, download_flac
            )
            
            if success:
                logger.info(f"Successfully downloaded slot {i}/{total_slots}")
            else:
                logger.warning(f"Failed to download slot {i}/{total_slots}")
            
            logger.info(f"Completed slot {i}/{total_slots} in {time.time() - slot_start:.2f}s")
            logger.info(f"Overall progress: {i}/{total_slots} slots completed")
        
        logger.info("Deployment-aware download completed")

    def download_with_deployment_check(self, deviceCode, start_date_object, filetype='png', data_length_seconds=1799, auto_select_deployment=False, download_flac=False):
        """
        Download spectrograms with deployment validation.
        
        :param deviceCode: Device code
        :param start_date_object: Start date (datetime object)
        :param filetype: File type ('png' or 'mat')
        :param data_length_seconds: Length of data to download in seconds
        :param auto_select_deployment: If True, automatically select best deployment
        :param download_flac: Whether to also download corresponding FLAC audio files
        :return: Success status and deployment info
        """
        # Ensure timezone-aware datetimes
        start_date_object = ensure_timezone_aware(start_date_object)
        end_date_object = start_date_object + timedelta(seconds=data_length_seconds)
        
        print(f"\nValidating deployment coverage for {deviceCode}...")
        has_coverage, deployments = self.validate_deployment_coverage(
            deviceCode, start_date_object, end_date_object
        )
        
        if not has_coverage:
            print(f"❌ No deployment coverage for {deviceCode} from {start_date_object.strftime('%Y-%m-%d %H:%M:%S')} to {end_date_object.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Suggest alternative dates - get all deployments for this device
            all_deployments = self._get_cached_deployments()
            device_deployments = [dep for dep in all_deployments if dep.device_code == deviceCode]
            
            if device_deployments:
                print("\nAvailable deployment periods:")
                for deployment in device_deployments:
                    end_str = deployment.end_date.strftime('%Y-%m-%d') if deployment.end_date else 'ongoing'
                    print(f"  • {deployment.begin_date.strftime('%Y-%m-%d')} to {end_str} at {deployment.location_name}")
            return False, None
        
        if auto_select_deployment:
            # Use the first available deployment for now
            selected_deployment = deployments[0]
        else:
            # Interactive selection if multiple deployments
            if len(deployments) > 1:
                print(f"\nMultiple deployments found for the requested time range.")
                selected_deployment = self.interactive_deployment_selection(
                    deviceCode, start_date_object, end_date_object
                )
                if not selected_deployment:
                    print("No deployment selected. Aborting download.")
                    return False, None
            else:
                selected_deployment = deployments[0]
        
        end_str = selected_deployment.end_date.strftime('%Y-%m-%d') if selected_deployment.end_date else 'ongoing'
        print(f"✅ Using deployment: {selected_deployment.begin_date.strftime('%Y-%m-%d')} to {end_str} at {selected_deployment.location_name}")
        
        # Proceed with download
        try:
            self.download_MAT_or_PNG(deviceCode, start_date_object, filetype=filetype, spectrograms_per_batch=6, download_flac=download_flac)
            return True, selected_deployment
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False, selected_deployment

    def show_available_deployments(self, device_code, start_date, end_date, check_data_availability=True):
        """
        Show available deployments for a device within a date range.
        
        :param device_code: Device code to check deployments for
        :param start_date: Start date (datetime object)
        :param end_date: End date (datetime object)
        :param check_data_availability: Whether to check data availability
        :return: List of deployment info objects
        """
        print(f"\nChecking deployments for {device_code} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Ensure timezone-aware datetimes
        start_date = ensure_timezone_aware(start_date)
        end_date = ensure_timezone_aware(end_date)
        
        # Use cached deployments to avoid redundant API calls
        all_deployments = self._get_cached_deployments()
        device_deployments = [dep for dep in all_deployments if dep.device_code == device_code]
        
        # Filter to deployments that overlap with the date range
        overlapping_deployments = []
        for deployment in device_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            
            # Check if deployment overlaps with requested time range
            if dep_start <= end_date and dep_end >= start_date:
                overlapping_deployments.append(deployment)
        
        if check_data_availability and overlapping_deployments:
            overlapping_deployments = self.deployment_checker.check_data_availability(
                overlapping_deployments, start_date, end_date
            )
        
        if not overlapping_deployments:
            print(f"No deployments found for {device_code} in the specified date range.")
            return []
        
        print(f"\nFound {len(overlapping_deployments)} deployment(s):")
        for i, deployment in enumerate(overlapping_deployments, 1):
            print(f"  {i}. {deployment.begin_date.strftime('%Y-%m-%d')} to {deployment.end_date.strftime('%Y-%m-%d') if deployment.end_date else 'ongoing'}")
            print(f"     Location: {deployment.location_name}")
            if hasattr(deployment, 'has_data'):
                print(f"     Data Available: {deployment.has_data}")
        
        return overlapping_deployments

    def interactive_deployment_selection(self, device_code, start_date, end_date):
        """
        Interactive deployment selection for a device within a date range.
        
        :param device_code: Device code to check deployments for
        :param start_date: Start date (datetime object)
        :param end_date: End date (datetime object)
        :return: Selected deployment info object or None
        """
        from .deployment_checker import interactive_deployment_selector
        return interactive_deployment_selector(self.deployment_checker, start_date, end_date)

    def validate_deployment_coverage(self, device_code, start_date, end_date):
        """
        Validate that the requested date range has deployment coverage.
        
        :param device_code: Device code to validate
        :param start_date: Start date (datetime object)
        :param end_date: End date (datetime object)
        :return: (bool, list) - (has_coverage, list_of_deployments)
        """
        # Ensure input dates are timezone-aware
        start_date = ensure_timezone_aware(start_date)
        end_date = ensure_timezone_aware(end_date)
        
        # Use cached deployments to avoid redundant API calls
        all_deployments = self._get_cached_deployments()
        device_deployments = [dep for dep in all_deployments if dep.device_code == device_code]
        
        # Filter to deployments that overlap with the date range
        overlapping_deployments = []
        for deployment in device_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            
            # Check if deployment overlaps with requested time range
            if dep_start <= end_date and dep_end >= start_date:
                overlapping_deployments.append(deployment)
        
        if not overlapping_deployments:
            return False, []
        
        # Check if any deployment covers the entire requested range
        for deployment in overlapping_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            if dep_start <= start_date and dep_end >= end_date:
                return True, [deployment]
        
        # Check if multiple deployments together cover the range
        deployments_sorted = sorted(overlapping_deployments, key=lambda x: x.begin_date)
        coverage_start = ensure_timezone_aware(deployments_sorted[0].begin_date)
        coverage_end = ensure_timezone_aware(deployments_sorted[-1].end_date) if deployments_sorted[-1].end_date else datetime.now(timezone.utc)
        
        if coverage_start <= start_date and coverage_end >= end_date:
            # Check for gaps
            for i in range(len(deployments_sorted) - 1):
                curr_end = ensure_timezone_aware(deployments_sorted[i].end_date) if deployments_sorted[i].end_date else datetime.now(timezone.utc)
                next_start = ensure_timezone_aware(deployments_sorted[i + 1].begin_date)
                if curr_end < next_start:
                    gap_start = curr_end
                    gap_end = next_start
                    if gap_start < end_date and gap_end > start_date:
                        print(f"Warning: Gap in deployment coverage from {gap_start.strftime('%Y-%m-%d')} to {gap_end.strftime('%Y-%m-%d')}")
            return True, deployments_sorted
        
        return False, overlapping_deployments

    def download_specific_spectrograms(self, device_times_dict, filetype='png', duration_seconds=300, download_flac=False):
        """
        Downloads spectrograms for specific device IDs and timestamps.
        
        :param device_times_dict: Dictionary where keys are device IDs, and values are lists of tuples (year, month, day, hour, minute, second).
        :param filetype: File type to download ('png' or 'mat').
        :param duration_seconds: Duration of each spectrogram in seconds (default: 300 for 5 minutes).
        :param download_flac: Whether to also download corresponding FLAC audio files.
        """
        
        for device_id, times in device_times_dict.items():
            # Calculate date range for this device
            if times:
                # Get min and max dates from the time list
                dates = [datetime(t[0], t[1], t[2]) for t in times]
                start_date = min(dates)
                end_date = max(dates)
                
                start_date_tuple = (start_date.year, start_date.month, start_date.day)
                end_date_tuple = (end_date.year, end_date.month, end_date.day) if start_date.date() != end_date.date() else None
                
                # Setup directories once per device with date range
                self.setup_directories(filetype, device_id, 'specific_times', start_date_tuple, end_date_tuple, duration_seconds)

            for time_tuple in times:
                year, month, day, hour, minute, second = time_tuple
                start_date_object = datetime(year, month, day, hour, minute, second)

                # Download specific spectrogram with custom duration
                self.download_MAT_or_PNG(device_id, start_date_object, filetype=filetype, spectrograms_per_batch=6, download_flac=download_flac)

                # Process the spectrograms
                # self.process_spectrograms(filetype)

    def quick_deployment_check(self, device_code, start_date, end_date):
        """
        Quick check for deployment availability in a date range.
        
        :param device_code: Device code to check
        :param start_date: Start date (datetime object)
        :param end_date: End date (datetime object)
        :return: Boolean indicating if deployments are available
        """
        # Ensure timezone-aware datetimes
        start_date = ensure_timezone_aware(start_date)
        end_date = ensure_timezone_aware(end_date)
        
        # Use cached deployments to avoid redundant API calls
        all_deployments = self._get_cached_deployments()
        device_deployments = [dep for dep in all_deployments if dep.device_code == device_code]
        
        # Check for overlapping deployments
        for deployment in device_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            
            # Check if deployment overlaps with requested time range
            if dep_start <= end_date and dep_end >= start_date:
                return True
        
        return False

    def interactive_download_with_deployments(self, device_code, filetype='png'):
        """
        Interactive download process with deployment guidance.
        
        :param device_code: Device code
        :param filetype: File type ('png' or 'mat')
        """
        print(f"\n🎯 Interactive Hydrophone Data Download for {device_code}")
        print("=" * 60)
        
        # Get all deployments for this device (using cache to avoid redundant API calls)
        all_deployments = self._get_cached_deployments()
        device_deployments = [dep for dep in all_deployments if dep.device_code == device_code]
        
        if not device_deployments:
            print(f"❌ No deployments found for device {device_code}")
            return
        
        print(f"\nAvailable deployments for {device_code}:")
        for i, deployment in enumerate(device_deployments, 1):
            end_str = deployment.end_date.strftime('%Y-%m-%d') if deployment.end_date else 'ongoing'
            print(f"  {i}. {deployment.begin_date.strftime('%Y-%m-%d')} to {end_str}")
            print(f"     Location: {deployment.location_name}")
        
        # Get user input for date range
        try:
            start_input = input("\nEnter start date (YYYY-MM-DD): ").strip()
            end_input = input("Enter end date (YYYY-MM-DD): ").strip()
            
            # Create timezone-aware datetimes
            start_date = ensure_timezone_aware(datetime.strptime(start_input, '%Y-%m-%d'))
            end_date = ensure_timezone_aware(datetime.strptime(end_input, '%Y-%m-%d'))
            
            if start_date >= end_date:
                print("❌ Start date must be before end date")
                return
            
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD")
            return
        
        # Check deployment coverage using already fetched data
        has_coverage, deployments = self._validate_deployment_coverage_with_data(
            device_deployments, start_date, end_date
        )
        
        if not has_coverage:
            print(f"❌ No deployment coverage for the requested date range")
            print("\nWould you like to see alternative date ranges? (y/n): ", end="")
            if input().strip().lower() == 'y':
                # Show deployments within an expanded range
                expanded_start = start_date - timedelta(days=30)
                expanded_end = end_date + timedelta(days=30)
                self._show_deployments_with_data(device_deployments, expanded_start, expanded_end)
            return
        
        # Check data availability for the deployments we found
        print("Checking data availability...")
        available_deployments = self.deployment_checker.check_data_availability(
            deployments, start_date, end_date
        )
        
        if not available_deployments:
            print("❌ No data available for the deployment periods covering your date range")
            return
        
        print(f"✅ Found {len(available_deployments)} deployment(s) with available data")
        
        # Get sampling parameters
        try:
            threshold_num = int(input("\nHow many spectrograms do you want to download? "))
            if threshold_num <= 0:
                print("❌ Number of spectrograms must be positive")
                return
        except ValueError:
            print("❌ Invalid number")
            return
        
        print(f"\nProceeding with download:")
        print(f"  Device: {device_code}")
        print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"  Target: {threshold_num} spectrograms")
        print(f"  File type: {filetype}")
        
        # Convert to the format expected by download_spectrograms_with_deployment_check
        start_date_tuple = (start_date.year, start_date.month, start_date.day)
        end_date_tuple = (end_date.year, end_date.month, end_date.day)
        num_days = (end_date - start_date).days
        
        # Setup directories with date range info (using default 5-minute duration)
        self.setup_directories(filetype, device_code, 'sampling', start_date_tuple, end_date_tuple, 300)
        
        self.download_spectrograms_with_deployment_check(
            device_code, start_date_tuple, threshold_num, num_days=num_days, 
            filetype=filetype, auto_select_deployment=True
        )
    
    def _validate_deployment_coverage_with_data(self, device_deployments, start_date, end_date):
        """
        Validate deployment coverage using pre-fetched deployment data.
        
        :param device_deployments: List of deployment objects for the device
        :param start_date: Start date (timezone-aware datetime)
        :param end_date: End date (timezone-aware datetime)
        :return: (bool, list) - (has_coverage, list_of_covering_deployments)
        """
        # Filter deployments that overlap with the date range
        overlapping_deployments = []
        for deployment in device_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            
            # Check if deployment overlaps with requested time range
            if dep_start <= end_date and dep_end >= start_date:
                overlapping_deployments.append(deployment)
        
        if not overlapping_deployments:
            return False, []
        
        # Check if any deployment covers the entire requested range
        for deployment in overlapping_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            if dep_start <= start_date and dep_end >= end_date:
                return True, [deployment]
        
        # Check if multiple deployments together cover the range
        deployments_sorted = sorted(overlapping_deployments, key=lambda x: x.begin_date)
        coverage_start = ensure_timezone_aware(deployments_sorted[0].begin_date)
        coverage_end = ensure_timezone_aware(deployments_sorted[-1].end_date) if deployments_sorted[-1].end_date else datetime.now(timezone.utc)
        
        if coverage_start <= start_date and coverage_end >= end_date:
            # Check for gaps
            for i in range(len(deployments_sorted) - 1):
                curr_end = ensure_timezone_aware(deployments_sorted[i].end_date) if deployments_sorted[i].end_date else datetime.now(timezone.utc)
                next_start = ensure_timezone_aware(deployments_sorted[i + 1].begin_date)
                if curr_end < next_start:
                    gap_start = curr_end
                    gap_end = next_start
                    if gap_start < end_date and gap_end > start_date:
                        print(f"Warning: Gap in deployment coverage from {gap_start.strftime('%Y-%m-%d')} to {gap_end.strftime('%Y-%m-%d')}")
            return True, deployments_sorted
        
        return False, overlapping_deployments
    
    def _show_deployments_with_data(self, device_deployments, start_date, end_date):
        """
        Show deployments using pre-fetched data instead of making new API calls.
        """
        overlapping = []
        for deployment in device_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            
            # Check if deployment overlaps with requested time range
            if dep_start <= end_date and dep_end >= start_date:
                overlapping.append(deployment)
        
        if overlapping:
            print(f"\nDeployments overlapping with {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}:")
            for i, deployment in enumerate(overlapping, 1):
                end_str = deployment.end_date.strftime('%Y-%m-%d') if deployment.end_date else 'ongoing'
                print(f"  {i}. {deployment.begin_date.strftime('%Y-%m-%d')} to {end_str}")
                print(f"     Location: {deployment.location_name}")
        else:
            print(f"\nNo deployments found overlapping with {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    def _get_cached_deployments(self, max_age_minutes=30):
        """
        Get deployments from cache or fetch new ones if cache is stale.
        
        :param max_age_minutes: Maximum age of cache in minutes
        :return: List of all deployment objects
        """
        now = datetime.now()
        
        # Check if cache is valid
        if (self._deployment_cache is not None and 
            self._cache_timestamp is not None and 
            (now - self._cache_timestamp).total_seconds() < max_age_minutes * 60):
            return self._deployment_cache
        
        # Cache is stale or doesn't exist, fetch fresh data
        print("Fetching deployment information...")
        self._deployment_cache = self.deployment_checker.get_all_hydrophone_deployments()
        self._cache_timestamp = now
        
        return self._deployment_cache