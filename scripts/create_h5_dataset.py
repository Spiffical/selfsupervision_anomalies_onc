import h5py
import numpy as np
import os
import sys
import scipy
import scipy.io as sio
import cv2 # type: ignore
from tqdm import tqdm
import argparse
import glob
import json
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.model_selection import train_test_split
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.ssamba.utils.data import defaults # type: ignore

# Add after other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def reshape_data(data_list, dim, verbose=False):
    """
    Resamples data to be of size dim.

    Parameters
    ----------
    data_list: list of np.array
        List of data arrays to be reshaped.
    dim: tuple of int
        Target dimensions (height, width).
    verbose: bool, optional
        If True, shows a progress bar.

    Returns
    -------
    np.array
        Reshaped data array.
    """
    reshaped_data = np.zeros((len(data_list), dim[0], dim[1], 1), dtype='float32')
    for i, data in enumerate(tqdm(data_list, disable=not verbose)):
        reshaped_data[i, ..., 0] = cv2.resize(data.squeeze(), dim)
    return reshaped_data

def process_single_file(mat_file, label_data, target_dim):
    """
    Process a single .mat file and return its data.
    
    Parameters
    ----------
    mat_file : str
        Path to .mat file
    label_data : dict
        Dictionary mapping filenames to lists of labels
    target_dim : tuple or None
        Target dimensions for reshaping
        
    Returns
    -------
    tuple or None
        (data, label_vector, source, label_str) if successful, None if file is empty or invalid
    """
    filename = os.path.basename(mat_file)
    EXPECTED_SHAPE = (854, 1000)
    
    try:
        mat_data = sio.loadmat(mat_file)
    except (scipy.io.matlab._miobase.MatReadError, ValueError) as e:
        logging.warning(f"Skipping empty or invalid MAT file: {filename}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error processing {filename}: {str(e)}")
        return None
    
    if 'SpectData' in mat_data:
        data = mat_data['SpectData']['PSD'][0, 0]
        
        # Check if spectrogram is shortened
        if data.shape[1] < EXPECTED_SHAPE[1]:
            logging.warning(f"{filename} has shape {data.shape}, padding to {EXPECTED_SHAPE}")
            padding_width = ((0, 0), (0, EXPECTED_SHAPE[1] - data.shape[1]))
            data = np.pad(data, padding_width, mode='constant', constant_values=0)
        
        # Create mask for valid data (non-inf)
        valid_mask = (data != -np.inf)
        
        # Replace -inf with zeros
        data[~valid_mask] = 0
        
        # Replace NaNs with zeros
        data = np.nan_to_num(data, nan=0.0)
    else:
        logging.warning(f"Skipping {filename}: No 'SpectData' key found")
        return None
    
    if target_dim:
        data = cv2.resize(data.squeeze(), target_dim)[..., np.newaxis]
    else:
        data = data[..., np.newaxis]
    
    # Create binary label vector and store string labels
    label_vector = np.zeros(len(defaults.anomalies), dtype=np.int8)
    label_strings = []
    
    if filename in label_data:
        for label in label_data[filename]:
            # Convert label to lowercase for comparison
            label = label.lower()
            # Normalize "unknown features" to "unknown feature"
            if label == "unknown features":
                label = "unknown feature"
            # Convert defaults.anomalies to lowercase for comparison
            if label in [a.lower() for a in defaults.anomalies]:
                # Find the index in the original list using case-insensitive matching
                label_idx = next(i for i, a in enumerate(defaults.anomalies) if a.lower() == label)
                label_vector[label_idx] = 1
                # Use the original case from defaults.anomalies for consistency
                label_strings.append(defaults.anomalies[label_idx])
                logging.info(f"Processing file {filename} with anomaly: {defaults.anomalies[label_idx]}")
    
    if not label_strings:
        label_strings.append('normal')
        logging.info(f"Processing normal file: {filename}")
        
    # Join multiple labels with semicolon
    label_str = ';'.join(label_strings)
    
    return data, label_vector, filename.encode('utf-8'), label_str.encode('utf-8')

def process_batch(mat_files, label_data, target_dim, hf, num_workers=None):
    """
    Process a batch of .mat files and save to HDF5 using parallel processing.
    
    Parameters
    ----------
    mat_files : list
        List of paths to .mat files
    label_data : dict
        Dictionary mapping filenames to lists of labels
    target_dim : tuple or None
        Target dimensions for reshaping
    hf : h5py.File
        Open HDF5 file
    num_workers : int, optional
        Number of worker processes to use
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    # Create a partial function with fixed arguments
    process_func = partial(process_single_file, label_data=label_data, target_dim=target_dim)
    
    # Process files in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, mat_files),
            total=len(mat_files),
            desc="Processing files in batch"
        ))
    
    # Filter out None results (failed processing)
    results = [r for r in results if r is not None]
    
    if not results:
        logging.warning("No valid results in this batch to save")
        return
    
    # Unzip results
    data_list, labels_matrix, source_list, label_strings_list = zip(*results)
    
    # Convert to arrays
    data_array = np.array(data_list, dtype='float32')
    labels_array = np.array(labels_matrix, dtype=np.int8)
    source_array = np.array(source_list, dtype='S100')
    label_strings_array = np.array(label_strings_list, dtype='S100')

    # Save to HDF5 file
    if 'spectrograms' not in hf:
        hf.create_dataset('spectrograms', data=data_array, maxshape=(None,) + data_array.shape[1:], chunks=True, compression='gzip')
        hf.create_dataset('labels', data=labels_array, maxshape=(None, len(defaults.anomalies)), chunks=True, compression='gzip')
        hf.create_dataset('sources', data=source_array, maxshape=(None,), chunks=True, compression='gzip')
        hf.create_dataset('label_strings', data=label_strings_array, maxshape=(None,), chunks=True, compression='gzip')
    else:
        hf['spectrograms'].resize((hf['spectrograms'].shape[0] + data_array.shape[0]), axis=0)
        hf['spectrograms'][-data_array.shape[0]:] = data_array

        hf['labels'].resize((hf['labels'].shape[0] + labels_array.shape[0]), axis=0)
        hf['labels'][-labels_array.shape[0]:] = labels_array

        hf['sources'].resize((hf['sources'].shape[0] + source_array.shape[0]), axis=0)
        hf['sources'][-source_array.shape[0]:] = source_array
        
        hf['label_strings'].resize((hf['label_strings'].shape[0] + label_strings_array.shape[0]), axis=0)
        hf['label_strings'][-label_strings_array.shape[0]:] = label_strings_array

def create_or_update_h5(h5_filename, data_folders, batch_size=10, target_dim=None):
    """
    Creates or updates HDF5 file using JSON label files in data folders.
    """
    logging.info(f"Starting dataset creation: {h5_filename}")
    os.makedirs(os.path.dirname(h5_filename), exist_ok=True)

    # First, collect all labels from JSON files
    all_labels = {}
    for folder in data_folders:
        json_file = os.path.join(folder, 'labels.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                folder_labels = json.load(f)
                all_labels.update(folder_labels)
                logging.info(f"Loaded {len(folder_labels)} labels from {json_file}")

    # Collect all mat files
    all_mat_files = []
    for folder in data_folders:
        logging.info(f"Processing folder: {folder}")
        
        # Get immediate subdirectories
        subdirs = next(os.walk(folder))[1]
        # Filter for mat-containing folder names
        mat_folders = [d for d in subdirs if 'mat' in d.lower()]
        
        for mat_folder in mat_folders:
            mat_folder_path = os.path.join(folder, mat_folder)
            mat_files = glob.glob(os.path.join(mat_folder_path, '*.mat'))
            all_mat_files.extend(mat_files)
            logging.info(f"Found {len(mat_files)} .mat files in {mat_folder_path}")
        
        # Check for Normal folder
        if 'Normal' in subdirs:
            normal_folder = os.path.join(folder, 'Normal')
            normal_mat_files = glob.glob(os.path.join(normal_folder, '*.mat'))
            all_mat_files.extend(normal_mat_files)
            normal_count = 0
            for mat_file in normal_mat_files:
                filename = os.path.basename(mat_file)
                if filename not in all_labels:
                    all_labels[filename] = []
                    normal_count += 1
            logging.info(f"Found {len(normal_mat_files)} files in Normal folder, {normal_count} new normal files")

    # Log summary statistics
    anomaly_counts = {}
    normal_count = 0
    for filename, labels in all_labels.items():
        if not labels:
            normal_count += 1
        else:
            for label in labels:
                if label == "unknown features":
                    label = "unknown feature"
                anomaly_counts[label] = anomaly_counts.get(label, 0) + 1

    logging.info(f"\nDataset Summary:")
    logging.info(f"Total files: {len(all_mat_files)}")
    logging.info(f"Normal files: {normal_count}")
    logging.info("Anomaly distribution:")
    for anomaly, count in anomaly_counts.items():
        logging.info(f"  - {anomaly}: {count}")

    # Process all files in batches into a single 'data' group
    with h5py.File(h5_filename, 'a') as hf:
        for i in tqdm(range(0, len(all_mat_files), batch_size), desc="Processing batches"):
            batch_files = all_mat_files[i:i + batch_size]
            process_batch(batch_files, all_labels, target_dim, hf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create HDF5 dataset from labeled spectrograms.')
    
    parser.add_argument('--h5_filename', type=str, required=True,
                      help='Path to output HDF5 file')
    parser.add_argument('--data_folders', type=str, nargs='+', required=True,
                      help='Folders containing a "matfiles" subfolder and labels.json file')
    parser.add_argument('--batch_size', type=int, default=10,
                      help='Files to process per batch')
    parser.add_argument('--target_dim', type=int, nargs=2,
                      help='Target dimensions (height width) for reshaping')
    parser.add_argument('--num_workers', type=int, default=None,
                      help='Number of worker processes to use (defaults to number of CPU cores)')

    args = parser.parse_args()
    target_dim = tuple(args.target_dim) if args.target_dim else None

    create_or_update_h5(
        args.h5_filename,
        args.data_folders,
        args.batch_size,
        target_dim
    )
