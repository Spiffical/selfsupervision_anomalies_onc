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

# Import configuration utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils', 'data'))
from config_utils import load_config

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
    # Load configuration
    config = load_config()
    filename = os.path.basename(mat_file)
    EXPECTED_SHAPE = tuple(config.expected_shape)
    
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
    label_vector = np.zeros(len(config.anomaly_labels), dtype=np.int8)
    label_strings = []
    
    if filename in label_data:
        for label in label_data[filename]:
            # Convert label to lowercase for comparison
            label = label.lower()
            # Normalize "unknown features" to "unknown feature"
            if label == "unknown features":
                label = "unknown feature"
            # Convert config.anomaly_labels to lowercase for comparison
            if label in [a.lower() for a in config.anomaly_labels]:
                # Find the index in the original list using case-insensitive matching
                label_idx = next(i for i, a in enumerate(config.anomaly_labels) if a.lower() == label)
                label_vector[label_idx] = 1
                # Use the original case from config.anomaly_labels for consistency
                label_strings.append(config.anomaly_labels[label_idx])
                logging.info(f"Processing file {filename} with anomaly: {config.anomaly_labels[label_idx]}")
    
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
    # Load configuration
    config = load_config()
    
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
        # Use config settings for compression and chunking
        compression_opts = {'compression': config.compression}
        if config.compression in ['gzip', 'lzf', 'szip']:
            compression_opts['compression_opts'] = config.compression_level
        
        chunk_size = tuple(config.chunk_size)
        
        hf.create_dataset('spectrograms', data=data_array, maxshape=(None,) + data_array.shape[1:], 
                         chunks=chunk_size, **compression_opts)
        hf.create_dataset('labels', data=labels_array, maxshape=(None, len(config.anomaly_labels)), 
                         chunks=True, **compression_opts)
        hf.create_dataset('sources', data=source_array, maxshape=(None,), 
                         chunks=True, **compression_opts)
        hf.create_dataset('label_strings', data=label_strings_array, maxshape=(None,), 
                         chunks=True, **compression_opts)
    else:
        hf['spectrograms'].resize((hf['spectrograms'].shape[0] + data_array.shape[0]), axis=0)
        hf['spectrograms'][-data_array.shape[0]:] = data_array

        hf['labels'].resize((hf['labels'].shape[0] + labels_array.shape[0]), axis=0)
        hf['labels'][-labels_array.shape[0]:] = labels_array

        hf['sources'].resize((hf['sources'].shape[0] + source_array.shape[0]), axis=0)
        hf['sources'][-source_array.shape[0]:] = source_array
        
        hf['label_strings'].resize((hf['label_strings'].shape[0] + label_strings_array.shape[0]), axis=0)
        hf['label_strings'][-label_strings_array.shape[0]:] = label_strings_array

def find_mat_files_and_labels(folder):
    """
    Find .mat files and labels.json files in various folder structures.
    
    Supports:
    1. Legacy structure: folder/matfiles/*.mat, folder/labels.json
    2. Legacy with Normal: folder/Normal/*.mat, folder/matfiles/*.mat
    3. New enhanced structure: folder/mat/DEVICE/METHOD_DATE_DURATION/processed/*.mat
    4. Simple flat structure: folder/*.mat, folder/labels.json
    
    Returns:
    --------
    tuple: (mat_files_list, labels_dict)
    """
    mat_files = []
    labels = {}
    
    logging.info(f"Analyzing folder structure: {folder}")
    
    # Strategy 1: Check for new enhanced structure (data/mat/DEVICE/METHOD_*/processed/*.mat)
    mat_root = os.path.join(folder, 'mat')
    if os.path.exists(mat_root):
        logging.info("  Detected enhanced structure (data/mat/...)")
        device_folders = [d for d in os.listdir(mat_root) if os.path.isdir(os.path.join(mat_root, d))]
        
        for device in device_folders:
            device_path = os.path.join(mat_root, device)
            method_folders = [d for d in os.listdir(device_path) if os.path.isdir(os.path.join(device_path, d))]
            
            for method_folder in method_folders:
                method_path = os.path.join(device_path, method_folder)
                
                # Check for processed and rejects subfolders
                for subfolder in ['processed', 'rejects']:
                    subfolder_path = os.path.join(method_path, subfolder)
                    if os.path.exists(subfolder_path):
                        mat_files_in_subfolder = glob.glob(os.path.join(subfolder_path, '*.mat'))
                        mat_files.extend(mat_files_in_subfolder)
                        logging.info(f"    Found {len(mat_files_in_subfolder)} .mat files in {subfolder_path}")
                
                # Check for labels.json in method folder
                labels_file = os.path.join(method_path, 'labels.json')
                if os.path.exists(labels_file):
                    with open(labels_file, 'r') as f:
                        method_labels = json.load(f)
                        labels.update(method_labels)
                        logging.info(f"    Loaded {len(method_labels)} labels from {labels_file}")
            
            # Check for device-level labels.json
            device_labels_file = os.path.join(device_path, 'labels.json')
            if os.path.exists(device_labels_file):
                with open(device_labels_file, 'r') as f:
                    device_labels = json.load(f)
                    labels.update(device_labels)
                    logging.info(f"    Loaded {len(device_labels)} labels from {device_labels_file}")
    
    # Strategy 2: Check for legacy structure with subdirectories
    if not mat_files:  # Only if we haven't found files yet
        logging.info("  Checking for legacy structure...")
        subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        
        # Look for folders with 'mat' in name
        mat_folders = [d for d in subdirs if 'mat' in d.lower()]
        for mat_folder in mat_folders:
            mat_folder_path = os.path.join(folder, mat_folder)
            folder_mat_files = glob.glob(os.path.join(mat_folder_path, '*.mat'))
            mat_files.extend(folder_mat_files)
            logging.info(f"    Found {len(folder_mat_files)} .mat files in {mat_folder_path}")
        
        # Check for Normal folder
        if 'Normal' in subdirs:
            normal_folder = os.path.join(folder, 'Normal')
            normal_mat_files = glob.glob(os.path.join(normal_folder, '*.mat'))
            mat_files.extend(normal_mat_files)
            logging.info(f"    Found {len(normal_mat_files)} .mat files in Normal folder")
        
        # Check for folder-level labels.json
        labels_file = os.path.join(folder, 'labels.json')
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                folder_labels = json.load(f)
                labels.update(folder_labels)
                logging.info(f"    Loaded {len(folder_labels)} labels from {labels_file}")
    
    # Strategy 3: Flat structure (folder/*.mat)
    if not mat_files:  # Only if we haven't found files yet
        logging.info("  Checking for flat structure...")
        flat_mat_files = glob.glob(os.path.join(folder, '*.mat'))
        if flat_mat_files:
            mat_files.extend(flat_mat_files)
            logging.info(f"    Found {len(flat_mat_files)} .mat files in flat structure")
            
            # Check for labels.json in same folder
            labels_file = os.path.join(folder, 'labels.json')
            if os.path.exists(labels_file):
                with open(labels_file, 'r') as f:
                    folder_labels = json.load(f)
                    labels.update(folder_labels)
                    logging.info(f"    Loaded {len(folder_labels)} labels from {labels_file}")
    
    # Strategy 4: Recursive search as fallback
    if not mat_files:
        logging.info("  Performing recursive search as fallback...")
        for root, dirs, files in os.walk(folder):
            mat_files_in_dir = [os.path.join(root, f) for f in files if f.endswith('.mat')]
            if mat_files_in_dir:
                mat_files.extend(mat_files_in_dir)
                logging.info(f"    Found {len(mat_files_in_dir)} .mat files in {root}")
            
            # Look for labels.json files
            if 'labels.json' in files:
                labels_file = os.path.join(root, 'labels.json')
                with open(labels_file, 'r') as f:
                    dir_labels = json.load(f)
                    labels.update(dir_labels)
                    logging.info(f"    Loaded {len(dir_labels)} labels from {labels_file}")
    
    return mat_files, labels

def create_or_update_h5(h5_filename, data_folders, batch_size=None, target_dim=None):
    """
    Creates or updates HDF5 file using JSON label files in data folders.
    
    Supports multiple folder structures:
    1. Enhanced: data/mat/DEVICE/METHOD_DATE_DURATION/processed/*.mat
    2. Legacy: folder/matfiles/*.mat + folder/Normal/*.mat + folder/labels.json
    3. Flat: folder/*.mat + folder/labels.json
    4. Recursive: any nested structure with .mat files and labels.json
    """
    # Load configuration and set defaults
    config = load_config()
    if batch_size is None:
        batch_size = config.batch_size
    if target_dim is None:
        target_dim = tuple(config.target_size)
    
    logging.info(f"Starting dataset creation: {h5_filename}")
    logging.info(f"Using batch_size: {batch_size}, target_dim: {target_dim}")
    os.makedirs(os.path.dirname(h5_filename), exist_ok=True)

    # Collect all mat files and labels from all folders
    all_mat_files = []
    all_labels = {}
    
    for folder in data_folders:
        logging.info(f"\nProcessing data folder: {folder}")
        mat_files, labels = find_mat_files_and_labels(folder)
        
        all_mat_files.extend(mat_files)
        all_labels.update(labels)
        
        logging.info(f"  Total from this folder: {len(mat_files)} .mat files, {len(labels)} labels")
    
    # Auto-label files as normal if they don't have labels
    normal_count = 0
    for mat_file in all_mat_files:
        filename = os.path.basename(mat_file)
        if filename not in all_labels:
            all_labels[filename] = []
            normal_count += 1
    
    logging.info(f"\nAuto-labeled {normal_count} files as normal (no existing labels)")

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
        # Save configuration metadata to H5 file (only on first creation)
        if 'anomaly_label_names' not in hf:
            config = load_config()
            config.save_metadata_to_h5(hf)
            logging.info("Saved configuration metadata to H5 file")
        
        for i in tqdm(range(0, len(all_mat_files), batch_size), desc="Processing batches"):
            batch_files = all_mat_files[i:i + batch_size]
            process_batch(batch_files, all_labels, target_dim, hf)

if __name__ == '__main__':
    # Load config to get defaults for argument parser
    config = load_config()
    
    parser = argparse.ArgumentParser(description='Create HDF5 dataset from labeled spectrograms.')
    
    parser.add_argument('--h5_filename', type=str, required=True,
                      help='Path to output HDF5 file')
    parser.add_argument('--data_folders', type=str, nargs='+', required=True,
                      help='Folders containing a "matfiles" subfolder and labels.json file')
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                      help=f'Files to process per batch (default: {config.batch_size})')
    parser.add_argument('--target_dim', type=int, nargs=2,
                      help=f'Target dimensions (height width) for reshaping (default: {config.target_size})')
    parser.add_argument('--num_workers', type=int, default=config.max_workers,
                      help='Number of worker processes to use (defaults from config or CPU cores)')

    args = parser.parse_args()
    target_dim = tuple(args.target_dim) if args.target_dim else None

    create_or_update_h5(
        args.h5_filename,
        args.data_folders,
        args.batch_size,
        target_dim
    )