import h5py
import numpy as np
import argparse
from collections import defaultdict
import os
from utilities.metrics.hydrophone_metrics import extract_hydrophone
from tqdm import tqdm

def create_test_dataset(input_path, output_path, samples_per_hydrophone=50):
    """
    Create a smaller test dataset from the input dataset by sampling a specified
    number of spectrograms from each hydrophone.
    
    Args:
        input_path: Path to the input H5 dataset
        output_path: Path to save the output H5 dataset
        samples_per_hydrophone: Number of samples to keep per hydrophone
    """
    print(f"Creating test dataset from {input_path}")
    print(f"Sampling {samples_per_hydrophone} spectrograms per hydrophone")
    
    # Read the input dataset
    with h5py.File(input_path, 'r') as f_in:
        # Get dataset sizes
        n_samples = len(f_in['spectrograms'])
        print(f"\nReading {n_samples} samples from input dataset...")
        
        # Get sources (these are small enough to load into memory)
        sources = f_in['sources'][:]
        
        # Group indices by hydrophone
        print("\nGrouping samples by hydrophone...")
        hydrophone_indices = defaultdict(list)
        for i in tqdm(range(len(sources)), desc="Processing sources"):
            hydrophone = extract_hydrophone(sources[i])
            hydrophone_indices[hydrophone].append(i)
        
        # Sample indices from each hydrophone
        print("\nSampling from each hydrophone...")
        selected_indices = []
        for hydrophone in tqdm(hydrophone_indices.keys(), desc="Sampling hydrophones"):
            indices = hydrophone_indices[hydrophone]
            n_samples = min(samples_per_hydrophone, len(indices))
            sampled_indices = np.random.choice(indices, size=n_samples, replace=False)
            selected_indices.extend(sampled_indices)
        
        selected_indices = np.sort(selected_indices)
        
        # Create the output dataset
        print("\nCreating output dataset...")
        with h5py.File(output_path, 'w') as f_out:
            # Get the shape and dtype of spectrograms
            spec_shape = f_in['spectrograms'].shape
            spec_dtype = f_in['spectrograms'].dtype
            
            # Create datasets with the same structure
            print("Creating datasets...")
            
            # Create and copy spectrograms in chunks
            print("Processing spectrograms...")
            f_out.create_dataset('spectrograms', 
                               shape=(len(selected_indices), *spec_shape[1:]),
                               dtype=spec_dtype,
                               chunks=True, 
                               compression='gzip')
            
            chunk_size = 100  # Process 100 spectrograms at a time
            for i in tqdm(range(0, len(selected_indices), chunk_size), desc="Copying spectrograms"):
                chunk_indices = selected_indices[i:i + chunk_size]
                chunk_data = f_in['spectrograms'][chunk_indices]
                f_out['spectrograms'][i:i + len(chunk_indices)] = chunk_data
            
            # Copy sources
            print("Copying sources...")
            f_out.create_dataset('sources', 
                               data=sources[selected_indices],
                               chunks=True, 
                               compression='gzip')
            
            # Copy labels if they exist
            if 'labels' in f_in:
                print("Copying labels...")
                labels = f_in['labels'][:]
                f_out.create_dataset('labels',
                                   data=labels[selected_indices],
                                   chunks=True,
                                   compression='gzip')
            
            # Copy label strings if they exist
            if 'label_strings' in f_in:
                print("Copying label strings...")
                f_out.create_dataset('label_strings',
                                   data=f_in['label_strings'][:],
                                   compression='gzip')
            
            # Copy attributes if any
            print("Copying attributes...")
            for key, value in f_in.attrs.items():
                f_out.attrs[key] = value
    
    # Print statistics
    print("\nTest dataset statistics:")
    print(f"Total samples: {len(selected_indices)}")
    
    hydrophone_counts = defaultdict(int)
    for idx in tqdm(selected_indices, desc="Calculating statistics"):
        hydrophone = extract_hydrophone(sources[idx])
        hydrophone_counts[hydrophone] += 1
    
    print("\nSamples per hydrophone:")
    for hydrophone, count in sorted(hydrophone_counts.items()):
        print(f"Hydrophone {hydrophone}: {count} samples")
    
    print(f"\nTest dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a smaller test dataset')
    parser.add_argument('input_path', type=str, help='Path to the input H5 dataset')
    parser.add_argument('output_path', type=str, help='Path to save the output H5 dataset')
    parser.add_argument('--samples_per_hydrophone', type=int, default=50,
                      help='Number of samples to keep per hydrophone')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    create_test_dataset(args.input_path, args.output_path, args.samples_per_hydrophone) 