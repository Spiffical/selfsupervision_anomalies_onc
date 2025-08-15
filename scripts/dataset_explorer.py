#!/usr/bin/env python3
"""
This script will walk you through opening, viewing, and understanding
the hydrophone dataset file step by step.

Run this with: python dataset_explorer.py your_dataset.h5
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import Counter

def explore_dataset_step_by_step(filepath):
    """
    A walkthrough of what's inside the hydrophone dataset
    """
    
    print("🌊 Hydrophone Dataset Explorer")
    print("=" * 50)
    print(f"📁 Opening file: {filepath}")
    print()
    
    # Step 1: Opening the file
    print("🔓 STEP 1: Opening the H5 file")
    
    with h5py.File(filepath, 'r') as hf:
        # Show what's inside at the top level
        keys = list(hf.keys())
        print(f"📂 Available datasets: {keys}")
        print()
        
        # Step 2: Understanding the basic structure
        print("🔍 STEP 2: Understanding what we have")
        total_samples = len(hf[keys[0]]) if keys else 0
        print(f"📊 Total audio samples in dataset: {total_samples:,}")
        print()
        
        # Step 3: Exploring spectrograms
        if 'spectrograms' in hf.keys():
            print("🎵 STEP 3: Looking at the spectrograms")
            print()
            
            spec = hf['spectrograms']
            print(f"Spectrogram shape: {spec.shape}")
            print(f"   - {spec.shape[0]:,} audio samples")
            print(f"   - {spec.shape[1]} frequency bins")
            print(f"   - {spec.shape[2]} time steps")
            print(f"   - {spec.shape[3]} channel")
            print()
            
            # Let's look at a single spectrogram
            print("Let's look at the first spectrogram...")
            first_spec = spec[0]  # Get the first one
            print(f"   Values range from {np.min(first_spec):.2f} to {np.max(first_spec):.2f}")
            print()
            
        # Step 4: Understanding where the data comes from
        if 'sources' in hf.keys():
            print("🎙️ STEP 4: Where did this audio come from?")
            print()
            
            sources = hf['sources'][:]
            
            # Show a few examples
            print("📋 Example sources:")
            for i in range(min(3, len(sources))):
                source = sources[i]
                if isinstance(source, bytes):
                    source = source.decode('utf-8')
                print(f"   {i+1}. {source}")
            print()
            
            # Extract hydrophone names
            hydrophones = []
            for source in sources:
                if isinstance(source, bytes):
                    source = source.decode('utf-8')
                
                # Look for hydrophone IDs (they usually start with IC or JASCO)
                if 'IC' in source:
                    # Find the IC code
                    import re
                    match = re.search(r'(IC[A-Z0-9]+)', source)
                    if match:
                        hydrophones.append(match.group(1))
                    else:
                        hydrophones.append('IC_Unknown')
                elif 'JASCO' in source:
                    hydrophones.append('JASCO_Hydrophone')
                else:
                    hydrophones.append('Unknown')
            
            hydro_counts = Counter(hydrophones)
            print(f"📊 We have data from {len(hydro_counts)} different hydrophones:")
            for hydro, count in hydro_counts.most_common():
                print(f"   - {hydro}: {count:,} recordings")
            print()
        
        # Step 5: Uunderstanding labels!
        if 'labels' in hf.keys() and 'label_strings' in hf.keys():
            print("🏷️ STEP 5: Understanding the labels")
            print()
            
            labels = hf['labels'][:]
            label_strings = hf['label_strings'][:]
            
            print(f"📊 We have {len(labels)} labeled audio samples")
            print()
            
            # Explain the label format
            print("🔢 About the labels:")
            print("There are TWO ways labels are stored:")
            print("1. 'labels' - these are 1-hot encoded vectors (arrays of 0s and 1s)")
            print("2. 'label_strings' - these are human-readable text descriptions")
            print()
            
            # Show the label shapes
            print(f"Label array shape: {labels.shape}")
            if len(labels.shape) > 1:
                print(f"   This means each sample has {labels.shape[1]} possible categories")
                print("   A '1' means that category is present, '0' means it's not")
            print()
            
            # Show some diverse examples
            print("Let's look at some different examples:")
            
            # Find 5 samples with different label patterns
            unique_patterns = {}
            for i in range(len(labels)):
                pattern = tuple(labels[i])
                if pattern not in unique_patterns and len(unique_patterns) < 5:
                    unique_patterns[pattern] = i
            
            # If we don't have 5 unique patterns, just take first 5
            if len(unique_patterns) < 5:
                for i in range(min(5, len(labels))):
                    pattern = tuple(labels[i])
                    unique_patterns[pattern] = i
            
            for idx, (pattern, sample_idx) in enumerate(unique_patterns.items()):
                label_vec = labels[sample_idx]
                label_str = label_strings[sample_idx]
                if isinstance(label_str, bytes):
                    label_str = label_str.decode('utf-8')
                
                print(f"   Sample {idx+1} (index {sample_idx}):")
                print(f"     1-hot vector: {label_vec}")
                print(f"     Human readable: '{label_str}'")
                print()
            
            # The anomaly types are predefined (from dataset config)
            anomaly_types = [
                "Anomaly",          # Generic anomaly marker
                "Data Gap",         # Missing or corrupted data segments  
                "Dropout",          # Signal dropouts or interruptions
                "Engine Noise",     # Ship engines and mechanical noise
                "Rain",             # Surface rain/weather effects
                "Sensitivity",      # Hydrophone sensitivity issues
                "Tonal",            # Tonal sounds (marine life, equipment)
                "Unknown Feature"   # Unidentified acoustic features
            ]
            
            print("🗺️  1-hot encoding map:")
            for pos, anomaly_type in enumerate(anomaly_types):
                print(f"   Position {pos}: {anomaly_type}")
            print()
            
            # Show label distribution
            print("📈 How common are different types of sounds?")
            label_counts = Counter()
            for label_str in label_strings:
                if isinstance(label_str, bytes):
                    label_str = label_str.decode('utf-8')
                
                if ';' in label_str:
                    for label in label_str.split(';'):
                        label_counts[label.strip()] += 1
                else:
                    label_counts[label_str] += 1
            
            print("   Top 10 most common sounds:")
            for label, count in label_counts.most_common(10):
                percentage = (count / len(labels)) * 100
                print(f"     {label}: {count:,} times ({percentage:.1f}%)")
            print()


def plot_example_spectrogram(filepath, sample_index=0):
    """
    Show what a spectrogram actually looks like visually
    """
    
    try:
        with h5py.File(filepath, 'r') as hf:
            if 'spectrograms' not in hf.keys():
                print("❌ No spectrograms found in this file")
                return
            
            spec = hf['spectrograms'][sample_index]
            
            # If it has multiple channels, just use the first one
            if len(spec.shape) == 3:
                spec = spec[:, :, 0]
            
            plt.figure(figsize=(10, 6))
            plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Amplitude')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.title(f'Spectrogram of Sample {sample_index}')
            
            # Add some context if we have labels
            if 'label_strings' in hf.keys():
                label_str = hf['label_strings'][sample_index]
                if isinstance(label_str, bytes):
                    label_str = label_str.decode('utf-8')
                plt.title(f'5-minute spectrogram of Sample {sample_index}: "{label_str}"')
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"❌ Couldn't create plot: {e}")
        print("   (You might need to install matplotlib: pip install matplotlib)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🎓 H5 dataset explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_explorer.py my_data.h5
  python dataset_explorer.py my_data.h5 --plot 5
        """
    )
    
    parser.add_argument("filepath", help="Path to your H5 dataset file")
    parser.add_argument("--plot", type=int, metavar="N", 
                       help="Also show a plot of spectrogram N (requires matplotlib)")
    
    args = parser.parse_args()
    
    try:
        explore_dataset_step_by_step(args.filepath)
        
        if args.plot is not None:
            plot_example_spectrogram(args.filepath, args.plot)
            
    except FileNotFoundError:
        print(f"❌ File not found: {args.filepath}")
        print("   Make sure the file path is correct!")
    except Exception as e:
        print(f"❌ Something went wrong: {e}")
        print("   The file might be corrupted or in a different format.") 