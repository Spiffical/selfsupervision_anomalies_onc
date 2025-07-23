#!/usr/bin/env python3
import h5py
import numpy as np
import argparse

def explore_h5_file(filepath):
    """Quick exploration of HDF5 file structure"""
    print(f"Exploring HDF5 file: {filepath}")
    print("=" * 60)
    
    with h5py.File(filepath, 'r') as hf:
        print("\nTop-level keys:")
        for key in hf.keys():
            dataset = hf[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            
            # Show first few values for smaller datasets
            if dataset.size < 100:
                print(f"    Sample values: {dataset[:5] if len(dataset) > 5 else dataset[:]}")
            elif key == 'label_strings':
                print(f"    Sample values: {dataset[:3]}")
            elif hasattr(dataset, 'shape') and len(dataset.shape) > 0:
                print(f"    Sample values: {dataset[:3]}")
        
        print("\nDetailed exploration:")
        print("-" * 40)
        
        # Look for hydrophone/location info
        for key in hf.keys():
            print(f"\n{key}:")
            dataset = hf[key]
            if 'hydro' in key.lower() or 'location' in key.lower() or 'station' in key.lower():
                print(f"  This might contain hydrophone info!")
                if dataset.size < 50:
                    print(f"  All values: {dataset[:]}")
                else:
                    print(f"  First 10 values: {dataset[:10]}")
            elif 'time' in key.lower() or 'date' in key.lower():
                print(f"  This might contain temporal info!")
                if dataset.size < 50:
                    print(f"  All values: {dataset[:]}")
                else:
                    print(f"  First 10 values: {dataset[:10]}")
            else:
                print(f"  Shape: {dataset.shape}, dtype: {dataset.dtype}")
                if dataset.size < 20:
                    print(f"  Values: {dataset[:]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore HDF5 dataset structure")
    parser.add_argument("filepath", help="Path to HDF5 file")
    args = parser.parse_args()
    
    explore_h5_file(args.filepath) 