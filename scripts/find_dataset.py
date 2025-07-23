#!/usr/bin/env python3
import os
import glob

def find_datasets():
    """Find HDF5 files in the workspace"""
    print("Looking for HDF5 dataset files...")
    
    # Common locations and patterns
    patterns = [
        "*.h5",
        "*.hdf5",
        "**/*.h5",
        "**/*.hdf5",
        "data/*.h5",
        "data/*.hdf5",
        "datasets/*.h5",
        "datasets/*.hdf5"
    ]
    
    found_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        found_files.extend(files)
    
    # Remove duplicates and sort
    found_files = sorted(list(set(found_files)))
    
    if found_files:
        print(f"Found {len(found_files)} HDF5 files:")
        for f in found_files:
            size = os.path.getsize(f) / (1024*1024)  # MB
            print(f"  {f} ({size:.1f} MB)")
    else:
        print("No HDF5 files found in workspace")
        print("Try running from the correct directory or check file extensions")
    
    return found_files

if __name__ == "__main__":
    find_datasets() 