#!/usr/bin/env python3
"""
Utility script to convert between legacy and hierarchical label formats.
Useful for migrating existing label files.
"""

import argparse
import json
import os
import sys

# add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_operations import load_labels, save_labels
from hierarchical_labels import (
    is_legacy_format, 
    convert_legacy_to_hierarchical, 
    convert_hierarchical_to_legacy,
    LEGACY_LABEL_MAPPING
)

def convert_file(input_file, output_file, to_format='hierarchical'):
    """Convert a label file between formats"""
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return False
    
    # load without automatic conversion
    data = load_labels(input_file, convert_to_hierarchical=False)
    
    if not data:
        print(f"No data found in {input_file}")
        return False
    
    # detect current format
    is_legacy = is_legacy_format(data)
    current_format = "legacy" if is_legacy else "hierarchical"
    
    print(f"Detected format: {current_format}")
    print(f"Converting to: {to_format}")
    
    if current_format == to_format:
        print(f"File is already in {to_format} format, copying...")
        converted_data = data
    elif to_format == 'hierarchical':
        # convert legacy to hierarchical
        converted_data = {}
        for filename, labels in data.items():
            converted_data[filename] = convert_legacy_to_hierarchical(labels)
        print(f"Converted {len(data)} files from legacy to hierarchical format")
    elif to_format == 'legacy':
        # convert hierarchical to legacy
        converted_data = {}
        for filename, labels in data.items():
            converted_data[filename] = convert_hierarchical_to_legacy(labels)
        print(f"Converted {len(data)} files from hierarchical to legacy format")
    else:
        print(f"Unknown target format: {to_format}")
        return False
    
    # save the converted data
    # write directly to avoid the save_labels complexity
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=4, sort_keys=True)
    
    print(f"Saved converted labels to {output_file}")
    return True

def show_mapping():
    """Show the mapping between legacy and hierarchical labels"""
    print("Legacy to Hierarchical Mapping:")
    print("=" * 50)
    for legacy, hierarchical in LEGACY_LABEL_MAPPING.items():
        print(f"{legacy:20} -> {hierarchical}")

def analyze_file(input_file):
    """Analyze a label file and show statistics"""
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        return
    
    data = load_labels(input_file, convert_to_hierarchical=False)
    is_legacy = is_legacy_format(data)
    
    print(f"File: {input_file}")
    print(f"Format: {'legacy' if is_legacy else 'hierarchical'}")
    print(f"Total files: {len(data)}")
    
    # collect all unique labels
    all_labels = set()
    for labels in data.values():
        all_labels.update(labels)
    
    print(f"Unique labels: {len(all_labels)}")
    print("\nAll labels:")
    for label in sorted(all_labels):
        print(f"  - {label}")

def main():
    parser = argparse.ArgumentParser(description='Convert between legacy and hierarchical label formats')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # convert command
    convert_parser = subparsers.add_parser('convert', help='Convert label file format')
    convert_parser.add_argument('input_file', help='Input JSON file')
    convert_parser.add_argument('output_file', help='Output JSON file')
    convert_parser.add_argument('--to', choices=['legacy', 'hierarchical'], 
                               default='hierarchical', help='Target format')
    
    # analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze label file')
    analyze_parser.add_argument('input_file', help='Input JSON file')
    
    # mapping command
    subparsers.add_parser('mapping', help='Show label mapping')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        convert_file(args.input_file, args.output_file, args.to)
    elif args.command == 'analyze':
        analyze_file(args.input_file)
    elif args.command == 'mapping':
        show_mapping()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()