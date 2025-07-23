#!/usr/bin/env python3
import h5py
import numpy as np
import re
import argparse

def investigate_unknown_hydrophones(filepath):
    """
    Investigate what the 'Unknown' hydrophones actually are
    """
    print(f"🔍 Investigating Unknown Hydrophones in: {filepath}")
    print("=" * 60)
    
    with h5py.File(filepath, 'r') as hf:
        if 'sources' not in hf.keys():
            print("No 'sources' key found in dataset")
            return
            
        sources = hf['sources'][:]
        
        print(f"Total sources: {len(sources)}")
        print()
        
        # Look for different hydrophone patterns
        patterns = {
            'ICLISTENHF': r'(ICLISTENHF[0-9]+)',
            'ICLISTENAF': r'(ICLISTENAF[0-9]+)',
            'ICLISTENLF': r'(ICLISTENLF[0-9]+)',
            'ICHYDROPHONE': r'(ICHYDROPHONE[0-9]+)',
            'JASCO': r'(JASCOAMARHYDROPHONE[A-Z0-9]+)',
            'NAXYS': r'(NAXYS_[A-Z0-9_]+)',
            'SONGMETER': r'(SONGMETERSM[A-Z0-9]+)',
            'IOS': r'(IOS[A-Z0-9]+)',
            'Generic_IC': r'(IC[A-Z0-9]+)'
        }
        
        matches = {pattern_name: [] for pattern_name in patterns}
        unknown_sources = []
        
        for source in sources:
            if isinstance(source, bytes):
                source = source.decode('utf-8')
            
            matched = False
            for pattern_name, pattern in patterns.items():
                match = re.search(pattern, source)
                if match:
                    matches[pattern_name].append(match.group(1))
                    matched = True
                    break
            
            if not matched:
                unknown_sources.append(source)
        
        print("🎙️ Hydrophone Type Distribution:")
        for pattern_name, found_hydrophones in matches.items():
            if found_hydrophones:
                unique_hydrophones = list(set(found_hydrophones))
                print(f"   {pattern_name}: {len(found_hydrophones)} samples, {len(unique_hydrophones)} unique hydrophones")
                if len(unique_hydrophones) <= 10:  # Show details for small numbers
                    for hydro in sorted(set(found_hydrophones)):
                        count = found_hydrophones.count(hydro)
                        print(f"     {hydro}: {count} samples")
                else:
                    print(f"     Top hydrophones:")
                    from collections import Counter
                    top_hydros = Counter(found_hydrophones).most_common(5)
                    for hydro, count in top_hydros:
                        print(f"       {hydro}: {count} samples")
        
        print()
        print(f"🔍 Truly Unknown Sources: {len(unknown_sources)}")
        if unknown_sources:
            print("   Sample unknown sources:")
            for i, source in enumerate(unknown_sources[:10]):  # Show first 10
                print(f"     {i+1}: {source}")
            if len(unknown_sources) > 10:
                print(f"     ... and {len(unknown_sources) - 10} more")
        
        print()
        print("🔍 All Unique Source Prefixes:")
        prefixes = set()
        for source in sources:
            if isinstance(source, bytes):
                source = source.decode('utf-8')
            # Extract everything before the first underscore or number
            prefix_match = re.search(r'^([A-Z]+)', source)
            if prefix_match:
                prefixes.add(prefix_match.group(1))
        
        for prefix in sorted(prefixes):
            print(f"   {prefix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Investigate unknown hydrophones")
    parser.add_argument("filepath", help="Path to HDF5 dataset file")
    
    args = parser.parse_args()
    investigate_unknown_hydrophones(args.filepath) 