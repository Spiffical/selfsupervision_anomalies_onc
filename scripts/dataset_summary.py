#!/usr/bin/env python3
import h5py
import numpy as np
from collections import Counter
import json
import argparse
from datetime import datetime
import re

def summarize_dataset(filepath):
    """
    Concise but detailed analysis of the HDF5 dataset
    """
    print(f"📊 Dataset Analysis: {filepath}")
    print("=" * 60)
    
    with h5py.File(filepath, 'r') as hf:
        print(f"🔑 Keys: {list(hf.keys())}")
        print(f"📈 Total samples: {len(hf[list(hf.keys())[0]]):,}")
        print()
        
        # Spectrograms (process in batches to avoid memory issues)
        if 'spectrograms' in hf.keys():
            spec = hf['spectrograms']
            print(f"🎵 Spectrograms:")
            print(f"   Shape: {spec.shape} (samples, freq, time, channels)")
            
            # Process in batches to avoid memory issues
            batch_size = 100
            n_samples = spec.shape[0]
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            print(f"   Processing {n_batches} batches of {batch_size} samples...")
            
            # Track statistics
            running_min = float('inf')
            running_max = float('-inf')
            running_sum = 0.0
            running_sum_sq = 0.0
            total_elements = 0
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch = spec[i:end_idx]
                
                # Update statistics
                batch_min = np.min(batch)
                batch_max = np.max(batch)
                batch_sum = np.sum(batch)
                batch_sum_sq = np.sum(batch ** 2)
                batch_elements = batch.size
                
                running_min = min(running_min, batch_min)
                running_max = max(running_max, batch_max)
                running_sum += batch_sum
                running_sum_sq += batch_sum_sq
                total_elements += batch_elements
            
            # Calculate final statistics
            mean = running_sum / total_elements
            variance = (running_sum_sq / total_elements) - (mean ** 2)
            std = np.sqrt(variance)
            
            print(f"   Data range: [{running_min:.2f}, {running_max:.2f}]")
            print(f"   Mean: {mean:.2f}, Std: {std:.2f}")
            print()
        
        # Extract hydrophone info from sources
        if 'sources' in hf.keys():
            sources = hf['sources'][:]
            hydrophones = []
            timestamps = []
            
            for source in sources:
                if isinstance(source, bytes):
                    source = source.decode('utf-8')
                
                # Extract hydrophone ID - handle both IC and JASCO types
                hydro_match = re.search(r'(IC[A-Z0-9]+)', source)
                if hydro_match:
                    hydrophones.append(hydro_match.group(1))
                else:
                    # Try JASCO pattern
                    jasco_match = re.search(r'(JASCOAMARHYDROPHONE[A-Z0-9]+)', source)
                    if jasco_match:
                        hydrophones.append(jasco_match.group(1))
                    else:
                        hydrophones.append('Unknown')
                
                # Extract timestamp
                timestamp_match = re.search(r'(\d{8}T\d{6}\.\d{3}Z)', source)
                timestamps.append(timestamp_match.group(1) if timestamp_match else 'Unknown')
            
            # Count hydrophones
            hydro_counts = Counter(hydrophones)
            print(f"🎙️ Hydrophones ({len(hydro_counts)} unique):")
            for hydro, count in sorted(hydro_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {hydro}: {count:,} samples")
            print()
            
            # Temporal analysis
            if timestamps and timestamps[0] != 'Unknown':
                try:
                    parsed_timestamps = []
                    for ts in timestamps:
                        if ts != 'Unknown':
                            dt = datetime.strptime(ts, '%Y%m%dT%H%M%S.%fZ')
                            parsed_timestamps.append(dt)
                    
                    if parsed_timestamps:
                        earliest = min(parsed_timestamps)
                        latest = max(parsed_timestamps)
                        print(f"📅 Time Range:")
                        print(f"   From: {earliest.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   To: {latest.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Duration: {(latest - earliest).days} days")
                        print(f"   Unique timestamps: {len(set(parsed_timestamps)):,}")
                        print()
                except Exception as e:
                    print(f"   Error parsing timestamps: {e}")
        
        # Labels
        if 'labels' in hf.keys() and 'label_strings' in hf.keys():
            labels = hf['labels'][:]
            label_strings = hf['label_strings'][:]
            
            # Process labels
            processed_labels = []
            for s in label_strings:
                if isinstance(s, bytes):
                    decoded = s.decode('utf-8')
                    if ';' in decoded:
                        processed_labels.append(decoded.split(';'))
                    else:
                        processed_labels.append([decoded])
                else:
                    processed_labels.append([str(s)])
            
            # Count combinations
            combo_counts = Counter(tuple(sorted(lbls)) for lbls in processed_labels)
            individual_counts = Counter()
            for label_list in processed_labels:
                for label in label_list:
                    individual_counts[label] += 1
            
            print(f"🏷️ Labels ({len(combo_counts)} combinations):")
            print("   Top combinations:")
            for combo, count in combo_counts.most_common(10):
                combo_str = " + ".join(combo) if combo != ('normal',) else "Normal"
                print(f"     {combo_str}: {count:,} ({count/len(labels)*100:.1f}%)")
            
            print("   Individual frequencies:")
            for label, count in individual_counts.most_common():
                print(f"     {label}: {count:,} ({count/len(labels)*100:.1f}%)")
            print()
        
        # Dataset statistics by hydrophone
        if 'sources' in hf.keys() and 'labels' in hf.keys():
            print(f"📊 Stats by Hydrophone:")
            
            # Group by hydrophone
            hydro_stats = {}
            for i, source in enumerate(sources):
                if isinstance(source, bytes):
                    source = source.decode('utf-8')
                
                hydro_match = re.search(r'(IC[A-Z0-9]+)', source)
                if hydro_match:
                    hydro = hydro_match.group(1)
                else:
                    jasco_match = re.search(r'(JASCOAMARHYDROPHONE[A-Z0-9]+)', source)
                    hydro = jasco_match.group(1) if jasco_match else 'Unknown'
                
                if hydro not in hydro_stats:
                    hydro_stats[hydro] = {'total': 0, 'normal': 0, 'anomalous': 0}
                
                hydro_stats[hydro]['total'] += 1
                
                # Check if normal (assuming normal is when all label bits are 0 or only 'normal' string)
                label_str = label_strings[i].decode('utf-8') if isinstance(label_strings[i], bytes) else str(label_strings[i])
                if label_str == 'normal' or not label_str:
                    hydro_stats[hydro]['normal'] += 1
                else:
                    hydro_stats[hydro]['anomalous'] += 1
            
            # Sort by total samples
            sorted_hydros = sorted(hydro_stats.items(), key=lambda x: x[1]['total'], reverse=True)
            
            for hydro, stats in sorted_hydros:
                normal_pct = (stats['normal'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(f"   {hydro}: {stats['total']:,} total, {stats['normal']:,} normal ({normal_pct:.1f}%), {stats['anomalous']:,} anomalous")
            
            print()
            
        # Get file size from the file path
        import os
        file_size = os.path.getsize(filepath)
        print(f"💾 Dataset file size: {file_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concise dataset analysis")
    parser.add_argument("filepath", help="Path to HDF5 dataset file")
    
    args = parser.parse_args()
    summarize_dataset(args.filepath) 