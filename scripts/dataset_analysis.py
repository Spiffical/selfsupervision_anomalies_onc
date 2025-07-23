#!/usr/bin/env python3
import h5py
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import json
import argparse
from datetime import datetime
import re

def analyze_dataset(filepath, save_json=True):
    """
    Comprehensive analysis of the HDF5 dataset
    Returns detailed info about hydrophones, dates, spectrograms, etc.
    """
    print(f"Analyzing dataset: {filepath}")
    print("=" * 80)
    
    results = {
        "file_info": {},
        "data_structure": {},
        "hydrophone_analysis": {},
        "temporal_analysis": {},
        "spectrogram_analysis": {},
        "label_analysis": {},
        "summary": {}
    }
    
    with h5py.File(filepath, 'r') as hf:
        # Basic file info
        results["file_info"] = {
            "filename": filepath,
            "total_keys": len(hf.keys()),
            "keys": list(hf.keys())
        }
        
        print(f"📁 File: {filepath}")
        print(f"🔑 Available keys: {list(hf.keys())}")
        print()
        
        # Analyze each dataset
        for key in hf.keys():
            dataset = hf[key]
            results["data_structure"][key] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "size": int(dataset.size)
            }
            
            print(f"📊 {key}:")
            print(f"   Shape: {dataset.shape}")
            print(f"   Type: {dataset.dtype}")
            print(f"   Size: {dataset.size:,} elements")
            
            # Sample a few values for inspection
            if dataset.size > 0:
                if dataset.size < 20:
                    print(f"   Values: {dataset[:]}")
                else:
                    print(f"   Sample: {dataset[:3]}")
            print()
        
        # Spectrogram analysis (batched to avoid memory issues)
        if 'spectrograms' in hf.keys():
            spectrograms = hf['spectrograms']
            
            # Process in batches for statistics
            batch_size = 100
            n_samples = spectrograms.shape[0]
            running_min = float('inf')
            running_max = float('-inf')
            running_sum = 0.0
            running_sum_sq = 0.0
            total_elements = 0
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch = spectrograms[i:end_idx]
                
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
            
            mean = running_sum / total_elements
            variance = (running_sum_sq / total_elements) - (mean ** 2)
            std = np.sqrt(variance)
            
            results["spectrogram_analysis"] = {
                "total_spectrograms": int(spectrograms.shape[0]),
                "dimensions": {
                    "frequency_bins": int(spectrograms.shape[1]) if len(spectrograms.shape) > 1 else None,
                    "time_bins": int(spectrograms.shape[2]) if len(spectrograms.shape) > 2 else None,
                    "channels": int(spectrograms.shape[3]) if len(spectrograms.shape) > 3 else None
                },
                "data_range": {
                    "min": float(running_min),
                    "max": float(running_max),
                    "mean": float(mean),
                    "std": float(std)
                }
            }
            
            print(f"🎵 Spectrogram Analysis:")
            print(f"   Total spectrograms: {spectrograms.shape[0]:,}")
            if len(spectrograms.shape) > 1:
                print(f"   Frequency bins: {spectrograms.shape[1]}")
            if len(spectrograms.shape) > 2:
                print(f"   Time bins: {spectrograms.shape[2]}")
            if len(spectrograms.shape) > 3:
                print(f"   Channels: {spectrograms.shape[3]}")
            print(f"   Data range: [{running_min:.3f}, {running_max:.3f}]")
            print(f"   Mean: {mean:.3f}, Std: {std:.3f}")
            print()
        
        # Extract hydrophone info from sources
        if 'sources' in hf.keys():
            sources = hf['sources'][:]
            hydrophone_data = []
            timestamp_data = []
            
            for source in sources:
                if isinstance(source, bytes):
                    source = source.decode('utf-8')
                
                # Extract hydrophone ID (e.g., ICLISTENHF1951 or JASCOAMARHYDROPHONE2402)
                hydro_match = re.search(r'(IC[A-Z0-9]+)', source)
                if hydro_match:
                    hydrophone_data.append(hydro_match.group(1))
                else:
                    # Try JASCO pattern
                    jasco_match = re.search(r'(JASCOAMARHYDROPHONE[A-Z0-9]+)', source)
                    if jasco_match:
                        hydrophone_data.append(jasco_match.group(1))
                    else:
                        hydrophone_data.append('Unknown')
                
                # Extract timestamp (e.g., 20240902T173038.996Z)
                timestamp_match = re.search(r'(\d{8}T\d{6}\.\d{3}Z)', source)
                if timestamp_match:
                    timestamp_data.append(timestamp_match.group(1))
                else:
                    timestamp_data.append('Unknown')
            
            # Count hydrophones
            hydro_counts = Counter(hydrophone_data)
            results["hydrophone_analysis"] = {
                "unique_count": len(hydro_counts),
                "unique_values": list(hydro_counts.keys()),
                "counts": dict(hydro_counts)
            }
            
            # Analyze timestamps
            if timestamp_data and timestamp_data[0] != 'Unknown':
                try:
                    # Convert timestamps to datetime objects
                    parsed_timestamps = []
                    for ts in timestamp_data:
                        if ts != 'Unknown':
                            # Parse format: 20240902T173038.996Z
                            dt = datetime.strptime(ts, '%Y%m%dT%H%M%S.%fZ')
                            parsed_timestamps.append(dt)
                    
                    if parsed_timestamps:
                        results["temporal_analysis"] = {
                            "total_entries": len(parsed_timestamps),
                            "date_range": {
                                "earliest": min(parsed_timestamps).strftime('%Y-%m-%d %H:%M:%S'),
                                "latest": max(parsed_timestamps).strftime('%Y-%m-%d %H:%M:%S')
                            },
                            "unique_timestamps": len(set(parsed_timestamps))
                        }
                except Exception as e:
                    print(f"   Error parsing timestamps: {e}")
            
            print("🎙️ Hydrophone Analysis:")
            print(f"   Unique hydrophones: {len(hydro_counts)}")
            print(f"   Hydrophone counts:")
            for hydro, count in sorted(hydro_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"     {hydro}: {count:,} samples")
            print()
        
        # Temporal analysis display
        if 'temporal_analysis' in results:
            print("📅 Temporal Analysis:")
            info = results['temporal_analysis']
            print(f"   Total entries: {info['total_entries']:,}")
            if 'date_range' in info:
                print(f"   Date range: {info['date_range']['earliest']} to {info['date_range']['latest']}")
            if 'unique_timestamps' in info:
                print(f"   Unique timestamps: {info['unique_timestamps']:,}")
            print()
        
        # Label analysis
        if 'labels' in hf.keys() and 'label_strings' in hf.keys():
            labels = hf['labels'][:]
            label_strings = hf['label_strings'][:]
            
            # Process label strings
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
            
            # Count combinations and individual labels
            combo_counts = Counter(tuple(sorted(labels)) for labels in processed_labels)
            individual_counts = Counter()
            for label_list in processed_labels:
                for label in label_list:
                    individual_counts[label] += 1
            
            results["label_analysis"] = {
                "total_samples": len(labels),
                "unique_combinations": len(combo_counts),
                "label_combinations": {str(k): int(v) for k, v in combo_counts.most_common()},
                "individual_labels": {str(k): int(v) for k, v in individual_counts.most_common()}
            }
            
            print("🏷️ Label Analysis:")
            print(f"   Total samples: {len(labels):,}")
            print(f"   Unique label combinations: {len(combo_counts)}")
            print(f"   Most common combinations:")
            for combo, count in combo_counts.most_common(10):
                combo_str = " + ".join(combo) if combo != ('normal',) else "Normal"
                print(f"     {combo_str}: {count:,} samples ({count/len(labels)*100:.1f}%)")
            
            print(f"   Individual label frequencies:")
            for label, count in individual_counts.most_common():
                print(f"     {label}: {count:,} samples ({count/len(labels)*100:.1f}%)")
            print()
        
        # Summary
        total_samples = len(hf[list(hf.keys())[0]]) if hf.keys() else 0
        results["summary"] = {
            "total_samples": total_samples,
            "unique_hydrophones": results.get("hydrophone_analysis", {}).get("unique_count", 0),
            "time_span": results.get("temporal_analysis", {}).get("date_range", "Unknown"),
            "spectrogram_dimensions": results.get("spectrogram_analysis", {}).get("dimensions", "Unknown"),
            "label_diversity": results.get("label_analysis", {}).get("unique_combinations", 0)
        }
        
        print("📋 Summary:")
        print(f"   Total samples: {total_samples:,}")
        if 'hydrophone_analysis' in results:
            hydro_count = results['hydrophone_analysis']['unique_count']
            print(f"   Unique hydrophones: {hydro_count}")
        if 'temporal_analysis' in results and 'date_range' in results['temporal_analysis']:
            time_range = results['temporal_analysis']['date_range']
            print(f"   Time span: {time_range['earliest']} to {time_range['latest']}")
        if 'spectrogram_analysis' in results:
            dims = results['spectrogram_analysis']['dimensions']
            print(f"   Spectrogram dimensions: {dims}")
        if 'label_analysis' in results:
            print(f"   Label diversity: {results['label_analysis']['unique_combinations']} combinations")
    
    # Save results
    if save_json:
        output_file = 'dataset_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive dataset analysis")
    parser.add_argument("filepath", help="Path to HDF5 dataset file")
    parser.add_argument("--no-json", action="store_true", help="Don't save JSON output")
    
    args = parser.parse_args()
    analyze_dataset(args.filepath, save_json=not args.no_json) 