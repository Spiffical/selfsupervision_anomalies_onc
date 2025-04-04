import h5py
import numpy as np
from collections import Counter
import json
import argparse

def analyze_label_distributions(data_path, exclude_labels=None):
    with h5py.File(data_path, 'r') as hf:
        # Print available keys in the HDF5 file
        print("\n=== HDF5 File Structure ===")
        print("Available keys:", list(hf.keys()))
        
        labels = hf['labels'][:]
        raw_label_strings = hf['label_strings'][:]
        
        print("\n=== Raw Label Strings Sample ===")
        print("First few raw label strings:", raw_label_strings[:5])
        print("Label strings dtype:", raw_label_strings.dtype)
        
        # Process raw label strings for each sample
        sample_labels = []
        for s in raw_label_strings:
            if isinstance(s, bytes):
                decoded = s.decode('utf-8')
                # Split if it contains semicolons (multiple labels)
                if ';' in decoded:
                    sample_labels.append(decoded.split(';'))
                else:
                    sample_labels.append([decoded])
            else:
                sample_labels.append([str(s)])
        
        # Get unique label types (flattening the list of lists)
        unique_labels = sorted(set(label for labels in sample_labels for label in labels))
        if 'normal' in unique_labels:
            unique_labels.remove('normal')
        
        # Debug information
        print("\n=== Debug Information ===")
        print(f"Labels array shape: {labels.shape}")
        print(f"Number of unique label types: {len(unique_labels)}")
        print(f"Unique label types: {unique_labels}")
        
        # Create readable output
        results = {
            "total_samples": len(labels),
            "label_combinations": [],
            "individual_label_counts": {}
        }
        
        # Count label combinations
        combination_counts = Counter(tuple(sorted(labels)) for labels in sample_labels)
        sorted_combinations = sorted(combination_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Analyze combinations
        for combo, count in sorted_combinations:
            if combo == ('normal',) or not combo:
                label_desc = "Normal (no anomalies)"
            else:
                label_desc = " + ".join(combo)
            
            results["label_combinations"].append({
                "labels": label_desc,
                "count": int(count),
                "percentage": f"{(count/len(labels))*100:.2f}%"
            })
        
        # Count individual label occurrences
        for label in unique_labels:
            count = sum(1 for sample in sample_labels if label in sample)
            results["individual_label_counts"][label] = {
                "count": int(count),
                "percentage": f"{(count/len(labels))*100:.2f}%"
            }
        
        # Add count of normal samples
        normal_count = sum(1 for sample in sample_labels if len(sample) == 1 and (sample[0] == 'normal' or not sample[0]))
        results["individual_label_counts"]["Normal (no anomalies)"] = {
            "count": int(normal_count),
            "percentage": f"{(normal_count/len(labels))*100:.2f}%"
        }
        
        # Print results in a readable format
        print("\n=== Dataset Label Analysis ===")
        print(f"\nTotal number of samples: {results['total_samples']}")
        
        print("\nLabel Combinations (sorted by frequency):")
        print("-" * 50)
        for combo in results["label_combinations"]:
            print(f"{combo['labels']}: {combo['count']} samples ({combo['percentage']})")
        
        print("\nIndividual Label Counts:")
        print("-" * 50)
        for label, stats in results["individual_label_counts"].items():
            print(f"{label}: {stats['count']} samples ({stats['percentage']})")
        
        # If exclude_labels is provided, analyze dataset composition after exclusion
        if exclude_labels:
            print("\n=== Analysis with Excluded Labels ===")
            print(f"Labels to exclude: {exclude_labels}")
            
            # Create masks for different subsets based on label strings
            excluded_mask = np.zeros(len(labels), dtype=bool)
            if exclude_labels:
                print("\n=== Detailed Exclusion Analysis ===")
                print(f"Labels to exclude: {exclude_labels}")
                
                # First, analyze which combinations contain excluded labels
                excluded_combinations = []
                for combo, count in sorted_combinations:
                    if any(label in combo for label in exclude_labels):
                        excluded_combinations.append((combo, count))
                
                print("\nCombinations containing excluded labels:")
                print("-" * 50)
                total_excluded = 0
                for combo, count in excluded_combinations:
                    print(f"{' + '.join(combo)}: {count} samples")
                    total_excluded += count
                print(f"\nTotal samples containing excluded labels: {total_excluded}")
                
                # Now mark samples for exclusion
                for i, sample_label_list in enumerate(sample_labels):
                    if any(label in sample_label_list for label in exclude_labels):
                        excluded_mask[i] = True
            
            # Identify normal samples (those with only 'normal' or empty label)
            normal_mask = np.array([len(sample) == 1 and (sample[0] == 'normal' or not sample[0]) 
                                  for sample in sample_labels])
            
            # Identify anomalous samples that aren't excluded
            anomalous_mask = ~normal_mask & ~excluded_mask
            
            # Get indices for different sets
            normal_indices = np.where(normal_mask)[0]
            anomalous_indices = np.where(anomalous_mask)[0]
            excluded_indices = np.where(excluded_mask)[0]
            
            # Calculate sizes for different splits
            train_ratio = 0.8
            val_ratio = 0.1
            
            # First split both normal and anomalous into train+val/test
            test_size = 1.0 - train_ratio - val_ratio
            
            # Split normal samples
            normal_trainval = normal_indices[:int(len(normal_indices) * (1 - test_size))]
            normal_test = normal_indices[int(len(normal_indices) * (1 - test_size)):]
            
            # Split anomalous samples (non-excluded)
            anomalous_trainval = anomalous_indices[:int(len(anomalous_indices) * (1 - test_size))]
            anomalous_test = anomalous_indices[int(len(anomalous_indices) * (1 - test_size)):]
            
            # Split remaining data into train/val
            val_size = val_ratio / (train_ratio + val_ratio)
            
            # Training set
            normal_train = normal_trainval[:int(len(normal_trainval) * (1 - val_size))]
            anomalous_train = anomalous_trainval[:int(len(anomalous_trainval) * (1 - val_size))]
            
            # Validation set
            normal_val = normal_trainval[int(len(normal_trainval) * (1 - val_size)):]
            anomalous_val = anomalous_trainval[int(len(anomalous_trainval) * (1 - val_size)):]
            
            # For supervised sets, balance normal and anomalous
            n_supervised_train = min(len(normal_train), len(anomalous_train))
            n_supervised_val = min(len(normal_val), len(anomalous_val))
            
            print("\nDataset Composition After Exclusion:")
            print("-" * 50)
            print(f"Total samples: {len(labels)}")
            print(f"Normal samples: {len(normal_indices)}")
            print(f"Anomalous samples (non-excluded): {len(anomalous_indices)}")
            print(f"Excluded samples: {len(excluded_indices)}")
            
            print("\nSplit Sizes with Balancing:")
            print("-" * 50)
            print("SSL Training Set (normal only):")
            print(f"  Normal: {len(normal_train)}")
            
            print("\nSSL Validation Set (normal only):")
            print(f"  Normal: {len(normal_val)}")
            
            print("\nSupervised Training Set (balanced):")
            print(f"  Normal: {n_supervised_train}")
            print(f"  Anomalous: {n_supervised_train}")
            print(f"  Total: {n_supervised_train * 2}")
            
            print("\nSupervised Validation Set (balanced):")
            print(f"  Normal: {n_supervised_val}")
            print(f"  Anomalous: {n_supervised_val}")
            print(f"  Total: {n_supervised_val * 2}")
            
            print("\nTest Set:")
            print(f"  Normal: {len(normal_test)}")
            print(f"  Anomalous (non-excluded): {len(anomalous_test)}")
            print(f"  Excluded anomalies: {len(excluded_indices)}")
            print(f"  Total: {len(normal_test) + len(anomalous_test) + len(excluded_indices)}")
            
            # Add excluded labels analysis to results
            results["excluded_labels_analysis"] = {
                "excluded_labels": exclude_labels,
                "dataset_composition": {
                    "total_samples": len(labels),
                    "normal_samples": int(len(normal_indices)),
                    "anomalous_samples_non_excluded": int(len(anomalous_indices)),
                    "excluded_samples": int(len(excluded_indices))
                },
                "split_sizes": {
                    "ssl_training": {
                        "normal": int(len(normal_train))
                    },
                    "ssl_validation": {
                        "normal": int(len(normal_val))
                    },
                    "supervised_training": {
                        "normal": int(n_supervised_train),
                        "anomalous": int(n_supervised_train),
                        "total": int(n_supervised_train * 2)
                    },
                    "supervised_validation": {
                        "normal": int(n_supervised_val),
                        "anomalous": int(n_supervised_val),
                        "total": int(n_supervised_val * 2)
                    },
                    "test": {
                        "normal": int(len(normal_test)),
                        "anomalous_non_excluded": int(len(anomalous_test)),
                        "excluded_anomalies": int(len(excluded_indices)),
                        "total": int(len(normal_test) + len(anomalous_test) + len(excluded_indices))
                    }
                }
            }
        
        # Save results to JSON
        with open('label_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze label distributions in HDF5 dataset")
    parser.add_argument("data_path", type=str, help="Path to the HDF5 dataset file")
    parser.add_argument("--exclude-labels", type=str, nargs="+", help="Labels to exclude from training")
    
    args = parser.parse_args()
    analyze_label_distributions(args.data_path, args.exclude_labels) 