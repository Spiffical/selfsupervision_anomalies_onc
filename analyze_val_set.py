import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def analyze_dataset(data_path, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Analyze the dataset distribution per hydrophone in validation set"""
    with h5py.File(data_path, 'r') as hf:
        # Get all indices and labels
        all_indices = np.arange(len(hf['labels']))
        labels = hf['labels'][:]
        
        # Get sources
        sources = [s.decode('utf-8').split('_')[0] for s in hf['sources'][:]]
        
        # Separate normal and anomalous samples
        normal_indices = all_indices[~np.any(labels, axis=1)]
        anomalous_indices = all_indices[np.any(labels, axis=1)]
        
        # First split both normal and anomalous into train+val/test
        test_size = 1.0 - train_ratio - val_ratio
        
        normal_trainval, normal_test = train_test_split(
            normal_indices,
            test_size=test_size,
            random_state=seed
        )
        
        anomalous_trainval, anomalous_test = train_test_split(
            anomalous_indices,
            test_size=test_size,
            random_state=seed
        )
        
        # Split normal training data into train/val
        normal_val_size = val_ratio/(train_ratio + val_ratio)
        normal_train, normal_val = train_test_split(
            normal_trainval,
            test_size=normal_val_size,
            random_state=seed
        )
        
        # For supervised training, use balanced normal/anomalous
        n_supervised = min(len(normal_trainval), len(anomalous_trainval))
        supervised_normal = np.random.choice(normal_trainval, size=n_supervised, replace=False)
        supervised_anomalous = np.random.choice(anomalous_trainval, size=n_supervised, replace=False)
        supervised_indices = np.concatenate([supervised_normal, supervised_anomalous])
        
        # Split supervised indices into train/val
        supervised_val_size = val_ratio/(train_ratio + val_ratio)
        supervised_train, supervised_val = train_test_split(
            supervised_indices,
            test_size=supervised_val_size,
            random_state=seed
        )
        
        # Analyze validation set distribution
        print("\nValidation Set Analysis:")
        print("------------------------")
        print(f"Total validation samples: {len(supervised_val)}")
        
        # Get validation set sources and labels
        val_sources = [sources[i] for i in supervised_val]
        val_labels = labels[supervised_val]
        
        # Analyze per hydrophone
        unique_hydrophones = sorted(set(val_sources))
        print(f"\nPer-hydrophone distribution:")
        print("-----------------------------")
        
        for hydrophone in unique_hydrophones:
            hydrophone_indices = [i for i, s in enumerate(val_sources) if s == hydrophone]
            hydrophone_labels = val_labels[hydrophone_indices]
            
            n_total = len(hydrophone_indices)
            n_normal = sum(~np.any(hydrophone_labels, axis=1))
            n_anomalous = sum(np.any(hydrophone_labels, axis=1))
            
            print(f"\nHydrophone: {hydrophone}")
            print(f"Total samples: {n_total}")
            print(f"Normal samples: {n_normal}")
            print(f"Anomalous samples: {n_anomalous}")
            print(f"Ratio (anomalous/total): {n_anomalous/n_total:.2f}")

if __name__ == "__main__":
    # You'll need to specify your data path here
    data_path = "/scratch/merileo/different_locations_incl_backgroundpipelinenormals_multilabel.h5"
    analyze_dataset(data_path) 