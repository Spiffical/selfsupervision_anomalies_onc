import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import h5py
import sys

class ONCSpectrogramDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        target_length: int = 1024,
        num_mel_bins: int = 128,
        freqm: int = 0,
        timem: int = 0,
        dataset_mean: float = None,
        dataset_std: float = None,
        mixup: float = 0.0,
        supervised: bool = True,
        ood: int = -1,
        amount: float = 1.0,
        subsample_test: bool = True
    ):
        self.data_path = data_path
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.target_length = target_length
        self.num_mel_bins = num_mel_bins
        self.freqm = freqm
        self.timem = timem
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.mixup = mixup
        self.supervised = supervised
        self.ood = ood
        self.amount = amount
        self.subsample_test = subsample_test
        self.test_seed = seed
            
        self._prepare_dataset()

    def _prepare_dataset(self):
        """Prepare dataset by collecting sample information and applying filtering"""
        with h5py.File(self.data_path, 'r') as hf:
            # Get all indices and labels
            all_indices = np.arange(len(hf['labels']))
            labels = hf['labels'][:]
            
            # Get label strings for each sample
            raw_label_strings = hf['label_strings'][:]
            sample_labels = []
            
            # Process each label string which may contain multiple labels
            for idx in all_indices:
                if idx < len(raw_label_strings):
                    s = raw_label_strings[idx]
                    if isinstance(s, bytes):
                        decoded = s.decode('utf-8')
                        # Split if it contains semicolons (multiple labels)
                        if ';' in decoded:
                            sample_labels.append(decoded.split(';'))
                        else:
                            sample_labels.append([decoded])
                    else:
                        sample_labels.append([str(s)])
                else:
                    # Handle case where there might be fewer label strings than indices
                    sample_labels.append([''])
            
            # Identify normal samples (those with only 'normal' or empty label)
            normal_mask = np.array([len(sample) == 1 and (sample[0] == 'normal' or not sample[0]) 
                                  for sample in sample_labels])
            
            # Identify anomalous samples
            anomalous_mask = ~normal_mask
            
            # Get indices for different sets
            normal_indices = all_indices[normal_mask]
            anomalous_indices = all_indices[anomalous_mask]
            
            if not self.supervised:  # SSL pretraining
                # Set random seed for reproducibility
                rng = np.random.RandomState(self.seed)
                
                # Shuffle normal indices
                shuffled_normal = normal_indices.copy()
                rng.shuffle(shuffled_normal)
                
                # Calculate sizes directly from total normal samples
                total_normal = len(normal_indices)
                n_val = int(total_normal * self.val_ratio)
                n_train = int(total_normal * self.train_ratio)
                n_test = total_normal - n_val - n_train
                
                # First grab validation set
                normal_val = shuffled_normal[:n_val]
                # Then grab training set
                normal_train = shuffled_normal[n_val:n_val + n_train]
                # Rest is test set
                normal_test = shuffled_normal[n_val + n_train:]
                
                if self.split == 'train':
                    self.indices = normal_train
                elif self.split == 'val':
                    self.indices = normal_val
                else:  # test
                    self.indices = normal_test
            else:  # Supervised finetuning
                # Set random seed for reproducibility
                rng = np.random.RandomState(self.seed)
                
                # Shuffle both normal and anomalous indices
                shuffled_normal = normal_indices.copy()
                shuffled_anomalous = anomalous_indices.copy()
                rng.shuffle(shuffled_normal)
                rng.shuffle(shuffled_anomalous)
                
                # First split both normal and anomalous into train+val/test
                test_size = 1.0 - self.train_ratio - self.val_ratio
                
                # Split normal samples
                normal_trainval = shuffled_normal[:int(len(shuffled_normal) * (1 - test_size))]
                normal_test = shuffled_normal[int(len(shuffled_normal) * (1 - test_size)):]
                
                # Split anomalous samples
                anomalous_trainval = shuffled_anomalous[:int(len(shuffled_anomalous) * (1 - test_size))]
                anomalous_test = shuffled_anomalous[int(len(shuffled_anomalous) * (1 - test_size)):]
                
                # Split remaining data into train/val
                val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
                
                # Training set
                normal_train = normal_trainval[:int(len(normal_trainval) * (1 - val_size))]
                anomalous_train = anomalous_trainval[:int(len(anomalous_trainval) * (1 - val_size))]
                
                # Validation set
                normal_val = normal_trainval[int(len(normal_trainval) * (1 - val_size)):]
                anomalous_val = anomalous_trainval[int(len(anomalous_trainval) * (1 - val_size)):]
                
                if self.split == 'test':
                    # For test set, include all samples
                    self.indices = np.concatenate([normal_test, anomalous_test])
                else:
                    if self.supervised:
                        # For supervised training/validation, use balanced normal/anomalous
                        if self.split == 'train':
                            n_supervised = min(len(normal_train), len(anomalous_train))
                            supervised_normal = np.random.choice(normal_train, size=n_supervised, replace=False)
                            supervised_anomalous = np.random.choice(anomalous_train, size=n_supervised, replace=False)
                            self.indices = np.concatenate([supervised_normal, supervised_anomalous])
                        else:  # val
                            n_supervised = min(len(normal_val), len(anomalous_val))
                            supervised_normal = np.random.choice(normal_val, size=n_supervised, replace=False)
                            supervised_anomalous = np.random.choice(anomalous_val, size=n_supervised, replace=False)
                            self.indices = np.concatenate([supervised_normal, supervised_anomalous])
            
            # Store sample info
            self.sample_info = []
            for idx in self.indices:
                # Get source from HDF5 file or use None
                if 'sources' in hf:
                    try:
                        source = hf['sources'][idx]
                        if isinstance(source, bytes):
                            source = source.decode('utf-8')
                        
                        # Extract just the hydrophone name (e.g., ICLISTENHF1951)
                        # Assuming format is HYDROPHONENAME_timestamp_timestamp-suffix.mat
                        if '_' in source:
                            hydrophone_name = source.split('_')[0]
                            source = hydrophone_name
                    except Exception as e:
                        source = None
                else:
                    source = None
                
                # For OOD filtering, we consider a sample anomalous if it has any anomaly
                is_anomalous = not normal_mask[idx]
                
                self.sample_info.append({
                    'index': idx,
                    'labels': sample_labels[idx],  # Store actual label strings instead of one-hot vector
                    'source': source,
                    'is_anomalous': is_anomalous,
                    'is_excluded': False  # Default to False, will be set by exclude_anomaly_type if needed
                })
            
            if self.subsample_test and self.split == 'test':
                self.subsample()

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample = self.sample_info[idx]
        
        with h5py.File(self.data_path, 'r') as hf:
            data = hf['spectrograms'][sample['index']]
            
            # Convert labels to binary (normal vs anomalous)
            # For test set, excluded samples are still labeled as anomalous
            labels = torch.tensor(float(sample['is_anomalous'])).float()
        
        # Apply normalization
        data = self.normalise(data)
        data = torch.from_numpy(data).permute(2, 0, 1)  # [C, F, T]
        
        # Apply mixup if enabled and in training mode
        if self.split == 'train' and np.random.random() < self.mixup:
            mix_idx = np.random.randint(0, len(self.sample_info))
            mix_sample = self.sample_info[mix_idx]
            with h5py.File(self.data_path, 'r') as hf:
                mix_data = hf['spectrograms'][mix_sample['index']]
                mix_labels = torch.tensor(float(mix_sample['is_anomalous'])).float()
            
            mix_data = self.normalise(mix_data)
            mix_data = torch.from_numpy(mix_data).permute(2, 0, 1)
            
            # Apply mixup
            lam = np.random.beta(0.4, 0.4)
            data = lam * data + (1 - lam) * mix_data
            labels = lam * labels + (1 - lam) * mix_labels
        
        return data, labels, sample['source']

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.sample_info)

    def normalise(self, data: np.array) -> np.array:
        """Normalize data using dataset statistics or percentile clipping"""
        data = data.astype(np.float32)
        
        if self.dataset_mean is not None and self.dataset_std is not None:
            # Use provided dataset statistics
            data = (data - self.dataset_mean) / (self.dataset_std * 2)
        else:
            # Use percentile clipping and log transformation
            _min, _max = np.percentile(data, [self.amount, 100-self.amount])
            data = np.clip(data, _min, _max)
            data = np.log(data)
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        return np.nan_to_num(data, 0)

    def subsample(self):
        """Subsample test set to maintain desired contamination ratios"""
        if not self.supervised or self.split != 'test':
            return
            
        np.random.seed(self.test_seed)
        
        # Separate normal and anomalous samples
        normal_samples = []
        anomaly_samples = {}  # Dictionary to store samples by anomaly type
        
        for sample in self.sample_info:
            if len(sample['labels']) == 1 and (sample['labels'][0] == 'normal' or not sample['labels'][0]):
                normal_samples.append(sample)
            else:
                # A sample can belong to multiple anomaly classes
                for label in sample['labels']:
                    if label and label != 'normal':  # Skip empty labels and 'normal'
                        if label not in anomaly_samples:
                            anomaly_samples[label] = []
                        anomaly_samples[label].append(sample)
        
        # Keep all normal samples
        new_samples = normal_samples.copy()
        num_normal = len(normal_samples)
        
        # Sample anomalies according to contamination percentages
        used_samples = set()  # Track which samples have been added
        for anomaly_type, anomaly_list in anomaly_samples.items():
            percentage = 0.1  # Default 10% contamination
            if percentage > 0 and anomaly_list:
                num_to_keep = int(num_normal * percentage)
                if num_to_keep > 0:
                    # Filter out already used samples
                    available_samples = [s for s in anomaly_list 
                                      if s['source'] not in used_samples]
                    if available_samples:
                        selected = np.random.choice(
                            available_samples,
                            size=min(num_to_keep, len(available_samples)),
                            replace=False
                        )
                        for sample in selected:
                            if sample['source'] not in used_samples:
                                new_samples.append(sample)
                                used_samples.add(sample['source'])
        
        
        self.sample_info = new_samples

def get_label_strings_from_raw(raw_label_strings):
    """Convert raw label strings from HDF5 file to list of label lists."""
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
    return sample_labels

def exclude_anomaly_type(datasets, exclude_labels, data_path):
    """
    Exclude specified anomaly types from training and validation sets and add them to test set.
    This is the single source of truth for handling excluded samples.
    
    Args:
        datasets: Tuple of (train_dataset, val_dataset, test_dataset)
        exclude_labels: List of labels to exclude
        data_path: Path to the HDF5 data file
    
    Returns:
        Modified (train_dataset, val_dataset, test_dataset) tuple
    """
    train_dataset, val_dataset, test_dataset = datasets
    
    with h5py.File(data_path, 'r') as hf:
        raw_label_strings = hf['label_strings'][:]
        sample_labels = get_label_strings_from_raw(raw_label_strings)
        
        # Count combinations containing excluded labels
        combinations = {}
        total_excluded = 0
        
        # Function to identify samples to exclude from a dataset
        def get_excluded_samples(dataset):
            excluded_samples = []
            remaining_samples = []
            
            for sample in dataset.sample_info:
                sample_label_list = sample_labels[sample['index']]
                if any(label in sample_label_list for label in exclude_labels):
                    # Track combinations for logging
                    combo = tuple(sorted(sample_label_list))
                    combinations[combo] = combinations.get(combo, 0) + 1
                    # Mark sample for exclusion
                    sample['is_excluded'] = True
                    excluded_samples.append(sample)
                else:
                    remaining_samples.append(sample)
            
            return excluded_samples, remaining_samples
        
        # Process training and validation sets
        train_excluded, train_remaining = get_excluded_samples(train_dataset)
        val_excluded, val_remaining = get_excluded_samples(val_dataset)
        
        # Update training and validation sets
        train_dataset.sample_info = train_remaining
        train_dataset.indices = np.array([s['index'] for s in train_remaining])
        
        val_dataset.sample_info = val_remaining
        val_dataset.indices = np.array([s['index'] for s in val_remaining])
        
        # Add excluded samples to test set and mark them as excluded
        test_dataset.sample_info.extend(train_excluded + val_excluded)
        test_dataset.indices = np.concatenate([
            test_dataset.indices,
            np.array([s['index'] for s in train_excluded + val_excluded])
        ])
        
        # Print combinations being excluded
        print("\n=== EXCLUSION ANALYSIS ===")
        print(f"Labels being excluded: {exclude_labels}")
        print("\nCombinations containing excluded labels:")
        print("-" * 50)
        for combo, count in sorted(combinations.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {' + '.join(combo)}: {count} samples")
            total_excluded += count
        print(f"\nTotal samples containing excluded labels: {total_excluded}")
        print(f"These samples have been moved to the test set")
        
        return train_dataset, val_dataset, test_dataset

def get_onc_spectrogram_data(
    data_path: str,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    target_length: int = 1024,
    num_mel_bins: int = 128,
    freqm: int = 0,
    timem: int = 0,
    dataset_mean: float = None,
    dataset_std: float = None,
    mixup: float = 0.0,
    ood: int = -1,
    amount: float = 1.0,
    subsample_test: bool = True,
    exclude_labels: list = None
) -> tuple:
    """Load and split data into train/val/test sets"""
    
    print("\n=== DATASET CONFIGURATION ===")
    print(f"Data path: {data_path}")
    print(f"Split ratios - Train: {train_ratio*100}%, Val: {val_ratio*100}%, Test: {(1-train_ratio-val_ratio)*100}%")
    print(f"Random seed: {seed}")
    if exclude_labels:
        print(f"Excluding labels: {exclude_labels}")
    
    # Create initial datasets without exclusion
    ssl_train_dataset = ONCSpectrogramDataset(
        data_path=data_path,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        target_length=target_length,
        num_mel_bins=num_mel_bins,
        freqm=freqm,
        timem=timem,
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        mixup=mixup,
        supervised=False,
        ood=ood,
        amount=amount
    )
    
    ssl_val_dataset = ONCSpectrogramDataset(
        data_path=data_path,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        target_length=target_length,
        num_mel_bins=num_mel_bins,
        freqm=0,
        timem=0,
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        mixup=0.0,
        supervised=False,
        ood=ood,
        amount=amount
    )
    
    train_dataset = ONCSpectrogramDataset(
        data_path=data_path,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        target_length=target_length,
        num_mel_bins=num_mel_bins,
        freqm=freqm,
        timem=timem,
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        mixup=mixup,
        supervised=True,
        ood=ood,
        amount=amount
    )
    
    val_dataset = ONCSpectrogramDataset(
        data_path=data_path,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        target_length=target_length,
        num_mel_bins=num_mel_bins,
        freqm=0,
        timem=0,
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        mixup=0.0,
        supervised=True,
        ood=ood,
        amount=amount
    )
    
    test_dataset = ONCSpectrogramDataset(
        data_path=data_path,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        target_length=target_length,
        num_mel_bins=num_mel_bins,
        freqm=0,
        timem=0,
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        mixup=0.0,
        supervised=True,
        ood=ood,
        amount=amount,
        subsample_test=subsample_test
    )
    
    # If exclude_labels is provided, modify the supervised datasets
    if exclude_labels:
        train_dataset, val_dataset, test_dataset = exclude_anomaly_type(
            (train_dataset, val_dataset, test_dataset),
            exclude_labels,
            data_path
        )
    
    # Print final dataset composition
    print("\n=== FINAL DATASET COMPOSITION ===")
    print("\nSSL Datasets (normal samples only):")
    print("-" * 50)
    print(f"Training:   {len(ssl_train_dataset)} samples")
    print(f"Validation: {len(ssl_val_dataset)} samples")
    
    print("\nSupervised Datasets:")
    print("-" * 50)
    if hasattr(train_dataset, 'sample_info'):
        train_normal = sum(1 for s in train_dataset.sample_info if not s['is_anomalous'])
        train_anomalous = sum(1 for s in train_dataset.sample_info if s['is_anomalous'])
        print(f"Training Set:")
        print(f"  - Normal samples:    {train_normal}")
        print(f"  - Anomalous samples: {train_anomalous}")
        print(f"  - Total:             {len(train_dataset)} samples")
        
        val_normal = sum(1 for s in val_dataset.sample_info if not s['is_anomalous'])
        val_anomalous = sum(1 for s in val_dataset.sample_info if s['is_anomalous'])
        print(f"\nValidation Set:")
        print(f"  - Normal samples:    {val_normal}")
        print(f"  - Anomalous samples: {val_anomalous}")
        print(f"  - Total:             {len(val_dataset)} samples")
        
        test_normal = sum(1 for s in test_dataset.sample_info if not s['is_anomalous'])
        test_anomalous = sum(1 for s in test_dataset.sample_info if s['is_anomalous'])
        test_excluded = sum(1 for s in test_dataset.sample_info if s['is_excluded'])
        
        print(f"\nTest Set:")
        print(f"  - Normal samples:    {test_normal}")
        print(f"  - Anomalous samples: {test_anomalous}")
        if exclude_labels:
            print(f"  - Excluded samples:  {test_excluded}")
            print(f"  - Total:             {len(test_dataset)} samples")
    
    print("\n" + "="*80)
    
    return ssl_train_dataset, ssl_val_dataset, test_dataset, train_dataset, val_dataset 