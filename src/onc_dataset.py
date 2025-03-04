import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import h5py
import logging

logging.basicConfig(filename='onc_dataset.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

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
        print(f"Dataset initialized with supervised={self.supervised}")
        
    def _prepare_dataset(self):
        """Prepare dataset by collecting sample information and applying OOD filtering"""
        with h5py.File(self.data_path, 'r') as hf:
            # Get all indices and labels
            all_indices = np.arange(len(hf['labels']))
            labels = hf['labels'][:]
            
            # Check if sources dataset exists
            if 'sources' in hf:
                print(f"[DEBUG] Found 'sources' dataset in HDF5 file with shape {hf['sources'].shape}")
                # Check the first few sources
                sample_sources = hf['sources'][:5]
                print(f"[DEBUG] First few sources: {sample_sources}")
                if isinstance(sample_sources[0], bytes):
                    print(f"[DEBUG] Sources are stored as bytes, will decode to utf-8")
            else:
                print(f"[DEBUG] WARNING: No 'sources' dataset found in HDF5 file.")
                print(f"[DEBUG] Hydrophone metrics will not be available.")
            
            # Separate normal and anomalous samples
            normal_indices = all_indices[~np.any(labels, axis=1)]
            anomalous_indices = all_indices[np.any(labels, axis=1)]
            
            # First split both normal and anomalous into train+val/test
            test_size = 1.0 - self.train_ratio - self.val_ratio
            
            normal_trainval, normal_test = train_test_split(
                normal_indices,
                test_size=test_size,
                random_state=self.seed
            )
            
            anomalous_trainval, anomalous_test = train_test_split(
                anomalous_indices,
                test_size=test_size,
                random_state=self.seed
            )
            
            # Split normal training data into train/val
            normal_train, normal_val = train_test_split(
                normal_trainval,
                test_size=self.val_ratio/(self.train_ratio + self.val_ratio),
                random_state=self.seed
            )
            
            # For supervised training, use balanced normal/anomalous
            n_supervised = min(len(normal_trainval), len(anomalous_trainval))
            supervised_normal = np.random.choice(normal_trainval, size=n_supervised, replace=False)
            supervised_anomalous = np.random.choice(anomalous_trainval, size=n_supervised, replace=False)
            supervised_indices = np.concatenate([supervised_normal, supervised_anomalous])
            
            # Split supervised indices into train/val
            supervised_train, supervised_val = train_test_split(
                supervised_indices,
                test_size=self.val_ratio/(self.train_ratio + self.val_ratio),
                random_state=self.seed
            )
            
            # Assign indices based on split
            if self.split == 'train':
                self.indices = supervised_train if self.supervised else normal_train
            elif self.split == 'val':
                self.indices = supervised_val if self.supervised else normal_val
            else:  # test
                self.indices = np.concatenate([normal_test, anomalous_test])
            
            # Store sample info
            self.sample_info = []
            for idx in self.indices:
                labels = hf['labels'][idx]
                
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
                        print(f"[DEBUG] Error getting source for index {idx}: {e}")
                        source = None
                else:
                    source = None
                
                label_string = hf['label_strings'][idx].decode('utf-8')
                
                # Skip samples that only contain the OOD class (if specified)
                if self.ood != -1:
                    if labels[self.ood] == 1 and sum(labels) == 1:
                        continue
                
                # For OOD filtering, we consider a sample anomalous if it has any anomaly
                is_anomalous = np.any(labels)
                
                self.sample_info.append({
                    'index': idx,
                    'labels': labels,
                    'source': source,
                    'label_string': label_string,
                    'is_anomalous': is_anomalous
                })
            
            print(f"[DEBUG] Created sample_info with {len(self.sample_info)} samples")
            print(f"[DEBUG] First few sources in sample_info: {[s['source'] for s in self.sample_info[:5]]}")
            
            if self.subsample_test and self.split == 'test':
                self.subsample()

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample = self.sample_info[idx]
        
        with h5py.File(self.data_path, 'r') as hf:
            data = hf['spectrograms'][sample['index']]
            # Convert multi-class labels to binary (normal vs anomalous)
            is_anomalous = float(np.any(sample['labels']))
            # Return single binary label instead of one-hot encoding
            labels = torch.tensor(is_anomalous).float()  # 1.0 for anomaly, 0.0 for normal
        
        # Apply normalization
        data = self.normalise(data)
        data = torch.from_numpy(data).permute(2, 0, 1)  # [C, F, T]
        
        # Apply mixup if enabled and in training mode
        if self.split == 'train' and np.random.random() < self.mixup:
            mix_idx = np.random.randint(0, len(self.sample_info))
            mix_sample = self.sample_info[mix_idx]
            with h5py.File(self.data_path, 'r') as hf:
                mix_data = hf['spectrograms'][mix_sample['index']]
                # Convert mix sample labels to binary
                mix_is_anomalous = float(np.any(mix_sample['labels']))
                mix_labels = torch.tensor(mix_is_anomalous).float()
            
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
        anomaly_samples = {i: [] for i in range(len(self.sample_info[0]['labels']))}
        
        for sample in self.sample_info:
            if not np.any(sample['labels']):  # Normal sample
                normal_samples.append(sample)
            else:
                # A sample can belong to multiple anomaly classes
                for i, has_anomaly in enumerate(sample['labels']):
                    if has_anomaly and i != self.ood:  # Skip OOD class
                        anomaly_samples[i].append(sample)
        
        # Keep all normal samples
        new_samples = normal_samples.copy()
        num_normal = len(normal_samples)
        
        # Sample anomalies according to contamination percentages
        used_samples = set()  # Track which samples have been added
        for i, anomaly_list in anomaly_samples.items():
            if i == self.ood:  # Skip OOD class
                continue
                
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
    subsample_test: bool = True
) -> tuple:
    """Load and split data into train/val/test sets"""
    
    # Create datasets
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
        freqm=0,  # No augmentation for validation
        timem=0,  # No augmentation for validation
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        mixup=0.0,  # No mixup for validation
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
        freqm=0,  # No augmentation for test
        timem=0,  # No augmentation for test
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        mixup=0.0,  # No mixup for test
        supervised=True,
        ood=ood,
        amount=amount,
        subsample_test=subsample_test
    )
    
    # Create SSL datasets (using only normal data)
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
        freqm=0,  # No augmentation for validation
        timem=0,  # No augmentation for validation
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        mixup=0.0,  # No mixup for validation
        supervised=False,
        ood=ood,
        amount=amount
    )
    
    print(f"Dataset sizes:")
    print(f"SSL Train (normal only): {len(ssl_train_dataset)}")
    print(f"SSL Val (normal only): {len(ssl_val_dataset)}")
    print(f"Supervised Train (balanced): {len(train_dataset)}")
    print(f"Supervised Val (balanced): {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    return ssl_train_dataset, ssl_val_dataset, test_dataset, train_dataset, val_dataset 