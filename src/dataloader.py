# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py.py

# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import logging
import h5py
from tqdm import tqdm
import os

logging.basicConfig(filename='audio_loading.log', level=logging.DEBUG, format='%(asctime)s %(message)s')


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))
        
        # Create sources attribute for hydrophone metrics
        self.sources = []
        for item in self.data:
            # Use filename as source if available, otherwise use a dummy source
            if 'wav' in item:
                # Extract hydrophone name from filename (e.g., "hydrophone_1_sample.wav" -> "hydrophone_1")
                filename = item['wav']
                basename = os.path.basename(filename)
                parts = basename.split('_')
                if len(parts) > 1:
                    source = f"{parts[0]}_{parts[1]}"
                else:
                    source = f"hydrophone_{hash(filename) % 5 + 1}"
            else:
                # Create a dummy source
                source = f"hydrophone_{len(self.sources) % 5 + 1}"
            self.sources.append(source)
        
        print(f"Created {len(self.sources)} sources with {len(set(self.sources))} unique hydrophones")

    def _wav2fbank(self, filename, filename2=None):
        """
        Loads and processes audio file into spectrogram.
        
        Args:
            filename: Path to audio file
            filename2: Path to second audio file for mixup (optional)
            
        Returns:
            fbank: Spectrogram tensor
        """
        # Mixup with two files
        if filename2 is not None:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)
            
            # Ensure same length by padding or truncating
            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # Pad waveform2
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # Truncate waveform1
                    waveform1 = waveform1[0, 0:waveform2.shape[1]]
            
            # Sample lambda from beta distribution
            mix_lambda = np.random.beta(10, 10)
            
            # Mix waveforms
            waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            # Load single file
            try:
                waveform, sr = torchaudio.load(filename)
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
                # Return zeros if file can't be loaded
                waveform = torch.zeros(1, 16000)
                sr = 16000
        
        # Resample if needed
        target_sr = self.audio_conf.get('target_sr', 16000)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        
        # Convert to spectrogram
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, 
            htk_compat=True,
            sample_frequency=target_sr,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=self.melbins,
            dither=0.0,
            frame_shift=10
        )
        
        # Normalize if not skipping
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        
        # Add noise if enabled
        if self.noise:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        
        # Return spectrogram
        return fbank

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data[index]
        label_indices = np.zeros(self.label_num)
        # wav
        fbank = self._wav2fbank(datum['wav'])
        label_indices = np.zeros(self.label_num)

        if not self.audio_conf.get('multi_label'):
            # integer encode
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0
        else:
            # one-hot encode
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0

        label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = torch.transpose(fbank, 0, 1)

        # mixup
        if random.random() < self.mixup:
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            mix_fbank = self._wav2fbank(mix_datum['wav'])
            mix_label_indices = np.zeros(self.label_num)
            
            if not self.audio_conf.get('multi_label'):
                for label_str in mix_datum['labels'].split(','):
                    mix_label_indices[int(self.index_dict[label_str])] = 1.0
            else:
                for label_str in mix_datum['labels'].split(','):
                    mix_label_indices[int(self.index_dict[label_str])] = 1.0
                    
            mix_label_indices = torch.FloatTensor(mix_label_indices)
            lam = np.random.beta(10, 10)
            fbank = lam * fbank + (1 - lam) * mix_fbank
            label_indices = lam * label_indices + (1 - lam) * mix_label_indices

        # Get source for this sample
        source = self.sources[index]
        
        # Return spectrogram, labels, and source
        return fbank, label_indices, source

    def __len__(self):
        return len(self.data)

def calculate_dataset_stats(h5_path, batch_size=100):
    """
    Calculate mean and standard deviation of a HDF5 dataset using online/streaming calculations.
    Uses Welford's online algorithm for numerical stability.
    
    Args:
        h5_path: Path to HDF5 file
        batch_size: Number of samples to process at once
    
    Returns:
        mean: Mean value of the dataset
        std: Standard deviation of the dataset
    """
    with h5py.File(h5_path, 'r') as f:
        spectrograms = f['spectrograms']
        total_samples = len(spectrograms)
        
        # Initialize variables for Welford's online algorithm
        mean = 0
        M2 = 0  # Sum of squared distances from mean
        count = 0
        
        # Process data in batches
        num_batches = (total_samples + batch_size - 1) // batch_size
        with tqdm(total=num_batches, desc="Calculating dataset statistics") as pbar:
            for i in range(0, total_samples, batch_size):
                batch = spectrograms[i:min(i + batch_size, total_samples)][:]
                batch_size_actual = batch.shape[0]
                
                # Flatten all dimensions except the batch dimension
                batch_flat = batch.reshape(batch_size_actual, -1)
                
                # Update running statistics for each sample in the batch
                for x in batch_flat:
                    count += 1
                    delta = x - mean
                    mean += delta / count
                    delta2 = x - mean
                    M2 += delta * delta2
                
                pbar.update(1)
        
        # Calculate final statistics
        std = np.sqrt(M2 / (count - 1))
        
        # Average across all dimensions
        mean = float(np.mean(mean))
        std = float(np.mean(std))
        
        print(f"\nDataset statistics:")
        print(f"Mean: {mean:.6f}")
        print(f"Std:  {std:.6f}")
        
        return mean, std

class HDF5Dataset(Dataset):
    def __init__(self, h5_path, split='train', train_ratio=0.8, val_ratio=0.1, seed=42, 
                 target_length=1024, num_mel_bins=128, freqm=0, timem=0, 
                 dataset_mean=None, dataset_std=None, mixup=0.0):
        """
        Args:
            h5_path: Path to HDF5 file
            split: One of ['train', 'val', 'test']
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            seed: Random seed for reproducibility
            target_length: Target length of spectrograms
            num_mel_bins: Number of mel bins in spectrograms
            freqm: Maximum number of frequency bands to mask
            timem: Maximum number of time steps to mask
            dataset_mean: Dataset mean for normalization (required)
            dataset_std: Dataset standard deviation for normalization (required)
            mixup: Mixup probability
        """
        super(HDF5Dataset, self).__init__()
        
        if dataset_mean is None or dataset_std is None:
            raise ValueError("dataset_mean and dataset_std must be provided")
            
        self.h5_path = h5_path
        self.split = split
        self.target_length = target_length
        self.num_mel_bins = num_mel_bins
        self.freqm = freqm
        self.timem = timem
        self.mixup = mixup
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Open HDF5 file
        self.h5_file = h5py.File(h5_path, 'r')
        
        # Create indices for train/val/test splits
        total_size = len(self.h5_file['spectrograms'])
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        
        val_size = int(val_ratio * total_size)

        if split == 'val':
            self.indices = indices[:val_size]  # Fixed validation set from the first val_size samples
        elif split == 'train':
            self.indices = indices[val_size:val_size + int(train_ratio * total_size)]  # Training set follows
        else:  # test
            self.indices = indices[val_size + int(train_ratio * total_size):]  # Remaining samples for test
        
        print(f"Using {len(self.indices)} samples for {split} split")
        
        # Load sources if available in the HDF5 file
        if 'sources' in self.h5_file:
            try:
                # Try to load all sources at once for better performance
                all_sources = self.h5_file['sources'][:]
                # Convert bytes to strings if needed
                if isinstance(all_sources[0], bytes):
                    all_sources = [s.decode('utf-8') for s in all_sources]
                self.sources = all_sources
                print(f"Loaded {len(self.sources)} sources from HDF5 file")
            except Exception as e:
                print(f"Error loading sources from HDF5 file: {e}")
                # Create dummy sources as fallback
                self._create_dummy_sources()
        else:
            print("No 'sources' dataset found in HDF5 file. Creating dummy sources.")
            self._create_dummy_sources()
    
    def _create_dummy_sources(self):
        """Create dummy sources for hydrophone metrics tracking."""
        total_size = len(self.h5_file['spectrograms'])
        # Create dummy hydrophone names (hydrophone_1, hydrophone_2, etc.)
        self.sources = []
        for i in range(total_size):
            # Create 5 different dummy hydrophones
            hydrophone_id = f"hydrophone_{i % 5 + 1}"
            self.sources.append(hydrophone_id)
        print(f"Created {len(self.sources)} dummy sources with {len(set(self.sources))} unique hydrophones")

    def __getitem__(self, index):
        """
        Returns a spectrogram and its labels
        """
        # Get actual index from split indices
        actual_index = self.indices[index]
        
        # Get spectrogram and convert to torch tensor
        fbank = torch.from_numpy(self.h5_file['spectrograms'][actual_index]).float()
        
        # Get labels
        label_indices = torch.from_numpy(self.h5_file['labels'][actual_index]).float()
        
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup and self.split == 'train':
            # select another sample
            mix_sample_idx = random.randint(0, len(self.indices)-1)
            mix_actual_index = self.indices[mix_sample_idx]
            
            # get the spectrogram and labels
            mix_fbank = torch.from_numpy(self.h5_file['spectrograms'][mix_actual_index]).float()
            mix_label_indices = torch.from_numpy(self.h5_file['labels'][mix_actual_index]).float()
            
            # sample lambda from beta distribution
            lam = np.random.beta(10, 10)
            
            # mix
            fbank = lam * fbank + (1 - lam) * mix_fbank
            label_indices = lam * label_indices + (1 - lam) * mix_label_indices
        
        # normalize the input
        fbank = (fbank - self.dataset_mean) / self.dataset_std
        
        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        
        if self.freqm != 0 and self.split == 'train':
            fbank = freqm(fbank)
        
        if self.timem != 0 and self.split == 'train':
            fbank = timem(fbank)
        
        # Get source for this sample
        source = self.sources[actual_index]
        
        # Return spectrogram, labels, and source
        return fbank, label_indices, source

    def __len__(self):
        return len(self.indices)

    def __del__(self):
        """Clean up any open HDF5 file handles"""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()