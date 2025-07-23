#!/usr/bin/env python3
"""
Configuration utilities for dataset creation and processing.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

class DatasetConfig:
    """
    Configuration manager for dataset creation and processing.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML config file. If None, uses default location.
        """
        if config_path is None:
            # Default to config/dataset_config.yaml relative to script location
            script_dir = Path(__file__).parent.parent.parent
            config_path = script_dir / "config" / "dataset_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @property
    def anomaly_labels(self) -> List[str]:
        """Get list of anomaly labels."""
        return self.config['anomaly_labels']
    
    @property
    def normal_label(self) -> str:
        """Get the normal/background label."""
        return self.config['label_encoding']['normal_label']
    
    @property
    def label_separator(self) -> str:
        """Get the separator for multiple labels."""
        return self.config['label_encoding']['separator']
    
    @property
    def max_string_length(self) -> int:
        """Get maximum string length for labels."""
        return self.config['label_encoding']['max_string_length']
    
    @property
    def target_size(self) -> List[int]:
        """Get target spectrogram size [height, width]."""
        return self.config['spectrogram']['target_size']
    
    @property
    def expected_shape(self) -> List[int]:
        """Get expected raw spectrogram shape [height, width]."""
        return self.config['spectrogram']['expected_shape']
    
    @property
    def channels(self) -> int:
        """Get number of channels."""
        return self.config['spectrogram']['channels']
    
    @property
    def data_type(self) -> str:
        """Get data type for spectrograms."""
        return self.config['spectrogram']['data_type']
    
    @property
    def batch_size(self) -> int:
        """Get processing batch size."""
        return self.config['processing']['batch_size']
    
    @property
    def max_workers(self) -> Optional[int]:
        """Get maximum number of worker processes."""
        return self.config['processing']['max_workers']
    
    @property
    def compression(self) -> str:
        """Get HDF5 compression method."""
        return self.config['output']['compression']
    
    @property
    def compression_level(self) -> int:
        """Get HDF5 compression level."""
        return self.config['output']['compression_level']
    
    @property
    def chunk_size(self) -> List[int]:
        """Get HDF5 chunk size."""
        return self.config['output']['chunk_size']
    
    @property
    def dataset_metadata(self) -> Dict[str, str]:
        """Get dataset metadata."""
        return self.config['dataset']
    
    @property
    def hydrophone_patterns(self) -> Dict[str, str]:
        """Get hydrophone ID patterns."""
        return self.config['hydrophone_patterns']
    
    def get_label_index(self, label: str) -> int:
        """
        Get the index of a specific anomaly label.
        
        Args:
            label: The anomaly label string
            
        Returns:
            Index of the label in the anomaly_labels list
            
        Raises:
            ValueError: If label is not found
        """
        try:
            return self.anomaly_labels.index(label)
        except ValueError:
            raise ValueError(f"Label '{label}' not found in anomaly_labels: {self.anomaly_labels}")
    
    def create_label_encoding_dict(self) -> Dict[str, int]:
        """
        Create a dictionary mapping label names to indices.
        
        Returns:
            Dictionary with label names as keys and indices as values
        """
        return {label: idx for idx, label in enumerate(self.anomaly_labels)}
    
    def validate_config(self) -> bool:
        """
        Validate the configuration for common issues.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration has issues
        """
        # Check required sections
        required_sections = ['dataset', 'anomaly_labels', 'label_encoding', 'spectrogram']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Check anomaly labels
        if not self.anomaly_labels:
            raise ValueError("anomaly_labels cannot be empty")
        
        if len(self.anomaly_labels) != len(set(self.anomaly_labels)):
            raise ValueError("Duplicate labels found in anomaly_labels")
        
        # Check target size
        if len(self.target_size) != 2:
            raise ValueError("target_size must be [height, width]")
        
        return True
    
    def save_metadata_to_h5(self, h5_file) -> None:
        """
        Save configuration metadata to HDF5 file as attributes.
        
        Args:
            h5_file: Open HDF5 file object
        """
        # Save dataset metadata
        for key, value in self.dataset_metadata.items():
            h5_file.attrs[f'dataset_{key}'] = value
        
        # Save anomaly labels as a dataset for easy access
        label_data = [label.encode('utf-8') for label in self.anomaly_labels]
        h5_file.create_dataset('anomaly_label_names', 
                             data=label_data,
                             dtype=f'S{max(len(label) for label in self.anomaly_labels)}')
        
        # Save label encoding info
        h5_file.attrs['normal_label'] = self.normal_label
        h5_file.attrs['label_separator'] = self.label_separator
        h5_file.attrs['max_string_length'] = self.max_string_length
        
        # Save processing metadata
        h5_file.attrs['target_height'] = self.target_size[0]
        h5_file.attrs['target_width'] = self.target_size[1]
        h5_file.attrs['channels'] = self.channels
        h5_file.attrs['data_type'] = self.data_type
        
        # Save creation timestamp
        import datetime
        h5_file.attrs['created_timestamp'] = datetime.datetime.now().isoformat()


def load_config(config_path: Optional[str] = None) -> DatasetConfig:
    """
    Convenience function to load dataset configuration.
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        DatasetConfig object
    """
    return DatasetConfig(config_path)


# For backward compatibility - mimic the old defaults import
class Defaults:
    """Backward compatibility wrapper for old defaults import."""
    
    def __init__(self):
        self.config = load_config()
    
    @property
    def anomalies(self) -> List[str]:
        """Get anomaly labels (for backward compatibility)."""
        return self.config.anomaly_labels


# Create a default instance for easy importing
defaults = Defaults()

if __name__ == "__main__":
    # Test the configuration
    config = load_config()
    config.validate_config()
    
    print("✅ Configuration loaded successfully!")
    print(f"📋 Anomaly labels: {config.anomaly_labels}")
    print(f"🎵 Target size: {config.target_size}")
    print(f"⚙️ Batch size: {config.batch_size}") 