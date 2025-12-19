"""ONC SSAMBA - Self-supervised anomaly detection for hydrophone spectrograms."""
__version__ = "0.1.0"

from .dataset import ONCSpectrogramDataset
from .utilities.training_utils import create_model