"""ONC SSAMBA - Self-supervised anomaly detection for hydrophone spectrograms."""
__version__ = "0.1.0"


def __getattr__(name):
    if name == "ONCSpectrogramDataset":
        from .dataset import ONCSpectrogramDataset

        return ONCSpectrogramDataset
    if name == "create_model":
        from .utilities.training_utils import create_model

        return create_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ONCSpectrogramDataset", "create_model"]
