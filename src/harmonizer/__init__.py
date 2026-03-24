"""
Harmonizer: Transformer-based SDRF extraction from proteomics publications.

This package provides utilities for training and inference on the
Kaggle SDRF metadata extraction competition.
"""

from .data import SDRFDataset, load_data
from .model import SDRFTransformer
from .inference import generate_submission
from .train import train_model

__all__ = [
    "SDRFDataset",
    "load_data",
    "SDRFTransformer",
    "generate_submission",
    "train_model",
]
