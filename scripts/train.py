"""
Training entrypoint script for SDRF transformer model.

Usage:
    python scripts/train.py [--config configs/training.yaml]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harmonizer.train import train_model


def main():
    """Train the model with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train transformer-based SDRF extraction model"
    )
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on",
    )
    
    args = parser.parse_args()
    
    print(f"Training with config: {args.config}")
    model_path = train_model(args.config, device=args.device)
    print(f"Training complete. Model saved to {model_path}")


if __name__ == "__main__":
    main()
