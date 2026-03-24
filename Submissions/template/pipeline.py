"""
Submission pipeline for transformer-based SDRF extraction.

This is a minimal working example that demonstrates how to:
1. Load the trained model
2. Generate predictions on test data
3. Format submission for Kaggle

Usage:
    python pipeline.py --model ../../outputs/models/transformer_sdrf
"""

import argparse
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.harmonizer.inference import generate_submission


def main():
    """Generate submission from model predictions."""
    parser = argparse.ArgumentParser(
        description="Generate SDRF extraction submission"
    )
    parser.add_argument(
        "--model",
        default="../../outputs/models/transformer_sdrf",
        help="Path to trained transformer model",
    )
    parser.add_argument(
        "--test-data",
        default="../../data/TestPubText",
        help="Path to test publications",
    )
    parser.add_argument(
        "--sample",
        default="../../data/SampleSubmission.csv",
        help="Path to sample submission template",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Output submission CSV path",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDRF Extraction - Transformer Pipeline")
    print("=" * 60)
    
    # Generate predictions
    submission_df = generate_submission(
        model_path=args.model if Path(args.model).exists() else None,
        test_data_path=args.test_data,
        sample_submission_path=args.sample,
        output_path=args.output,
    )
    
    print(f"\nSubmission shape: {submission_df.shape}")
    print(f"Columns: {list(submission_df.columns[:5])}...")
    print(f"Saved to: {Path(args.output).resolve()}")
    
    return submission_df


if __name__ == "__main__":
    main()
