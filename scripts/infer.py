"""
Inference entrypoint script for generating SDRF predictions.

Usage:
    python scripts/infer.py [--model outputs/models/transformer_sdrf] \
                            [--output outputs/predictions/submission.csv]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harmonizer.inference import generate_submission


def main():
    """Generate predictions with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SDRF predictions from trained model"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--test-data",
        default="data/TestPubText",
        help="Path to test publication texts",
    )
    parser.add_argument(
        "--sample",
        default="data/SampleSubmission.csv",
        help="Path to sample submission template",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions/submission.csv",
        help="Output path for submission CSV",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    
    args = parser.parse_args()
    
    print(f"Generating predictions...")
    submission_df = generate_submission(
        model_path=args.model,
        test_data_path=args.test_data,
        sample_submission_path=args.sample,
        output_path=args.output,
        device=args.device,
    )
    print(f"Submission shape: {submission_df.shape}")
    print(f"Submission saved to {args.output}")


if __name__ == "__main__":
    main()
