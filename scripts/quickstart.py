"""
One-command quickstart for train -> infer -> score.

Usage:
    python scripts/quickstart.py --device cpu
    python scripts/quickstart.py --skip-train --device cpu
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.Scoring import score
from src.harmonizer.inference import generate_submission
from src.harmonizer.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quickstart pipeline: train -> infer -> score"
    )
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device for training and inference",
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/models/transformer_sdrf",
        help="Model directory to use when --skip-train is set",
    )
    parser.add_argument(
        "--test-data",
        default="data/TestPubText",
        help="Path to test publication texts",
    )
    parser.add_argument(
        "--sample-submission",
        default="data/SampleSubmission.csv",
        help="Path to sample submission template",
    )
    parser.add_argument(
        "--submission-output",
        default="outputs/predictions/submission.csv",
        help="Output path for generated submission CSV",
    )
    parser.add_argument(
        "--solution",
        default="data/SampleSubmission.csv",
        help="Path to solution CSV for local scoring",
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/metrics/detailed_metrics.csv",
        help="Output path for detailed metrics CSV",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and use --model-dir for inference",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Skip scoring step",
    )
    return parser.parse_args()


def run_quickstart(args: argparse.Namespace) -> None:
    model_dir: Path

    print("=" * 72)
    print("Quickstart: train -> infer -> score")
    print("=" * 72)

    if args.skip_train:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            print(f"Warning: model directory not found: {model_dir}")
            print("Inference will run with template fallback if no model is available.")
        else:
            print(f"[1/3] Skipping train; using model at: {model_dir}")
    else:
        print(f"[1/3] Training model with config: {args.config}")
        model_dir = train_model(config_path=args.config, device=args.device)
        print(f"Training done. Model saved at: {model_dir}")

    print(f"[2/3] Running inference to: {args.submission_output}")
    model_path_for_infer = str(model_dir) if model_dir.exists() else None
    submission_df = generate_submission(
        model_path=model_path_for_infer,
        test_data_path=args.test_data,
        sample_submission_path=args.sample_submission,
        output_path=args.submission_output,
        device=args.device,
    )
    print(f"Inference done. Submission shape: {submission_df.shape}")

    if args.skip_score:
        print("[3/3] Skipped scoring (--skip-score set)")
        return

    print(f"[3/3] Scoring submission and saving metrics to: {args.metrics_output}")
    solution_df = pd.read_csv(args.solution)
    submission_df = pd.read_csv(args.submission_output)
    eval_df, final_score = score(solution_df, submission_df, row_id_column_name="ID")

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(metrics_path, index=False)

    print(f"Final SDRF Average F1 Score: {final_score:.6f}")
    print(f"Saved detailed metrics to: {metrics_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run_quickstart(cli_args)
