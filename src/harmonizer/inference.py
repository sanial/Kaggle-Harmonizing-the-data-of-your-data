"""
Inference and submission generation for SDRF extraction.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from .data import load_data, load_sample_submission
from .model import SDRFTransformer


def generate_submission(
    model_path: Optional[str] = None,
    test_data_path: str = "data/TestPubText",
    sample_submission_path: str = "data/SampleSubmission.csv",
    output_path: str = "outputs/predictions/submission.csv",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> pd.DataFrame:
    """
    Generate submission CSV from trained model predictions.
    
    Args:
        model_path: Path to saved model directory
        test_data_path: Path to test publication texts
        sample_submission_path: Path to sample submission template
        output_path: Where to save the submission CSV
        device: PyTorch device
        
    Returns:
        Submission DataFrame
    """
    
    # Load sample template
    submission_df = load_sample_submission(sample_submission_path)
    
    # Load test data
    _, _, test_texts = load_data(test_path=test_data_path)
    
    if not test_texts:
        print("Warning: No test data found. Using template as-is.")
        return submission_df
    
    # Load model (if provided)
    if model_path:
        model = SDRFTransformer()
        state_dict = torch.load(
            Path(model_path) / "model.pt",
            map_location=device,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        tokenizer = model.get_tokenizer()
        
        # Generate predictions
        predictions = []
        with torch.no_grad():
            for text in tqdm(test_texts, desc="Generating predictions"):
                # Tokenize
                encoding = tokenizer(
                    text,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                
                # Predict
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                
                outputs = model(input_ids, attention_mask)
                pred = outputs.argmax(dim=-1)
                
                predictions.append(pred)
        
        # TODO: Map predictions to SDRF fields
        print(f"Generated {len(predictions)} predictions")
    else:
        print("No model provided. Using sample submission template.")
    
    # Save submission
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission saved to {output_path}")
    return submission_df
