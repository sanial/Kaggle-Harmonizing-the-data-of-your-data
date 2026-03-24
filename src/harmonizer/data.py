"""
Data loading and preprocessing for SDRF extraction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import pandas as pd
from torch.utils.data import Dataset


class SDRFDataset(Dataset):
    """
    PyTorch Dataset for SDRF extraction task.
    
    Loads publication texts and corresponding SDRF annotations.
    """

    def __init__(
        self,
        text_list: List[str],
        labels: Optional[List[Dict[str, str]]] = None,
        tokenizer=None,
        max_length: int = 512,
    ):
        """
        Args:
            text_list: List of publication texts
            labels: List of SDRF annotation dictionaries
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.texts = text_list
        self.labels = labels or [{}] * len(text_list)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns tokenized text and labels for a single example.
        """
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        
        # Add labels if available
        if self.labels[idx]:
            item["labels"] = self.labels[idx]
        
        return item


def load_data(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
) -> Tuple[List[str], List[Dict]]:
    """
    Load training texts and SDRF annotations.
    
    Args:
        train_path: Path to training publication texts
        test_path: Path to test publication texts
        
    Returns:
        Tuple of (texts, labels, test_texts)
    """
    texts = []
    labels = []
    test_texts = []
    
    # Load training texts
    if train_path:
        train_dir = Path(train_path)
        if train_dir.exists():
            for json_file in train_dir.glob("*_PubText.json"):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        # Extract text
                        text = data.get("text", "") if isinstance(data, dict) else str(data)
                        texts.append(text)
                        labels.append({})  # Placeholder for SDRF labels
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
    
    # Load test texts
    if test_path:
        test_dir = Path(test_path)
        if test_dir.exists():
            for json_file in sorted(test_dir.glob("*_PubText.json")):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        text = data.get("text", "") if isinstance(data, dict) else str(data)
                        test_texts.append(text)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
    
    return texts, labels, test_texts


def load_sample_submission(path: str) -> pd.DataFrame:
    """
    Load the sample submission template.
    
    Args:
        path: Path to SampleSubmission.csv
        
    Returns:
        DataFrame with required columns
    """
    return pd.read_csv(path)
