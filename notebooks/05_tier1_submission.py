"""
05_tier1_submission.py
======================
Tier 1 multi-column TextCNN pipeline for SDRF metadata prediction.
Trains one TextCNN per Tier 1 column, runs inference on test set,
and writes a submission-ready CSV.

Run this in a Kaggle notebook kernel (T4 GPU available) or locally on CPU.
All 8 Tier 1 columns complete in under 15 minutes total.
"""

from __future__ import annotations

import math
import os
import random
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
OUTPUT_DIR = ROOT / "outputs"
TRAIN_PATH = OUTPUT_DIR / "train_preprocessed.csv"
TEST_PATH  = OUTPUT_DIR / "test_preprocessed.csv"   # adjust if different name
SUBMISSION_PATH = OUTPUT_DIR / "submission_tier1.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Tier 1 columns (CNN-suitable: keyword-dominated, closed vocabulary) ────────
TIER1_COLS = [
    "Comment[Instrument]",
    "Characteristics[Organism]",
    "Characteristics[Sex]",
    "Comment[FragmentationMethod]",
    "Comment[FragmentMassTolerance]",
    "Comment[PrecursorMassTolerance]",
    "Comment[FractionIdentifier]",
    "Characteristics[BiologicalReplicate]",
]

# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_VOCAB    = 40_000
MIN_FREQ     = 2
MAX_LENGTH   = 512      # tokens per document
EMBED_DIM    = 256
NUM_FILTERS  = 128
KERNEL_SIZES = (3, 5, 7)
DROPOUT      = 0.2
BATCH_SIZE   = 32
LR           = 1e-3
EPOCHS       = 3
RANDOM_STATE = 42
NOT_APPLICABLE = "Not Applicable"

# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ── Vocabulary ─────────────────────────────────────────────────────────────────
class SimpleVocab:
    PAD, UNK = 0, 1

    def __init__(self, min_freq: int = MIN_FREQ, max_vocab: int = MAX_VOCAB):
        self.min_freq  = min_freq
        self.max_vocab = max_vocab
        self.token_to_id: dict[str, int] = {"<pad>": self.PAD, "<unk>": self.UNK}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return str(text).lower().split()

    def fit(self, texts: list[str]) -> "SimpleVocab":
        counter: Counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))
        for token, freq in counter.most_common(self.max_vocab):
            if freq < self.min_freq or token in self.token_to_id:
                continue
            self.token_to_id[token] = len(self.token_to_id)
        return self

    def encode(self, text: str, max_length: int = MAX_LENGTH) -> list[int]:
        ids = [
            self.token_to_id.get(t, self.UNK)
            for t in self.tokenize(text)[:max_length]
        ]
        ids += [self.PAD] * (max_length - len(ids))
        return ids

    @property
    def size(self) -> int:
        return len(self.token_to_id)


# ── Dataset ────────────────────────────────────────────────────────────────────
class CNNDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int],
                 vocab: SimpleVocab, max_length: int = MAX_LENGTH):
        self.encoded = [vocab.encode(t, max_length) for t in texts]
        self.labels  = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],  dtype=torch.long),
        )


class InferenceDataset(Dataset):
    """No labels — for test set inference."""
    def __init__(self, texts: list[str], vocab: SimpleVocab,
                 max_length: int = MAX_LENGTH):
        self.encoded = [vocab.encode(t, max_length) for t in texts]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        return torch.tensor(self.encoded[idx], dtype=torch.long)


# ── Model ──────────────────────────────────────────────────────────────────────
class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int,
                 embed_dim: int = EMBED_DIM,
                 num_filters: int = NUM_FILTERS,
                 kernel_sizes: tuple = KERNEL_SIZES,
                 dropout: float = DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids).transpose(1, 2)          # (B, E, L)
        pooled = [torch.amax(torch.relu(c(x)), dim=2)
                  for c in self.convs]                          # each (B, F)
        out = self.dropout(torch.cat(pooled, dim=1))            # (B, 3F)
        return self.classifier(out)                             # (B, num_labels)


# ── Per-column trainer ─────────────────────────────────────────────────────────
@dataclass
class ColumnResult:
    col: str
    val_f1: float
    id_to_label: dict[int, str]
    model: TextCNN
    vocab: SimpleVocab
    label_to_id: dict[str, int] = field(default_factory=dict)


def train_column(col: str, train_df: pd.DataFrame) -> ColumnResult | None:
    """Train a TextCNN for one SDRF column. Returns None if not enough data."""
    df = train_df[["text_input", col]].copy()
    df.columns = ["text", "label"]
    df = df[df["label"] != NOT_APPLICABLE].dropna()

    if len(df) < 20:
        print(f"  [SKIP] {col} — only {len(df)} labelled rows")
        return None

    # Build label maps (keep all classes with ≥ 2 samples)
    counts = df["label"].value_counts()
    keep   = counts[counts >= 2].index.tolist()
    df     = df[df["label"].isin(keep)].reset_index(drop=True)

    categories    = pd.Categorical(df["label"])
    df["label_id"] = categories.codes
    id_to_label   = dict(enumerate(categories.categories))
    label_to_id   = {v: k for k, v in id_to_label.items()}
    num_labels    = len(id_to_label)

    # Train / val split
    try:
        tr, va = train_test_split(
            df, test_size=0.15, random_state=RANDOM_STATE,
            stratify=df["label_id"]
        )
    except ValueError:
        tr, va = train_test_split(df, test_size=0.15, random_state=RANDOM_STATE)

    vocab = SimpleVocab().fit(tr["text"].tolist())

    tr_ds = CNNDataset(tr["text"].tolist(), tr["label_id"].tolist(), vocab)
    va_ds = CNNDataset(va["text"].tolist(), va["label_id"].tolist(), vocab)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    model     = TextCNN(vocab.size, num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    best_f1, best_state = 0.0, None

    for epoch in range(EPOCHS):
        # ── train ──
        model.train()
        for ids, labels in tqdm(tr_dl, desc=f"  {col[:30]} ep{epoch+1}", leave=False):
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(ids), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── validate ──
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for ids, labels in va_dl:
                ids = ids.to(device)
                preds.extend(model(ids).argmax(1).cpu().tolist())
                targets.extend(labels.tolist())

        f1 = f1_score(targets, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1   = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  [OK] {col:<45} val macro-F1 = {best_f1:.4f}  classes={num_labels}")
    return ColumnResult(col, best_f1, id_to_label, model, vocab, label_to_id)


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_column(result: ColumnResult, texts: list[str]) -> list[str]:
    ds = InferenceDataset(texts, result.vocab)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    result.model.eval().to(device)
    preds = []
    with torch.no_grad():
        for ids in dl:
            ids = ids.to(device)
            preds.extend(result.model(ids).argmax(1).cpu().tolist())
    return [result.id_to_label[p] for p in preds]


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Load data ──
    print("Loading data …")
    train = pd.read_csv(TRAIN_PATH, dtype=str).fillna(NOT_APPLICABLE)

    # Build text_input if not already present
    if "text_input" not in train.columns:
        train["text_input"] = (
            "PXD: " + train["PXD"].astype(str) + "\n\n"
            + train["pub_text"].astype(str)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
              .str.slice(0, 16_000)
        )

    # ── Load test set ──
    if TEST_PATH.exists():
        test = pd.read_csv(TEST_PATH, dtype=str).fillna("")
        if "text_input" not in test.columns:
            test["text_input"] = (
                "PXD: " + test["PXD"].astype(str) + "\n\n"
                + test["pub_text"].astype(str)
                  .str.replace(r"\s+", " ", regex=True)
                  .str.strip()
                  .str.slice(0, 16_000)
            )
        test_texts = test["text_input"].tolist()
        print(f"Test rows: {len(test):,}")
    else:
        print(f"WARNING: test file not found at {TEST_PATH}.")
        print("  Running validation-only mode on 20% of train split.")
        train, test_df = train_test_split(train, test_size=0.2,
                                          random_state=RANDOM_STATE)
        test = test_df.reset_index(drop=True)
        test_texts = test["text_input"].tolist()

    print(f"Train rows: {len(train):,}\n")

    # ── Train one model per Tier 1 column ──
    results: dict[str, ColumnResult] = {}
    available_cols = [c for c in TIER1_COLS if c in train.columns]

    for col in available_cols:
        result = train_column(col, train)
        if result is not None:
            results[col] = result

    # ── Build submission dataframe ──
    submission = pd.DataFrame(index=range(len(test)))

    # Carry over any ID columns present in test
    for id_col in ["row_id", "PXD"]:
        if id_col in test.columns:
            submission[id_col] = test[id_col].values

    for col, result in results.items():
        print(f"Predicting {col} …")
        submission[col] = predict_column(result, test_texts)

    # Columns we didn't model → fill with NOT_APPLICABLE as placeholder
    missing_cols = [c for c in TIER1_COLS if c not in submission.columns]
    for col in missing_cols:
        submission[col] = NOT_APPLICABLE

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission saved → {SUBMISSION_PATH}")
    print(f"Shape: {submission.shape}")
    print("\nPer-column validation F1:")
    for col, res in results.items():
        print(f"  {col:<45} {res.val_f1:.4f}")

    overall = np.mean([r.val_f1 for r in results.values()])
    print(f"\n  Mean Tier-1 macro-F1: {overall:.4f}")


if __name__ == "__main__":
    main()
