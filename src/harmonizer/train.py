"""
Training loop for SDRF transformer model.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import SDRFDataset, load_data
from .model import SDRFTransformer


def train_model(
    config_path: str = "configs/training.yaml",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Path:
    """
    Train the SDRF extraction model.
    
    Args:
        config_path: Path to training configuration YAML
        device: PyTorch device (cuda/cpu)
        
    Returns:
        Path to saved model directory
    """
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Training on device: {device}")
    
    # Initialize model
    model = SDRFTransformer(
        model_name=config["model"]["model_name"],
        hidden_size=config["model"]["hidden_size"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
    )
    model.to(device)
    
    tokenizer = model.get_tokenizer()
    
    # Load data
    print("Loading data...")
    texts, labels, _ = load_data(
        train_path=config["data"]["train_path"],
        test_path=config["data"]["test_path"],
    )
    
    if not texts:
        print("Warning: No training data found. Using dummy data.")
        texts = ["dummy text"] * 10
        labels = [{}] * 10
    
    # Create dataset
    dataset = SDRFDataset(
        text_list=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=config["data"]["max_length"],
    )
    
    # Train/val split
    val_size = int(len(dataset) * config["data"]["val_split"])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    total_steps = len(train_loader) * config["training"]["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Compute loss (dummy for now)
            loss = outputs.mean()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["max_grad_norm"],
            )
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask)
                loss = outputs.mean()
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        model.train()
    
    # Save model
    model_path = Path(config["output"]["model_path"])
    model_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_path / "model.pt")
    print(f"\nModel saved to {model_path}")
    
    return model_path


if __name__ == "__main__":
    train_model()
