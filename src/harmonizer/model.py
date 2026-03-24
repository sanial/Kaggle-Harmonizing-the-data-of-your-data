"""
Transformer-based model for SDRF extraction.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SDRFTransformer(nn.Module):
    """
    Transformer-based model for SDRF metadata extraction.
    
    Uses a pre-trained language model (e.g., BERT) as the backbone
    with task-specific heads for SDRF field extraction.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        num_labels: int = 20,
        dropout: float = 0.1,
    ):
        """
        Args:
            model_name: Hugging Face model identifier
            hidden_size: Hidden dimension size
            num_labels: Number of SDRF field types
            dropout: Dropout probability
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained model
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for SDRF extraction.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs for BERT-style models
            
        Returns:
            Logits for each token [batch_size, seq_length, num_labels]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Use [CLS] token representation (first token)
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout and classify
        dropped = self.dropout(sequence_output)
        logits = self.classifier(dropped)
        
        return logits
    
    @classmethod
    def from_pretrained(cls, path: str) -> "SDRFTransformer":
        """
        Load a saved model from disk.
        
        Args:
            path: Path to saved model directory
            
        Returns:
            Loaded SDRFTransformer instance
        """
        # Implementation depends on how you save the model
        raise NotImplementedError("Load model from checkpoint")
    
    def get_tokenizer(self):
        """Get the tokenizer for this model."""
        return AutoTokenizer.from_pretrained(self.model_name)
