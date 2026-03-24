# Transformer-based SDRF Extraction

## Overview

This submission uses a fine-tuned transformer model (BERT-based) to extract SDRF metadata from proteomics publications.

## Method

### Approach
- **Model**: Pre-trained BERT encoder with task-specific classification heads
- **Task**: Sequence token classification for SDRF field extraction
- **Strategy**: Fine-tune on training SDRF annotations then apply to test publications

### Architecture
```
Input Text
    ↓
Tokenizer (WordPiece)
    ↓
BERT Encoder (12 layers, 768 hidden)
    ↓
Dropout (0.1)
    ↓
Classification Head (num_labels outputs)
    ↓
Predicted SDRF Fields
```

### Key Components
1. **Data Loading** (`data.py`): Handles JSON publication texts and SDRF annotations
2. **Model** (`model.py`): BERT wrapper with token classification head
3. **Training** (`train.py`): Fine-tuning loop with validation
4. **Inference** (`inference.py`): Prediction and submission generation

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
# From project root
python scripts/train.py --config configs/training.yaml --device cuda
```

### Inference
```bash
# Generate predictions
python scripts/infer.py --model outputs/models/transformer_sdrf --output predictions.csv

# Or from submission directory
python pipeline.py --model ../../outputs/models/transformer_sdrf
```

### Configuration
Edit `configs/training.yaml` to adjust:
- Model name (e.g., `bert-base-uncased`, `RobertaForSequenceClassification`)
- Learning rate, batch size, epochs
- Token max length (512 for BERT)
- Output paths

## Results

| Metric | Score |
|--------|-------|
| Local F1 | 0.XX |
| Kaggle F1 | 0.XX |

## Files

- `pipeline.py` - Minimal working example (reproducible inference)
- `requirements.txt` - Python dependencies  
- `submission.csv` - Generated predictions
- `README.md` - This file

## References

- [Transformers](https://huggingface.co/transformers/) - Hugging Face library
- [SDRF Standard](https://www.ebi.ac.uk/ols/ontologies/msi) - MS metadata ontology
- [Competition](https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data)

## Notes

- Large models (>10MB) are gitignored; store in `outputs/models/`
- Training logs saved to `outputs/logs/`
- All predictions use the same SDRF column structure as `SampleSubmission.csv`
