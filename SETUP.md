# Project Setup: Transformer-based SDRF Extraction

## Overview

This project structure is optimized for training transformer models for SDRF (Sample Data Relationship Format) metadata extraction from proteomics publications.

## Directory Structure

```
.
├── configs/
│   └── training.yaml          # Hyperparameters and paths (EDIT THIS)
├── data/
│   ├── SampleSubmission.csv   # Submission template
│   ├── TrainingPubText/       # Training publication texts (JSON)
│   ├── TrainingSDRFs/         # Training SDRF annotations (CSV)
│   └── TestPubText/           # Test publications (JSON)
├── notebooks/                 # Exploration and analysis
│   └── README.md              # Notebook workflow guide
├── outputs/                   # Generated artifacts
│   ├── logs/                  # Training logs
│   ├── metrics/               # Evaluation metrics
│   ├── models/                # Saved model checkpoints
│   └── predictions/           # Generated submission CSVs
├── scripts/                   # Entrypoint scripts
│   ├── train.py               # Training runner
│   └── infer.py               # Inference runner
├── src/
│   ├── Scoring.py             # Official competition scorer
│   └── harmonizer/            # Reusable pipeline package
│       ├── __init__.py
│       ├── data.py            # Data loading & preprocessing
│       ├── model.py           # Transformer model wrapper
│       ├── train.py           # Training loop
│       └── inference.py       # Prediction & submission generation
├── Submissions/
│   └── template/              # Submission template (copy for new approach)
│       ├── pipeline.py        # Minimal working example
│       ├── requirements.txt   # Dependencies
│       └── README.md          # Approach documentation
├── .gitignore                 # Excludes large models, logs, etc.
├── requirements.txt           # Project dependencies
├── LICENSE
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Training
Edit **`configs/training.yaml`** to set:
- Model name (e.g., `bert-base-uncased`, `roberta-base`)
- Learning rate, batch size, number of epochs
- Data paths
- Output directories

### 3. Train Model
```bash
python scripts/train.py --config configs/training.yaml --device cuda
```
- Saves model to `outputs/models/transformer_sdrf/model.pt`
- Logs saved to `outputs/logs/`

### 4. Generate Predictions
```bash
python scripts/infer.py --model outputs/models/transformer_sdrf
```
- Creates `outputs/predictions/submission.csv`

### 5. Score Submission (Optional)
```bash
python src/Scoring.py \
  --solution data/TrainingSDRFs/gold_standard.csv \
  --submission outputs/predictions/submission.csv \
  --output outputs/metrics/scores.csv
```

## Module Descriptions

### `src/harmonizer/`

**`data.py`** - Data loading and preprocessing
- `SDRFDataset`: PyTorch Dataset for SDRF task
- `load_data()`: Load training/test texts from JSON
- `load_sample_submission()`: Load template CSV

**`model.py`** - Transformer model definition
- `SDRFTransformer`: BERT-based token classification model
- Handles tokenization, embeddings, and classification heads
- Configurable for different transformer architectures

**`train.py`** - Training loop
- `train_model()`: Full training pipeline with validation
- Handles optimizer, scheduler, gradient clipping
- Saves checkpoints to disk
- Configure via YAML for reproducibility

**`inference.py`** - Prediction and submission
- `generate_submission()`: Load model and predict on test data
- Formats output as competition-required CSV
- Works with or without trained model (fallback to template)

### `scripts/`

**`train.py`** - Training entrypoint
```bash
python scripts/train.py --config configs/training.yaml --device cuda
```

**`infer.py`** - Inference entrypoint
```bash
python scripts/infer.py --model outputs/models/transformer_sdrf --output submission.csv
```

## Code Quality Standards

To meet submission guidelines, this project includes:

✅ **Clean, Documented Code**
- Docstrings on all functions and classes
- Type hints throughout
- Inline comments for complex logic

✅ **Minimal Working Example**
- `Submissions/template/pipeline.py` demonstrates full pipeline
- Executable without external setup
- Clear input/output format

✅ **Reproducibility**
- All hyperparameters in `configs/training.yaml`
- Fixed random seed
- Dependency versions pinned in `requirements.txt`

✅ **Error Handling**
- Try-catch blocks for file I/O
- Validation of data shapes
- Informative error messages

✅ **Large File Management**
- `.gitignore` excludes models (>10MB)
- Models stored in `outputs/models/` locally
- Provide download/checkpoint info in docs

## Workflow for New Experiments

When trying a new approach:

1. **Copy template directory:**
   ```bash
   cp -r Submissions/template Submissions/my-approach-name
   ```

2. **Modify configuration** in `configs/training.yaml`:
   ```yaml
   model:
     model_name: "roberta-base"  # Try different model
     num_labels: 25              # Adjust for your task
   ```

3. **Experiment in notebooks** (optional):
   ```bash
   # Create a notebook in notebooks/
   # Prototype and validate approach
   # Move stable code to src/harmonizer/
   ```

4. **Update pipeline code**:
   - Modify `src/harmonizer/model.py` for new architecture
   - Update `src/harmonizer/train.py` if using new loss function
   - Adjust `src/harmonizer/inference.py` for output formatting

5. **Test locally**:
   ```bash
   python scripts/train.py --config configs/training.yaml
   python scripts/infer.py --model outputs/models/transformer_sdrf
   ```

6. **Create submission**:
   ```bash
   # Copy final version to Submissions/ and test
   python Submissions/my-approach-name/pipeline.py
   ```

## Kaggle Submission Checklist

Before uploading:

- [ ] All code has docstrings and type hints
- [ ] `pipeline.py` runs without errors
- [ ] `requirements.txt` lists all dependencies with versions
- [ ] `README.md` explains your approach clearly
- [ ] `submission.csv` format matches `SampleSubmission.csv`
- [ ] No large model files (>10MB) in git
- [ ] No training data included (reference data/ folder)
- [ ] Test code locally before submission

## Useful Commands

```bash
# List all output artifacts
ls -la outputs/models outputs/predictions outputs/metrics

# Check model file size
du -sh outputs/models/*

# Run only on CPU
python scripts/train.py --device cpu

# Test submission format
python -c "import pandas as pd; df = pd.read_csv('outputs/predictions/submission.csv'); print(df.shape, df.columns.tolist()[:5])"

# Profile training speed
python -m cProfile -s cumtime scripts/train.py
```

## Troubleshooting

**Out of memory during training:**
- Reduce `training.batch_size` in `configs/training.yaml`
- Reduce `data.max_length` for shorter sequences

**Model not converging:**
- Try different learning rate (e.g., 1e-5, 5e-5)
- Increase `training.num_epochs`
- Check that training data is loaded

**Submission format errors:**
- Verify `submission.csv` has same columns as `SampleSubmission.csv`
- Check no NaN values (if required)
- Ensure ID column is unique

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SDRF Standard](https://www.ebi.ac.uk/ols/ontologies/msi)
- [Kaggle Competition](https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data)

---

**Ready to train!** Start with `scripts/train.py` and iterate from there.
