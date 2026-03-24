# Harmonizing the Data of Your Data

**Metadata Extraction from Mass Spectrometry Proteomics Publications**

A Kaggle competition focused on building text mining systems to automatically generate Sample and Data Relationship Format (SDRF) metadata from scientific publications.

- **Competition Link**: https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data
- **Prize Pool**: $10,000 (1st: $5,000 | 2nd: $3,000 | 3rd: $2,000)
- **Authorship Opportunity**: Co-author consideration for scientific publication. 

## Table of Contents

- [About the Competition](#about-the-competition)
- [Evaluation Metric](#evaluation-metric)
- [Using the Scoring Function](#using-the-scoring-function)
- [How to Submit](#how-to-submit)
  - [Option 1: Submit via Kaggle (Zip File)](#option-1-submit-via-kaggle-zip-file)
  - [Option 2: Contribute via Pull Request](#option-2-contribute-via-pull-request)
- [Repository Structure](#repository-structure)
- [Examples](#examples)

## About the Competition

### Problem Statement

Scientific knowledge is disseminated through research articles where experimental metadata, sample types, conditions, and analytical methods are described in natural language. The **Sample Data Relationship Format (SDRF)** is a standardized, machine-readable format that describes these experimental details.

**Challenge**: Most published studies lack complete or consistent SDRF annotations, preventing large-scale data integration and AI-driven discovery.

**Your Goal**: Build a solution that can read scientific papers and automatically extract and structure experimental information into valid SDRF metadata. Your pipeline can use any approach: rule-based systems, LLM prompting, classical NLP, or fine-tuned models.

### Use Case: Proteomics

Proteomics is the comprehensive study of proteins—their structures, functions, and interactions. Understanding proteins is essential for:
- Advancing medicine and biotechnology
- Understanding disease mechanisms
- Accelerating experimental research

A successful solution will revolutionize our ability to perform large-scale comparative analyses of scientific data by removing the barrier to accessing structured experimental metadata.

## Evaluation Metric

Submissions are scored using a **macro-averaged F1 score** comparing your predicted SDRF to the gold standard across all SDRF fields.

### Scoring Details

The evaluation uses:
- **Primary Metric**: Macro-averaged F1 score per SDRF field
- **Clustering**: Text values are clustered using string similarity (difflib) with agglomerative clustering
- **Ranking**: Leaderboard ranks by macro-F1 on the held-out test set
- **Validation**: Submissions failing schema or validation checks receive a score of 0.00

## Using the Scoring Function

This repository includes the expanded scoring function used in the Kaggle leaderboard. Use it locally to evaluate your submissions before uploading to Kaggle.

### Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/Kaggle-Harmonizing-the-data-of-your-data.git
cd Kaggle-Harmonizing-the-data-of-your-data

# Install dependencies
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

### Quick Start

#### End-to-End Pipeline (Train → Infer → Score)

Run the full workflow with one command:

```bash
.venv\Scripts\python.exe scripts/quickstart.py --device cpu
```

Useful variants:

```bash
# Reuse an existing model checkpoint
.venv\Scripts\python.exe scripts/quickstart.py --skip-train --device cpu

# Run train + infer only
.venv\Scripts\python.exe scripts/quickstart.py --skip-score --device cpu
```

Default outputs:
- `outputs/predictions/submission.csv`
- `outputs/metrics/detailed_metrics.csv`

The scoring function is located in [src/Scoring.py](src/Scoring.py).

#### Command Line Usage

```bash
python src/Scoring.py \
  --solution path/to/solution.csv \
  --submission path/to/submission.csv \
  --output detailed_metrics.csv
```

**Arguments:**
- `--solution`: Path to the gold-standard SDRF CSV file
- `--submission`: Path to your predicted SDRF CSV file
- `--output`: (Optional) Path to save detailed evaluation metrics (default: `detailed_evaluation_metrics.csv`)

**Output:**
- Prints the detailed evaluation metrics dataframe
- Prints the final SDRF Average F1 Score
- Saves detailed metrics to CSV file

#### Python API Usage

```python
import pandas as pd
from src.Scoring import score

# Load your dataframes
solution_df = pd.read_csv('path/to/solution.csv')
submission_df = pd.read_csv('path/to/submission.csv')

# Compute score
eval_df, final_score = score(solution_df, submission_df, row_id_column_name='ID')

print(f"Final Score: {final_score:.6f}")
print(eval_df)
```

### Key Functions

#### `score(solution, submission, row_id_column_name)`

Main scoring function that compares a submission against the gold standard.

**Parameters:**
- `solution` (pd.DataFrame): Gold-standard SDRF dataframe
- `submission` (pd.DataFrame): Predicted SDRF dataframe
- `row_id_column_name` (str): Name of the row ID column (will be dropped before scoring)

**Returns:**
- Tuple of `(eval_df, final_score)` where:
  - `eval_df`: Detailed metrics per PXD and annotation type
  - `final_score`: Single float in [0, 1]

#### `Harmonize_and_Evaluate_datasets(A, B, threshold=0.80)`

Harmonizes two SDRF datasets by clustering similar text values and computing precision, recall, F1, and Jaccard scores.

**Parameters:**
- `A`, `B` (Dict): SDRF data as nested dictionaries (keyed by PXD then column)
- `threshold` (float): Similarity threshold for clustering (default: 0.80)

**Returns:**
- `(harmonized_A, harmonized_B, eval_df)`: Cluster-mapped data and evaluation metrics

### CSV Format Requirements

Your submission must follow the **SampleSubmission.csv** format exactly. Each row represents a data file with its associated experimental metadata.

**Required Columns:**
- **`ID`**: Unique identifier for each row (will be dropped during scoring)
- **`PXD`**: Publication identifier (e.g., "PXD004010")
- **`Raw Data File`**: Associated raw data file name
- **`Characteristics[*]`**: SDRF characteristic annotations (organism, disease, cell type, etc.)
- **`Comment[*]`**: Technical metadata comments (instrument, MS analyzer, fragmentation method, etc.)
- **`FactorValue[*]`**: Experimental factor values (treatment, temperature, etc.)
- **`Usage`**: Specifies how the sample/file is used

**Key Notes:**
- Fill cells with the extracted metadata values or leave blank if unknown
- All rows must be aligned with the same column structure as SampleSubmission.csv
- "Text Span" in the template indicates where you should fill in actual values
- The scoring function uses the `PXD`, `ID`, and all metadata columns for comparison

**Example:**
```csv
ID,PXD,Raw Data File,Characteristics[Organism],Characteristics[Disease],FactorValue[Treatment],Usage
1,PXD004010,ad_pl01.raw,Homo sapiens,cancer,control,Raw Data File
2,PXD004010,ad_pl02.raw,Homo sapiens,cancer,treated,Raw Data File
```

Refer to [data/SampleSubmission.csv](data/SampleSubmission.csv) for the complete column structure with all required fields.

## How to Submit

### Option 1: Submit via Kaggle (Zip File)

Submit your work directly to the Kaggle competition. This is the main submission method along with submitting a submission.csv for scoring on the leaderboard.

**Package your submission as a ZIP file containing:**

1. **`submission.csv`** - Your predicted SDRF following the SampleSubmission.csv format
2. **Minimal Working Example** - Code demonstrating your complete pipeline

**Structure:**
```
my_submission.zip
├── submission.csv
├── pipeline.py (or notebook)
├── requirements.txt
└── README.md (optional: explain your approach)
```

**Steps:**
1. Go to the [competition page](https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data)
2. Click "Make a Submission"
3. Upload your ZIP file
4. Your submission will be scored and ranked on the leaderboard

### Option 2: Contribute via Pull Request

**We strongly encourage sharing your complete pipeline with the community!** Contributing via pull request allows others to build upon your work, learn from your approach, and potentially improve the overall solution.

#### Benefits of Contributing

✅ Share your solution with the broader community  
✅ Get credited for your approach  
✅ Help others improve their pipelines  
✅ Build a portfolio of open-source contributions  
✅ Potentially increase visibility for co-authorship consideration  

#### Step-by-Step Contribution Guide

##### Step 1: Fork the Repository

Click the "Fork" button on GitHub to create your own copy of the repository.

```bash
# Clone your fork locally
git clone https://github.com/YOUR-USERNAME/Kaggle-Harmonizing-the-data-of-your-data.git
cd Kaggle-Harmonizing-the-data-of-your-data
```

##### Step 2: Create a Feature Branch

Create a new branch for your contribution with a descriptive name:

```bash
git checkout -b feat/your-pipeline-name

# Example names:
# feat/gpt4-prompt-based-extraction
# feat/finetuned-bert-sdrf
# feat/rule-based-parser
# feat/ensemble-hybrid-approach
```

##### Step 3: Create Your Submission Folder

Each submission should be in a dedicated folder within `Submissions/`:

```bash
mkdir -p Submissions/your-team-or-name

# Create the following structure:
Submissions/your-team-or-name/
├── pipeline.py (or .ipynb - main implementation)
├── submission.csv (your best submission results)
├── requirements.txt (Python dependencies)
├── README.md (documentation of your approach)
└── data/ (optional: training data, models, etc.)
```

##### Step 4: Implement Your Solution

**`pipeline.py` (or Jupyter Notebook)**

Your main implementation should:
- Load the training and test data
- Implement your metadata extraction approach
- Generate the submission CSV
- Be executable as a minimal working example

```python
# Example structure:
import pandas as pd
from src.Scoring import score

def extract_sdrf(paper_text):
    """Your extraction logic here"""
    pass

def main():
    # Load data
    papers = load_papers()
    
    # Extract metadata
    predictions = [extract_sdrf(p) for p in papers]
    
    # Generate submission
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv('submission.csv', index=False)
    
    # (Optional) Score locally if you have test labels
    if has_test_labels():
        solution_df = pd.read_csv('test_solution.csv')
        eval_df, score = score(solution_df, submission_df, 'ID')
        print(f"Local F1 Score: {score:.6f}")

if __name__ == "__main__":
    main()
```

**`requirements.txt`**

List all Python dependencies:

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.10.0
transformers>=4.0.0
# Add any other libraries your pipeline needs
```

**`README.md`**

Document your approach:

```markdown
# Your Solution Name

Brief description of your approach.

## Method

Explain your pipeline:
- Data preprocessing
- Feature extraction/engineering
- Model architecture or rules
- Post-processing
- Key hyperparameters

## Results

- Local F1 Score: X.XXX
- Kaggle Score: X.XXX
- Key findings

## Installation & Usage

```bash
pip install -r requirements.txt
python pipeline.py
```

## References

- Papers you cited
- Datasets you used
- Inspiration from other work
```

##### Step 5: Test Your Code

Ensure your minimal working example runs correctly:

```bash
# Install dependencies
pip install -r Submissions/your-team-or-name/requirements.txt

# Test your pipeline
python Submissions/your-team-or-name/pipeline.py

# (Optional) Test the scoring function
python src/Scoring.py \
  --solution test_data/solution.csv \
  --submission Submissions/your-team-or-name/submission.csv
```

##### Step 6: Commit and Push

Commit your changes with clear, descriptive messages:

```bash
# Stage your changes
git add Submissions/your-team-or-name/

# Commit with a descriptive message
git commit -m "feat: Add GPT-4 based SDRF extraction pipeline

- Implements few-shot prompting strategy
- Includes validation and error handling
- Local F1 score: 0.85"

# Push to your fork
git push origin feat/your-pipeline-name
```

##### Step 7: Create a Pull Request

1. Go to the original repository: https://github.com/[original-owner]/Kaggle-Harmonizing-the-data-of-your-data
2. Click "Pull Requests" → "New Pull Request"
3. Click "compare across forks"
4. Select your fork and branch
5. Click "Create Pull Request"

**PR Title:** Keep it concise
```
[Submission] Your Pipeline Name - Brief Description
```

**PR Description:** Provide context for reviewers

```markdown
## Description
Brief explanation of your approach

## Method
- Key techniques used
- Model/algorithm details
- Notable features

## Results
- Kaggle leaderboard score
- Local evaluation results
- Performance metrics

## Files Added
- `pipeline.py` - Main implementation
- `requirements.txt` - Dependencies
- `submission.csv` - Results file

## References
- Relevant papers
- Data sources
- Inspiration

## Checklist
- [ ] Code runs without errors
- [ ] Dependencies listed in requirements.txt
- [ ] README documents the approach
- [ ] Submission file is properly formatted
```

##### Step 8: Address Feedback

Maintainers may request changes:
- Make edits locally on the same branch
- Commit and push again
- The PR will automatically update

```bash
# Make updates to your code
git add .
git commit -m "fix: Improve preprocessing logic"
git push origin feat/your-pipeline-name
```

#### Submission Guidelines

**Do's:**
- ✅ Write clean, readable code with comments
- ✅ Include a working minimal example
- ✅ Document your approach clearly
- ✅ Test your code before submitting
- ✅ Include a realistic requirements.txt
- ✅ Add proper docstrings and type hints
- ✅ Consider edge cases and error handling

**Don'ts:**
- ❌ Large binary files (>10MB) - use .gitignore
- ❌ Training data or large models (reference external sources instead)
- ❌ Undocumented or hard-to-follow code
- ❌ Broken or incomplete pipelines
- ❌ Duplicate submissions (check existing ones first)

## Repository Structure

```
.
├── README.md                          # This file
├── LICENSE                            # Repository license
├── src/
│   └── Scoring.py                     # Competition scoring function
├── data/
│   ├── SampleSubmission.csv           # Template for submission format
│   ├── TrainingPubText/               # Training set publication texts (JSON)
│   ├── TrainingSDRFs/                 # Gold-standard SDRF annotations (CSV)
│   ├── TestPubText/                   # Test set publication texts (JSON)
│   └── detailed_evaluation_metrics.csv # Example evaluation output
├── detailed_evaluation_metrics.csv    # Example evaluation output
└── Submissions/                       # Community contributions
    ├── example-submission-1/
    │   ├── pipeline.py
    │   ├── submission.csv
    │   ├── requirements.txt
    │   └── README.md
    ├── example-submission-2/
    │   └── ...
    └── your-submission/
        └── ...
```

### Data Directory

The `data/` folder contains essential files for the competition:

- **`SampleSubmission.csv`**: Template showing the exact format your submission must follow. All columns and structure must match this file.
- **`TrainingPubText/`**: Collection of training publication texts stored as JSON files. Each file contains the full text of a scientific publication.
- **`TrainingSDRFs/`**: Gold-standard SDRF annotations for the training set stored as CSV files. Use these to train and validate your model.
- **`TestPubText/`**: Collection of test publication texts (JSON). Your model must extract SDRF metadata from these publications.
- **`detailed_evaluation_metrics.csv`**: Example output from running the scoring function, showing detailed metrics per publication and annotation type.

## Examples

### Running the Scorer Locally

```bash
# Generate detailed metrics and save to CSV
python src/Scoring.py \
  --solution data/gold_standard_sdrf.csv \
  --submission my_predictions.csv \
  --output my_detailed_metrics.csv

# Output example:
# pxd  AnnotationType  precision  recall  f1      jacc
# PXD1 Organism        0.95       0.92    0.935   0.88
# PXD1 Disease         0.88       0.91    0.895   0.82
# PXD2 Treatment       0.92       0.89    0.905   0.84
# Final SDRF Average F1 Score: 0.9117
```

### Example Submission Structure

See [Submissions/](Submissions/) folder for example community contributions once they're added.

## Additional Resources

- **Competition Page**: https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data
- **Co-author Form**: https://docs.google.com/forms/d/e/1FAIpQLSeyqKmmTjHmTDPzkX2WBy7mUuUS68eO0ITkdC1Z5v5QiiRlpg/viewform
- **SDRF Standard**: https://www.ebi.ac.uk/ols/ontologies/msi
- **Proteomics Resources**: https://www.ncbi.nlm.nih.gov/pmc/

## Questions?

- Post in the [Kaggle competition discussions](https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data/discussion)
- Check existing GitHub issues
- Reach out to the organizers

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NSF National Center for the Emergence of Molecular and Cellular Sciences (NSF-NCEMS)**
- **Penn State Institute for Computational and Data Sciences (ICDS)**
- **The Huck Institutes of the Life Sciences**

## Citation

If you use this repository or competition framework, please cite:

```bibtex
@misc{sitarik2026harmonizing,
  title={Harmonizing the Data of Your Data},
  author={Sitarik, Ian and Friedberg, Iddo and Claeys, Tine and Bittremieux, Wout},
  year={2026},
  howpublished={\url{https://kaggle.com/competitions/harmonizing-the-data-of-your-data}}
}
```

---

**Happy competing! 🚀** We look forward to your innovative solutions and community contributions!
