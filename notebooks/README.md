# Notebooks

This folder is for exploratory analysis and prototyping.

## Workflow

1. **EDA** (`01_eda.ipynb`): Load data, inspect structure, identify patterns
2. **Preprocessing** (`02_preprocessing.ipynb`): Clean text, handle missing values
3. **Baseline** (`03_baseline.ipynb`): Simple heuristics to understand task
4. **Model Development** (`04_model_development.ipynb`): Experiment with architectures

## Guidelines

- Keep notebooks **focused** on one task each
- Move reusable code to `src/harmonizer/` modules
- Document findings and export clean code before submission
- Use version control for important notebooks

## Best Practices

- Add markdown cells explaining your approach
- Include comments in code cells
- Store trained models in `outputs/models/`, not notebooks
- Don't commit large datasets or models
- Use relative paths from project root
