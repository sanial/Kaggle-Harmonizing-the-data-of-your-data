# Notebooks

This folder contains all exploratory, prototyping, and pipeline notebooks developed during the competition.

## The Journey
This competition required extracting 77 SDRF metadata columns from proteomics research papers, scored via entity overlap F1 against ground truth ontology strings. The pipeline evolved across 22 notebooks over several weeks.
Early phases established a baseline with heuristic rules and majority fallback strategies. Then introduced BioBERT and PubMedBERT for named entity recognition, which initially seemed promising. In practice, both models consistently overwrote authoritative values from the PRIDE API with confident wrong predictions, a failure mode that hurt more than it helped in a schema-strict scoring environment where format precision matters as much as extraction recall.
The key insight came late: the competition scoring was built on the same ontologies served by EBI's Ontology Lookup Service. Instead of training models to guess canonical strings, notebook 16 queries OLS directly at runtime, resolving extracted terms against the authoritative source. This alignment between the pipeline's normalization source and the scoring function's ground truth produced the highest leaderboard score.
A secondary insight was finding the Coleridge Initiative competition too late — a prior Kaggle competition tackling similar scientific entity extraction problems whose winning solutions would have pointed toward authoritative source alignment from week one.
Best public score: 0.27273 (notebook 16)
## ⭐ Submitted Notebook

**`16_ols_entity_linking_pipeline.ipynb`** — This is the final notebook submitted to the Kaggle competition. It implements an OLS (Ontology Lookup Service) entity linking pipeline that combines ontology-based entity resolution with structured extraction to generate SDRF metadata from proteomics publications.

---

## Full Notebook Catalog

### Phase 1 — Exploration & Baseline

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Exploratory data analysis: loads training publications and SDRF files, inspects structure, analyses column distributions and missing values |
| `02_preprocessing.ipynb` | Text cleaning and feature preparation: tokenisation, normalisation, and building a preprocessed dataset saved to `outputs/` |
| `03_baseline.ipynb` | Simple heuristic/rule-based baseline to establish a lower bound on performance |
| `04_model_development.ipynb` | Initial model experiments: tests multiple architectures and feature representations |

### Phase 2 — Early Submissions

| Notebook | Description |
|----------|-------------|
| `05_tier1_submission.ipynb` | First competition submission pipeline; output saved as `outputs/submission_tier1.csv` |
| `06_majority_fallback.ipynb` | Majority-vote fallback strategy for fields where models are uncertain; output: `outputs/submission_with_fallback.csv` |
| `07_llm_extraction.ipynb` | Prompt-based LLM metadata extraction using the baseline prompt from `data/BaselinePrompt.txt` |
| `08_final_submission (2).ipynb` | Revised submission after initial feedback (iteration 2) |

### Phase 3 — NER & Ontology Pipelines

| Notebook | Description |
|----------|-------------|
| `10_pride_regex_biobert.ipynb` | Combines PRIDE-specific regex rules with BioBERT NER; output: `outputs/submission_pride_regex_biobert.csv` |
| `11_per_file_ontology_pipeline.ipynb` | Per-file ontology lookup pipeline (initial version) |
| `11_per_file_ontology_pipeline_v2.ipynb` | Improved per-file ontology pipeline with better field coverage; output: `outputs/submission_v2_per_file.csv` |
| `13_combined_regex_llm_pipeline.ipynb` | Hybrid pipeline combining regex rules and LLM-generated annotations |
| `14_scispacy_v17_pipeline.ipynb` | SciSpaCy NER pipeline (v17); output: `outputs/submission_v17.csv` |

### Phase 4 — Fine-tuned Models

| Notebook | Description |
|----------|-------------|
| `15_pubmedbert_finetune_pipeline.ipynb` | Fine-tunes PubMedBERT per-field on training SDRFs; model checkpoints written to `outputs/pubmedbert_models/`; output: `outputs/submission_pubmedbert.csv` |

### Phase 5 — OLS Entity Linking (Final)

| Notebook | Description |
|----------|-------------|
| **`16_ols_entity_linking_pipeline.ipynb`** | **⭐ SUBMITTED.** Full OLS entity linking pipeline: extracts candidate terms from publication text, resolves them against ontology terms via the OLS API, applies field-specific heuristics, and outputs a compliant SDRF CSV. Output: `outputs/submission_ols.csv` |
| `16_v2_ols_entity_linking_pipeline.ipynb` | Extended v2 variant with additional ontology coverage; output: `outputs/submission_v2_ols.csv` |

### Phase 6 — Advanced Pipelines

| Notebook | Description |
|----------|-------------|
| `17_graph_pipeline.ipynb` | Graph-based entity resolution using co-occurrence and similarity edges; output: `outputs/submission_graph.csv` |
| `17_mdc_hybrid_pipeline.ipynb` | MDC hybrid NER pipeline combining multiple named-entity classifiers |
| `18_semantic_similarity_pipeline.ipynb` | Semantic similarity matching between extracted spans and ontology labels using sentence embeddings; output: `outputs/submission_semantic.csv` |
| `20_biobert_ols_pipeline.ipynb` | BioBERT NER combined with OLS entity linking; output: `outputs/submission_biobert_ols.csv` |
| `21.ipynb` / `21_modified (1).ipynb` | Iteration 21 and a modified variant exploring additional field improvements |
| `22_pubmedbert_ols_pipeline.ipynb` | PubMedBERT combined with OLS entity linking; output: `outputs/submission_pubmedbert_ols.csv` (if generated) |

---

## Supporting Scripts

| File | Description |
|------|-------------|
| `prefetch_caches.py` | Pre-fetches and serialises OLS and PRIDE API lookup caches to speed up notebook runs |

---

## Guidelines

- Keep notebooks **focused** on one task each
- Move reusable code to `src/harmonizer/` modules
- Document findings with markdown cells
- Use relative paths from the project root
- Large binary outputs (models, caches) are git-ignored — see `.gitignore`
