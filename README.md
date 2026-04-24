# Kaggle — Harmonizing the Data of Your Data

**Competition:** EBI Proteomics SDRF Metadata Extraction  
**Best Public Score:** 0.27273 (Notebook 16, OLS Entity Linking Pipeline)  
**Leaderboard Position:** 89/250  

---

## Problem Statement

Proteomics research papers describe experimental metadata in natural language, sample 
types, conditions, organisms, instruments, and analytical methods buried in dense 
scientific prose. The Sample and Data Relationship Format (SDRF) is the standardized 
machine-readable format that captures this information, but most published studies lack 
complete or consistent SDRF annotations. This prevents large-scale data integration and 
AI-driven discovery across proteomics datasets.

The competition task was to build a pipeline that reads scientific papers and automatically 
extracts and structures experimental information into valid SDRF metadata across 77 columns, 
scored via macro-averaged F1 against ground truth ontology strings.

## What I Built

A pipeline for extracting 77 SDRF metadata columns from proteomics research papers, scored via entity overlap F1 against ground truth ontology strings. The task requires not just extracting the right value from paper text, but producing it in the exact canonical format the scoring function expects which means normalization discipline matters as much as extraction recall.

---

## The Approach That Worked

The highest-scoring submission uses a six-layer priority pipeline that defers to authoritative sources rather than generating answers:

**Layer 1 — Training SDRF Overlap**  
Ground truth values from the competition's own training set take absolute priority.

**Layer 2 — PRIDE API Enrichment**  
Authoritative biological metadata fetched directly from PRIDE, normalized against the Ontology Lookup Service with LRU caching so each unique term is only queried once.

**Layer 3 — PX XML Backup**  
Project XML title and description used as a fallback when the PRIDE API returns nothing useful.

**Layer 4 — Regex Extraction**  
Pattern matching for tissues, instruments, and protocol parameters when structured sources fail.

**Layer 5 — Filename Parsing**  
Raw mass spec filenames often encode methodology directly. Used for the five test PXDs that had no paper text available.

**Layer 6 — Conservative Majority Fallback**  
Only triggered when a value appears in more than 80% of files for a given experiment. Excluded entirely for experiment-specific columns where majority logic does not apply.

---

## What Didn't Work

Early iterations used PubMedBERT for entity extraction. In practice it consistently overwrote authoritative PRIDE API values with confident wrong predictions, a failure mode that hurt more than it helped in a schema-strict scoring environment. The clearest lesson: when domain APIs provide ground truth, the pipeline should defer to them rather than let a model override them.

Notebook 11 used hand-coded ontology dictionaries with 65 tissue entries, 30 instrument entries, and 26 cleavage agent synonyms. It scored reasonably well but missed format variants automatically. "Blood Serum", "blood serum", and "EDTA plasma" all needed to resolve to the same UBERON node (`NT=blood serum;AC=UBERON:0001977`). Notebook 16 replaced these static dictionaries with live OLS4 API queries, resolving terms against the same ontologies the competition scoring was built on.

One thing I would do differently: study prior winning solutions earlier. The Coleridge Initiative competition tackled similar scientific entity extraction problems, those pipelines would have pointed toward authoritative source alignment weeks before I found it through trial and error.

---

## Notebook Progression

22 notebooks across six development phases. See [notebooks/README.md](notebooks/README.md) for the full catalog with descriptions of each approach, what it tried, and what it produced.

The submitted notebook is [16_ols_entity_linking_pipeline.ipynb](notebooks/16_ols_entity_linking_pipeline.ipynb).

---

## Tech Stack

Python, pandas, requests, EBI OLS4 API, PRIDE API, ProteomeXchange XML, regex, LRU caching

---

## Competition Details

For full competition details, scoring function documentation, and submission guidelines see [COMPETITION.md](COMPETITION.md).

**Organizers:** NSF NCEMS, Penn State ICDS, The Huck Institutes  
**Prize Pool:** $10,000  
**Scoring:** Macro-averaged F1 across all SDRF fields  
**Competition Link:** https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data
