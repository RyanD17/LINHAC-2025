# LINHAC 2025 — Situational Goaltender Evaluation

A hockey analytics project exploring situational expected-goals differentials for goaltender evaluation in the Swedish Hockey League (SHL). The analysis combines event-level Sportlogiq data with sequence and Markov-chain analysis to examine how scoring opportunities develop, rather than treating every shot as independent.

## Project layout

- `src/` — reusable analysis code and the command-line pipeline
- `notebooks/` — numbered exploratory analyses in execution order
- `data/` — source Sportlogiq event data
- `assets/figures/` — figures generated for exploratory and model analysis
- `results/` — derived tables and written summaries
- `paper/` — final manuscript (when available)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.pipeline
```

The pipeline reads `data/Linhac24-25_Sportlogiq.csv`, writes derived tables to `results/goalie_index/`, and writes figures to `assets/figures/goalie_index/`.

## Data

The included Sportlogiq CSV is approximately 74 MB, below GitHub’s 100 MB single-file limit. Use of the data remains subject to the applicable Sportlogiq data license and access terms; do not redistribute it outside those terms.

## Notebook guide

1. `01_eda.ipynb` — initial data exploration
2. `02_advanced_eda.ipynb` — event, shot, and situational analysis
3. `03_markov_chain_analysis.ipynb` — sequence modelling
4. `04_passing_networks.ipynb` — passing-network analysis
5. `05_segd_evaluation.ipynb` — SEGD evaluation

## Reproducibility

The analyses target Python 3.10+. Generated artifacts are retained in the repository to make results reviewable, while source code and notebooks remain separated from data and figures.
