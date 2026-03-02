# Resource-Constrained Stackelberg Games for Auditing AI PRs

This repository accompanies the paper:

> **"Strategic Attention in AI-Generated Code Review: A Resource-Constrained Stackelberg Game for Small Language Models"**
> Submitted to ASE 2026 (double-blind)

## Overview

As AI-generated code floods repositories, manual review is bottlenecked. Even using LLMs for auditing is constrained by token budgets and context windows. This tool uses **Stackelberg Security Games (SSG)** to solve a minimax LP that optimally allocates a limited token budget across vulnerability-prone chunks of a Pull Request.

### Key Results (40 % token budget, Devign-100 benchmark)

| Strategy   | VDR    | F1     | Precision | Recall | Efficiency |
|------------|--------|--------|-----------|--------|------------|
| **SSG**    | **42%**| **0.47**| **0.68** | **0.42**| **1.05** |
| Sequential | 20%    | 0.29   | 0.50      | 0.20   | 0.50       |
| Random     | 14%    | 0.21   | 0.44      | 0.14   | 0.35       |

SSG achieves **2.1× higher VDR** vs Sequential (Wilcoxon p < 0.001, Cohen's d > 0.8).

### Key Components

1. **Risk Profiler** (`src/risk_profiler.py`): Chunks code and assigns payoffs (U_d, L_d) via structural heuristics.
2. **Stackelberg LP Solver** (`src/solver.py`): Minimax LP solved with `scipy.optimize.linprog` (HiGHS backend).
3. **SLM Audit Agent** (`src/slm_agent.py`): Mock detection model `P(detect|vuln) = min(0.95, 0.15 + 0.80 × risk)`.
4. **Evaluator** (`src/evaluate.py`): Cross-pool evaluation — defender selects from the full PR chunk pool.
5. **Significance** (`src/significance.py`): Bootstrap 95 % CI, Wilcoxon signed-rank, Cohen's d.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Experiment

One-command reproduction:

```bash
python main.py
```

The pipeline runs four stages:

| Stage | Description | Output |
|-------|-------------|--------|
| 1 | Single-budget evaluation (40 %) | `results/comparison_*.png` |
| 2 | Budget sweep (10 %–80 %) | `results/budget_sweep_*.png` |
| 3 | N=30 repeated runs + CI bars | `results/ci_bars_*.png` |
| 4 | Chunk-size ablation (6 sizes) | `results/ablation_chunk_size.png` |

All CSVs and PNGs are written to `results/`.

## Running on Kaggle

1. Open `kaggle_notebook.ipynb` on [Kaggle](https://kaggle.com).
2. Enable **Internet** (required to clone this repo and fetch Devign-100).
3. Run all cells. The full pipeline — including Steps 1–5 (single-budget, sweep, CI, ablation, radar chart) — takes ≈ 10 min on a Kaggle CPU instance.

## Project Structure

```
Stackelberg_Code_Review/
├── main.py                   # Entry point + all plot generation
├── requirements.txt
├── kaggle_notebook.ipynb     # Self-contained Kaggle notebook
├── data/
│   └── samples_devign_100.json
├── results/                  # Auto-generated CSVs and plots
└── src/
    ├── config.py             # Hyperparameters (CHUNK_TOKEN_SIZE, BUDGET_RATIO, …)
    ├── data_loader.py        # Devign / BigVul dataset loader
    ├── risk_profiler.py      # Code chunking & heuristic risk scoring
    ├── solver.py             # Stackelberg LP (SSG / Sequential / Random)
    ├── slm_agent.py          # Mock SLM vulnerability detector
    ├── evaluate.py           # Core evaluation pipeline + ablation + repeated runs
    └── significance.py       # Statistical significance helpers
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{stackelberg_code_review_2026,
  title     = {Strategic Attention in AI-Generated Code Review:
               A Resource-Constrained Stackelberg Game for
               Small Language Models},
  author    = {Anonymous},
  booktitle = {Proc.\ 41st IEEE/ACM International Conference on
               Automated Software Engineering (ASE)},
  year      = {2026}
}
```
