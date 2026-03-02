# Resource-Constrained Stackelberg Games for Auditing AI PRs

This project implements the core logic for the paper: **"Strategic Attention in AI-Generated Code Review: A Resource-Constrained Stackelberg Game for Small Language Models"**.

## 🚀 Overview

As AI-generated code floods repositories, manual review is bottlenecked. Even using LLMs for auditing is constrained by token budgets and context windows. This tool uses **Game Theory (Stackelberg Security Games)** to optimally allocate a limited token budget across "vulnerability-prone" chunks of code in a Pull Request.

### Key Components

1.  **Risk Profiler**: Analyzes code structure and keywords to assign payoffs ($U_d, L_d$).
2.  **Stackelberg LP Solver**: Formulates the review as a leader-follower game to find the optimal chunk selection strategy.
3.  **SLM Audit Agent**: (Mocked by default) Simulates the vulnerability detection of a sub-70B model.
4.  **Evaluator**: Compares the SSG strategy against Sequential and Random baselines.

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📈 Running the Experiment

To run the full pipeline and generate plots:

```bash
python main.py
```

Results (CSV and PNG charts) will be saved in the `results/` directory.

## 📓 Running on Kaggle

1.  Create a new Python Notebook on Kaggle.
2.  Upload the `kaggle_run.ipynb` or copy-paste the sections.
3.  Set "Internet" to "On" to fetch datasets from HuggingFace.
4.  The notebook will clone/setup the environment and run the benchmark.

## 📂 Project Structure

- `src/config.py`: Configuration and hyperparameters.
- `src/data_loader.py`: Dataset fetching (Devign/BigVul).
- `src/risk_profiler.py`: Code chunking and heuristic risk scoring.
- `src/solver.py`: Stackelberg Linear Programming logic.
- `src/slm_agent.py`: Interface for SLM audits.
- `src/evaluate.py`: The evaluation pipeline core.
- `main.py`: Entry point and charting.
