"""
evaluate.py
Runs the evaluation pipeline comparing:
  1. Sequential Review  (Baseline)
  2. Random Review      (Baseline)
  3. SSG-Optimized      (Proposed)

Metrics
-------
  VDR       – Vulnerability Detection Rate  = detected_vulns / total_vulns
  FPR       – False Positive Rate           = false_positives / total_clean
  Precision – TP / (TP + FP)
  Recall    – TP / (TP + FN)  (= VDR)
  F1        – harmonic mean of Precision & Recall
  Efficiency– detections per 1 K tokens spent
"""

import random as _random
import time
import os
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

from src.data_loader import load_samples, get_stats
from src.risk_profiler import profile_samples
from src.solver import select_chunks_ssg, select_chunks_sequential, select_chunks_random
from src.slm_agent import SLMAuditAgent
from src.config import BUDGET_RATIO, RESULTS_FILE, RESULTS_DIR

# ─── Strategies registry ──────────────────────────────────────────────────────

STRATEGIES = {
    "Sequential": select_chunks_sequential,
    "Random":     select_chunks_random,
    "SSG":        select_chunks_ssg,
}


# ─── Core single-run evaluation ───────────────────────────────────────────────

def _evaluate_strategy(
    name: str,
    selector_fn,
    all_chunks: List[Dict],
    sample_label_map: Dict[int, int],
    samples: List[Dict],
    budget_ratio: float,
    agent: SLMAuditAgent,
) -> Dict:
    """
    Evaluate one strategy by selecting from the FULL chunk pool.

    This is the correct Stackelberg formulation: the defender allocates
    a token budget across the *entire* PR (all functions/samples) rather
    than independently within each function.

    SSG concentrates the budget on high-risk chunks; Sequential reads
    top-to-bottom; Random picks uniformly.  The difference in WHICH chunks
    each strategy selects drives the VDR gap.
    """
    total_vulns  = sum(1 for s in samples if s["label"] == 1)
    total_clean  = len(samples) - total_vulns

    start_time = time.time()

    # ── Select across the entire PR chunk pool ────────────────────────────────
    selected, _ = selector_fn(all_chunks, budget_ratio=budget_ratio)
    total_tokens_used = sum(c["tokens"] for c in selected)

    # ── Audit each selected chunk; track sample-level detections ─────────────
    detected_vuln_samples: set = set()
    fp_samples:            set = set()

    for chunk in selected:
        detected = agent.audit_chunk(chunk["text"], chunk["label"], chunk["risk"])
        if detected:
            sid = chunk["sample_id"]
            if sample_label_map[sid] == 1:
                detected_vuln_samples.add(sid)
            else:
                fp_samples.add(sid)

    detected_vulns  = len(detected_vuln_samples)
    false_positives = len(fp_samples)
    latency         = time.time() - start_time

    vdr       = detected_vulns / total_vulns if total_vulns > 0 else 0.0
    fpr       = false_positives / total_clean if total_clean > 0 else 0.0
    precision = (
        detected_vulns / (detected_vulns + false_positives)
        if (detected_vulns + false_positives) > 0
        else 0.0
    )
    recall    = vdr
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    efficiency = detected_vulns / (total_tokens_used / 1000) if total_tokens_used > 0 else 0.0

    return {
        "Strategy":    name,
        "Budget_Ratio": budget_ratio,
        "VDR":         round(vdr, 4),
        "FPR":         round(fpr, 4),
        "Precision":   round(precision, 4),
        "Recall":      round(recall, 4),
        "F1":          round(f1, 4),
        "Tokens_Used": total_tokens_used,
        "Efficiency":  round(efficiency, 4),
        "Latency_s":   round(latency, 4),
    }


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_experiment(
    num_samples: int = 100,
    budget_ratio: float = BUDGET_RATIO,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run a single experiment at a fixed budget ratio."""
    print(f"\n--- Experiment  budget={budget_ratio:.0%}  n={num_samples} ---")

    # Seed Python's random for reproducible mock-agent decisions
    _random.seed(random_seed)

    # Load & profile
    samples    = load_samples(n=num_samples)
    print(f"  Dataset: {get_stats(samples)}")
    all_chunks = profile_samples(samples)
    print(f"  Total chunks: {len(all_chunks)}")

    # Sample-level ground-truth lookup (used in _evaluate_strategy)
    sample_label_map: Dict[int, int] = {s["id"]: s["label"] for s in samples}

    agent   = SLMAuditAgent(mode="mock")
    results = []

    for name, selector_fn in STRATEGIES.items():
        # Re-seed before each strategy so all three face the same random draws
        # on equivalent chunks — this isolates selection quality, not luck.
        _random.seed(random_seed)
        row = _evaluate_strategy(
            name, selector_fn, all_chunks, sample_label_map, samples, budget_ratio, agent
        )
        results.append(row)
        print(
            f"  [{name:12s}] VDR={row['VDR']:.2%}  F1={row['F1']:.3f}  "
            f"Prec={row['Precision']:.3f}  Tokens={row['Tokens_Used']:,}"
        )

    return pd.DataFrame(results)


# ─── Budget sweep ─────────────────────────────────────────────────────────────

def run_budget_sweep(
    num_samples: int = 100,
    budget_ratios: List[float] | None = None,
) -> pd.DataFrame:
    """
    Sweep over a range of budget ratios and collect VDR / F1 / Efficiency
    for each strategy.  Returns a combined DataFrame.
    """
    if budget_ratios is None:
        budget_ratios = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    print("\n========== Budget Sweep ==========")
    frames = []
    for br in tqdm(budget_ratios, desc="Budget sweep"):
        df = run_experiment(num_samples=num_samples, budget_ratio=br)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    sweep_file = os.path.join(RESULTS_DIR, "budget_sweep.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    combined.to_csv(sweep_file, index=False)
    print(f"Budget sweep saved to {sweep_file}")
    return combined


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = run_experiment(num_samples=100)
    print("\nResults Summary:")
    print(df.to_string(index=False))
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved to {RESULTS_FILE}")
