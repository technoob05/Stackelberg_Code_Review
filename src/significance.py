"""
significance.py
Statistical significance testing for the Stackelberg Code Review evaluation.

Provides:
  - bootstrap_ci()        : 95 % bootstrap confidence interval for any metric
  - wilcoxon_test()       : Wilcoxon signed-rank test between two paired strategies
  - cohens_d()            : Effect-size (Cohen's d)
  - summarise_runs()      : Aggregate mean ± CI across N repeated runs
"""

import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    data = np.array(values, dtype=float)
    means = np.array(
        [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1.0 - ci) / 2
    return float(data.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


# ─── Wilcoxon signed-rank test ────────────────────────────────────────────────

def wilcoxon_test(
    a: List[float],
    b: List[float],
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test (paired, two-sided) between vectors a and b.
    Falls back to a permutation test if scipy is unavailable.

    Returns:
        (statistic, p_value)
    """
    try:
        from scipy.stats import wilcoxon
        stat, pval = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(pval)
    except ImportError:
        pass

    # Simple permutation test fallback
    diffs   = [x - y for x, y in zip(a, b)]
    obs     = abs(sum(diffs))
    n_perm  = 10_000
    rng     = random.Random(0)
    count   = 0
    for _ in range(n_perm):
        perm = sum(rng.choice([-1, 1]) * d for d in diffs)
        if abs(perm) >= obs:
            count += 1
    return obs, count / n_perm


# ─── Effect size ──────────────────────────────────────────────────────────────

def cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d effect size between two groups."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    pooled_std = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_std == 0:
        return float("inf") if a.mean() != b.mean() else 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def effect_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:   return "negligible"
    if d < 0.5:   return "small"
    if d < 0.8:   return "medium"
    return "large"


# ─── Summarise repeated-run results ──────────────────────────────────────────

def summarise_runs(
    all_rows: List[Dict],
    metric: str = "VDR",
) -> Dict[str, Dict]:
    """
    Given a list of result dicts (one per strategy-seed pair),
    return per-strategy statistics:
      {strategy: {mean, std, ci_lo, ci_hi, n}}
    """
    from collections import defaultdict
    per_strategy: Dict[str, List[float]] = defaultdict(list)
    for row in all_rows:
        per_strategy[row["Strategy"]].append(float(row[metric]))

    out = {}
    for strat, vals in per_strategy.items():
        mean, lo, hi = bootstrap_ci(vals)
        out[strat] = {
            "mean":  mean,
            "std":   float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "ci_lo": lo,
            "ci_hi": hi,
            "n":     len(vals),
        }
    return out


# ─── CLI quick test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random as _r
    _r.seed(7)
    ssg  = [_r.uniform(0.40, 0.60) for _ in range(20)]
    seq  = [_r.uniform(0.10, 0.30) for _ in range(20)]
    stat, pval = wilcoxon_test(ssg, seq)
    d          = cohens_d(ssg, seq)
    mean_s, lo, hi = bootstrap_ci(ssg)
    print(f"SSG   mean={mean_s:.3f}  95%-CI=[{lo:.3f}, {hi:.3f}]")
    print(f"Wilcoxon stat={stat:.2f}  p={pval:.4f}  {'*' if pval<0.05 else 'ns'}")
    print(f"Cohen's d={d:.2f}  ({effect_label(d)})")
