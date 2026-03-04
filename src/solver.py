"""
solver.py
Stackelberg LP Solver for the Resource-Constrained Code Review problem.

Defender problem (LP):
  Maximise:  sum_i  p_i * Ud_i * risk_i
  Subject to:
    sum_i  p_i * tokens_i  <= B        (token budget)
    0 <= p_i <= 1                      (probability)
    sum_i p_i <= K_max                 (optional: read at most K chunks)

This is a simplified one-sided Stackelberg / greedy knapsack dual
representation that is equivalent to solving the defender's best-response
given a threat model where the Attacker embeds bugs proportionally to Ld_i.

The object also exposes a method `stackelberg_minimax` that uses a zero-sum
formulation (small LP) to compute the mixed NE defence strategy.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linprog, OptimizeResult

from src.config import BUDGET_RATIO


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_payoff_vector(chunks: List[Dict]) -> np.ndarray:
    """Expected defender utility for reviewing chunk i = Ud_i * risk_i."""
    return np.array([c["Ud"] * c["risk"] for c in chunks], dtype=float)


def _build_cost_vector(chunks: List[Dict]) -> np.ndarray:
    """Token cost for reading chunk i."""
    return np.array([max(1, c["tokens"]) for c in chunks], dtype=float)


def _build_threat_vector(chunks: List[Dict]) -> np.ndarray:
    """Attacker payoff = Ld_i * risk_i (attacker prefers high-Ld chunks)."""
    return np.array([c["Ld"] * c["risk"] for c in chunks], dtype=float)


def _effective_budget(costs: np.ndarray, budget_ratio: float) -> float:
    """
    Compute effective token budget.
    Floor: always enough to cover at least the cheapest single chunk so that
    the defender can always review something (avoids VDR=0 on tiny samples).
    """
    raw = budget_ratio * float(costs.sum())
    if len(costs) == 0:
        return raw
    return max(raw, float(costs.min()))


# ─── Defender LP ─────────────────────────────────────────────────────────────

def solve_defender_lp(
    chunks: List[Dict],
    budget_ratio: float = BUDGET_RATIO,
) -> Tuple[np.ndarray, float]:
    """
    Solve the Defender's linear programme to find the optimal mixed strategy p*.

    Returns:
        p_star  : np.ndarray of shape (n_chunks,), defender probabilities
        utility : float, expected utility under p*
    """
    n = len(chunks)
    if n == 0:
        return np.array([]), 0.0

    payoff = _build_payoff_vector(chunks)
    costs  = _build_cost_vector(chunks)
    budget = _effective_budget(costs, budget_ratio)

    # LP:  minimize  -payoff^T p  (linprog minimises)
    # s.t.  costs^T p <= budget
    #        sum p    <= n   (trivially satisfied; included for robustness)
    #        0 <= p <= 1

    c_obj  = -payoff                           # negate for minimisation
    A_ub   = costs.reshape(1, -1)              # budget constraint
    b_ub   = np.array([budget])
    bounds = [(0.0, 1.0)] * n

    result: OptimizeResult = linprog(
        c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if result.success:
        p_star  = np.clip(result.x, 0, 1)
    else:
        # Fallback: proportional allocation
        p_star = np.clip(payoff / payoff.sum(), 0, 1) if payoff.sum() > 0 else np.ones(n) / n

    utility = float(np.dot(p_star, payoff))
    return p_star, utility


# ─── Zero-Sum Minimax (Stackelberg NE) ───────────────────────────────────────

def solve_stackelberg_minimax(
    chunks: List[Dict],
    budget_ratio: float = BUDGET_RATIO,
) -> Tuple[np.ndarray, float]:
    """
    Stackelberg Leader (Defender) commits first; Attacker best-responds.
    Formulated as a minimax LP:

    Defender wants:   max_{p} min_{a} [ sum_i p_i * Ud_i - a_j * Ld_j ]
    where a_j is the attacker's pure strategy (choose one chunk to corrupt).

    Linearised minimax LP (standard form):
      Variables: p_1..p_n, v (v = min over j of defender gain)
      Maximise  v
      s.t.  for each j:  sum_i  p_i * Ud_i  -  p_j * Ld_j  >= v
              sum_i  p_i * cost_i             <= B
              0 <= p_i <= 1

    This equals the standard Security Game LP.
    """
    n = len(chunks)
    if n == 0:
        return np.array([]), 0.0

    Ud     = np.array([c["Ud"] for c in chunks], dtype=float)
    Ld     = np.array([c["Ld"] for c in chunks], dtype=float)
    costs  = _build_cost_vector(chunks)
    budget = _effective_budget(costs, budget_ratio)

    # Variables: [p_0 .. p_{n-1}, v]  (total = n+1)
    # Objective: maximise v  ->  minimise [-v]
    c_obj = np.zeros(n + 1)
    c_obj[-1] = -1.0  # -v

    # Constraints:
    # 1) Minimax: for each attacker pure-strategy j (attack chunk j):
    #    Defender utility = sum_i p_i*Ud_i + p_j*Ld_j - Ld_j
    #    The p_j*Ld_j term REWARDS covering the attacked target (catches attacker).
    #    The -Ld_j constant is the base loss from any attack.
    #    Constraint (>= v form): sum_i p_i*Ud_i + p_j*Ld_j - Ld_j >= v
    #    Converted to <= form:  -sum_i p_i*Ud_i - p_j*Ld_j + v <= -Ld_j
    A_minimax = []
    b_minimax = []
    for j in range(n):
        row = -Ud.copy()         # - Ud_i * p_i  for all i
        row[j] -= Ld[j]          # - Ld_j * p_j  (covering j helps defender)
        row_ext = np.append(row, 1.0)   # + v
        A_minimax.append(row_ext)
        b_minimax.append(-Ld[j])  # RHS = -Ld_j (base loss constant)

    # 2) Budget: costs^T p <= B
    budget_row = np.append(costs, 0.0)
    A_minimax.append(budget_row)
    b_minimax.append(budget)

    A_ub = np.array(A_minimax)
    b_ub = np.array(b_minimax)
    bounds = [(0.0, 1.0)] * n + [(None, None)]   # p in [0,1]; v free

    result: OptimizeResult = linprog(
        c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if result.success:
        p_star  = np.clip(result.x[:n], 0, 1)
        utility = float(-result.fun)
    else:
        # Fallback to simple LP
        p_star, utility = solve_defender_lp(chunks, budget_ratio)

    return p_star, utility


# ─── Strategy selectors ───────────────────────────────────────────────────────

def select_chunks_ssg(
    chunks: List[Dict],
    budget_ratio: float = BUDGET_RATIO,
) -> Tuple[List[Dict], List[float]]:
    """
    Run the Stackelberg minimax solver and return the chunks selected
    (deterministically sampled from p*) and their probabilities.
    """
    p_star, _ = solve_stackelberg_minimax(chunks, budget_ratio)
    costs      = _build_cost_vector(chunks)
    budget     = _effective_budget(costs, budget_ratio)

    # Deterministic selection: pick chunks by LP-optimal probability p*.
    # p* already encodes the game-theoretic allocation (Ud + Ld tradeoff),
    # so we use it directly as priority.  Tiny Ud*risk tiebreaker avoids
    # arbitrary ordering among chunks with equal p*.
    tiebreak = 1e-8 * np.array([c["Ud"] * c["risk"] for c in chunks])
    priority = p_star + tiebreak
    order    = np.argsort(-priority)

    selected, used_tokens = [], 0.0
    for idx in order:
        c = chunks[idx]
        if used_tokens + costs[idx] <= budget:
            selected.append(c)
            used_tokens += costs[idx]

    probs = [float(p_star[chunks.index(c)]) for c in selected]
    return selected, probs


def select_chunks_sequential(
    chunks: List[Dict],
    budget_ratio: float = BUDGET_RATIO,
) -> Tuple[List[Dict], List[float]]:
    """Baseline 1: Read chunks top-to-bottom until budget exhausted."""
    costs  = _build_cost_vector(chunks)
    budget = _effective_budget(costs, budget_ratio)
    selected, used = [], 0.0
    for i, c in enumerate(chunks):
        if used + costs[i] <= budget:
            selected.append(c)
            used += costs[i]
    return selected, [1.0] * len(selected)


def select_chunks_random(
    chunks: List[Dict],
    budget_ratio: float = BUDGET_RATIO,
    seed: int = 42,
) -> Tuple[List[Dict], List[float]]:
    """Baseline 2: Select chunks randomly until budget exhausted."""
    import random
    rng    = random.Random(seed)
    costs  = _build_cost_vector(chunks)
    budget = _effective_budget(costs, budget_ratio)
    order  = list(range(len(chunks)))
    rng.shuffle(order)
    selected, used = [], 0.0
    for idx in order:
        c = chunks[idx]
        if used + costs[idx] <= budget:
            selected.append(c)
            used += costs[idx]
    return selected, [1.0] * len(selected)


def select_chunks_greedy_value(
    chunks: List[Dict],
    budget_ratio: float = BUDGET_RATIO,
) -> Tuple[List[Dict], List[float]]:
    """
    Baseline 3 – Greedy Value-Density (Operations Research knapsack).

    Ranks chunks by value-density = (Ud * risk) / tokens and greedily
    selects the highest-value-per-token chunks until the budget is
    exhausted.  This is the classical fractional-knapsack heuristic
    applied to the same payoff model used by SSG, but WITHOUT
    adversarial reasoning (no minimax, no attacker model).

    Comparison with SSG isolates the contribution of game-theoretic
    allocation over a strong non-adversarial optimisation baseline.
    """
    costs  = _build_cost_vector(chunks)
    budget = _effective_budget(costs, budget_ratio)

    # Value-density: benefit per token
    density = np.array([
        (c["Ud"] * c["risk"]) / max(1, c["tokens"])
        for c in chunks
    ], dtype=float)
    order = np.argsort(-density)

    selected, used = [], 0.0
    for idx in order:
        c = chunks[idx]
        if used + costs[idx] <= budget:
            selected.append(c)
            used += costs[idx]
    return selected, [1.0] * len(selected)


def select_chunks_top_risk(
    chunks: List[Dict],
    budget_ratio: float = BUDGET_RATIO,
) -> Tuple[List[Dict], List[float]]:
    """
    Baseline 4 – Top-Risk (static-analysis priority).

    Ranks chunks purely by their heuristic risk score (descending)
    and greedily selects until the budget is exhausted.  This models
    a SAST-style approach that triages by suspiciousness without any
    payoff modelling, cost-awareness, or adversarial game theory.

    Comparison with SSG isolates the contribution of the full
    Stackelberg payoff model (Ud, Ld, cost-aware LP) over using
    raw risk scores alone.
    """
    costs  = _build_cost_vector(chunks)
    budget = _effective_budget(costs, budget_ratio)

    # Sort by risk score only
    risk_scores = np.array([c["risk"] for c in chunks], dtype=float)
    order = np.argsort(-risk_scores)

    selected, used = [], 0.0
    for idx in order:
        c = chunks[idx]
        if used + costs[idx] <= budget:
            selected.append(c)
            used += costs[idx]
    return selected, [1.0] * len(selected)


if __name__ == "__main__":
    # Quick smoke test
    from src.risk_profiler import profile_code
    code = """
void vuln(char *src) { char buf[64]; strcpy(buf, src); system(buf); }
int safe_add(int a, int b) { return a + b; }
void sql(char *u) { char q[256]; sprintf(q, "SELECT * FROM users WHERE name='%s'", u); }
int max_val(int a, int b) { return a > b ? a : b; }
"""
    chunks = profile_code(code, ground_truth_label=1)
    print(f"Total chunks: {len(chunks)}")
    print("\n=== SSG Strategy ===")
    ssg_sel, probs = select_chunks_ssg(chunks)
    for c, p in zip(ssg_sel, probs):
        print(f"  chunk[{c['chunk_id']}] risk={c['risk']:.2f} p={p:.3f}: {c['text'][:60].strip()}")
    print("\n=== Sequential Strategy ===")
    seq_sel, _ = select_chunks_sequential(chunks)
    for c in seq_sel:
        print(f"  chunk[{c['chunk_id']}] risk={c['risk']:.2f}: {c['text'][:60].strip()}")
