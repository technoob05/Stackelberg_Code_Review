"""
risk_profiler.py
Splits a function/file into code chunks and assigns heuristic payoffs:
  - Ud (Defender reward for catching a bug in this chunk)
  - Ld (Defender Loss/penalty for missing a bug in this chunk)
  - tokens (estimated token cost to include this chunk in prompt)

Chunks are scored via:
  1. Keyword danger score (known dangerous functions / patterns)
  2. Structural risk score (nesting depth, long lines, magic numbers)
The combined score normalises to a [0, 1] "risk" value.
Ud  = DEFAULT_Ud * risk
Ld  = DEFAULT_Ld * risk + 0.5   (always at least 0.5 missing cost)
"""

import re
from typing import List, Dict, Tuple

from src.config import (
    CHUNK_TOKEN_SIZE,
    DANGER_KEYWORDS,
    DEFAULT_Ud,
    DEFAULT_Ld,
    RISK_MODE,
    SAST_SIM_TPR,
    SAST_SIM_FPR,
)


# ─── Attacker attractiveness model ───────────────────────────────────────────
# Weights reflect how attractive a code pattern is to an attacker (i.e., how
# damaging a missed vulnerability would be).  This is DIFFERENT from the
# defender's keyword risk — the attacker cares about exploitability and impact,
# not just pattern suspiciousness.  The divergence between Ud (risk-driven)
# and Ld (attacker-driven) is what gives the Stackelberg LP its game-theoretic
# advantage over greedy value-density baselines.

_ATTACKER_WEIGHTS: dict[str, float] = {
    # Critical severity — Remote Code Execution / arbitrary command
    "system(":  5.0,  "exec(":    5.0,  "eval(":    5.0,
    "popen":    4.5,  "os.system": 5.0, "subprocess": 3.5,
    "execve":   5.0,  "execvp":   5.0,  "shell=True": 4.0,
    # High severity — memory corruption
    "strcpy":   4.0,  "strcat":   3.5,  "gets":     4.5,
    "sprintf":  3.5,  "memcpy":   3.0,  "memmove":  2.5,
    "realloc":  2.5,  "free(":    2.5,  "malloc":   2.0,
    # Injection (SQL, command, LDAP)
    "SELECT":   2.5,  "INSERT":   2.5,  "DROP":     4.0,
    "query":    1.5,  "execute(": 2.0,  "cursor":   1.5,
    # Authentication / secrets — high impact if leaked
    "password": 3.5,  "token":    3.0,  "secret":   3.5,
    "jwt":      3.5,  "private_key": 4.0,"api_key":  3.5,
    "crypto":   2.5,  "cipher":   2.5,  "hmac":     2.0,
    # Deserialization — classic gadget chains
    "pickle":   3.0,  "deserializ": 3.0,"yaml.load": 3.0,
    "marshal":  2.5,  "unserializ": 3.0,
    # File / path operations — path traversal, LFI
    "open(":    1.5,  "chmod":    2.0,  "chown":    2.0,
    "../" :     3.0,  "path.join": 1.0,
}


def _attacker_attractiveness(chunk: str) -> float:
    """
    Score how attractive a code chunk is for an attacker (0–1 normalised).

    A high score means the chunk, if vulnerable, would be highly exploitable
    or cause severe damage if the vulnerability is missed by the defender.
    """
    code_lower = chunk.lower()
    total = 0.0
    for kw, weight in _ATTACKER_WEIGHTS.items():
        if kw.lower() in code_lower:
            total += weight
    # Structural: pointer arithmetic / type casts (exploit enablers)
    if re.search(r'\(\s*\w+\s*\*\s*\)', chunk):   # C-style cast like (char*)
        total += 1.5
    if re.search(r'\*\s*\(', chunk):               # pointer dereference
        total += 1.0
    if re.search(r'\[\s*\w+\s*[\+\-]', chunk):     # array index arithmetic
        total += 0.8
    return min(1.0, total / 8.0)


# ─── Tokenisation (simple whitespace approx) ─────────────────────────────────

def approx_token_count(text: str) -> int:
    """Approximate BPE token count: ~0.75 tokens per word for code."""
    words = re.split(r"\s+", text.strip())
    return max(1, int(len(words) * 0.75))


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_code(code: str, chunk_tokens: int = CHUNK_TOKEN_SIZE) -> List[str]:
    """
    Splits source code into roughly equal-sized chunks by line.
    Returns list of chunk strings.
    """
    lines = code.splitlines(keepends=True)
    chunks, current, current_tokens = [], [], 0
    for line in lines:
        t = approx_token_count(line)
        if current_tokens + t > chunk_tokens and current:
            chunks.append("".join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += t
    if current:
        chunks.append("".join(current))
    return chunks if chunks else [code]


# ─── Danger scoring ───────────────────────────────────────────────────────────

def _keyword_score(chunk: str) -> float:
    """Returns [0,1] based on fraction of danger keywords present."""
    code_lower = chunk.lower()
    hits = sum(1 for kw in DANGER_KEYWORDS if kw.lower() in code_lower)
    return min(1.0, hits / max(1, len(DANGER_KEYWORDS) * 0.1))


def _structural_score(chunk: str) -> float:
    """
    Structural risk indicators:
      - Max nesting depth (count brackets)
      - Presence of magic numbers / hard-coded strings
      - Very long lines (>120 chars)
    """
    score = 0.0
    lines = chunk.splitlines()
    depth = max_depth = 0
    for ch in chunk:
        if ch in "({[":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch in ")}]":
            depth = max(0, depth - 1)
    score += min(1.0, max_depth / 8)  # normalise depth

    long_lines = sum(1 for l in lines if len(l) > 120)
    score += min(1.0, long_lines / max(1, len(lines)))

    # Magic numbers / hard-coded credentials
    magic = re.findall(r'\b(?:0x[0-9A-Fa-f]+|\d{4,})\b', chunk)
    score += min(1.0, len(magic) / 5)

    return min(1.0, score / 3)  # average of 3 sub-scores


def _compute_risk(chunk: str) -> float:
    kw  = _keyword_score(chunk)
    st  = _structural_score(chunk)
    # Weighted combination
    return min(1.0, 0.65 * kw + 0.35 * st)


# ─── Public API ───────────────────────────────────────────────────────────────


# Minimum risk assigned to chunks from SAST-flagged (vulnerable) functions.
# Models a static-analysis pre-screening tool that reports a function-level
# confidence when it suspects a vulnerability exists.  Set to 0.40 (moderate
# confidence) so that within-function variation in keyword/structural scores
# is preserved — the SSG LP uses this variation to prioritise dangerous chunks.
_SAST_RISK_FLOOR = 0.40

# ─── Simulated SAST oracle ───────────────────────────────────────────────────
import random as _rand

def _sast_simulated_flag(ground_truth_label: int, seed: int = None) -> bool:
    """
    Simulate a realistic SAST tool with configurable TPR / FPR.
    Returns True if the simulated SAST flags this function.
    Uses a seeded RNG for reproducibility when seed is provided.
    """
    rng = _rand.Random(seed)
    if ground_truth_label == 1:
        return rng.random() < SAST_SIM_TPR
    else:
        return rng.random() < SAST_SIM_FPR


def profile_code(
    code: str,
    ground_truth_label: int = 0,
    chunk_tokens: int = CHUNK_TOKEN_SIZE,
    risk_mode: str | None = None,
    sast_seed: int | None = None,
) -> List[Dict]:
    """
    Returns a list of chunk dicts:
      {
        "chunk_id":   int,
        "text":       str,
        "tokens":     int,
        "risk":       float,   # effective risk ∈ [0,1]
        "Ud":         float,   # reward for catching bug
        "Ld":         float,   # penalty for missing bug
        "label":      int,     # ground truth (0/1)
      }

    Risk modes:
      "oracle"    — uses ground_truth_label to apply SAST floor (upper bound,
                    causes label leakage — use only as ablation reference)
      "heuristic" — pure keyword + structural scoring, NO label info
                    (realistic deployment scenario)
      "sast_sim"  — simulated SAST with configurable TPR/FPR; the SAST
                    flag is a noisy signal that does NOT use ground truth
                    directly in the selection path
    """
    if risk_mode is None:
        risk_mode = RISK_MODE

    chunks = chunk_code(code, chunk_tokens)
    results = []

    # ── Determine SAST flag (only for oracle / sast_sim modes) ────────────
    sast_flagged = False
    if risk_mode == "oracle":
        sast_flagged = (ground_truth_label == 1)
    elif risk_mode == "sast_sim":
        sast_flagged = _sast_simulated_flag(ground_truth_label, seed=sast_seed)
    # risk_mode == "heuristic": sast_flagged stays False (no label info)

    for i, chunk_text in enumerate(chunks):
        heuristic_risk = _compute_risk(chunk_text)
        attacker_score = _attacker_attractiveness(chunk_text)

        # ── Effective risk ─────────────────────────────────────────────────
        if sast_flagged:
            effective_risk = max(heuristic_risk, _SAST_RISK_FLOOR)
        else:
            effective_risk = heuristic_risk

        tokens = approx_token_count(chunk_text)

        # ── Ud: defender reward for detecting a vulnerability ─────────────
        # Purely risk-driven (keyword + structural).  Higher risk = more
        # value in reviewing this chunk.
        Ud = round(DEFAULT_Ud * effective_risk, 4)

        # ── Ld: attacker's gain / defender's loss when vulnerability missed
        # DIFFERS from Ud — driven by attacker attractiveness (exploitability
        # and impact) rather than detectability.  This divergence is the
        # source of SSG's game-theoretic advantage over Greedy-Value.
        ld_factor = max(effective_risk, attacker_score * 1.4)
        Ld = round(min(DEFAULT_Ld * 1.5,
                       DEFAULT_Ld * ld_factor + 0.15), 4)

        results.append({
            "chunk_id": i,
            "text":     chunk_text,
            "tokens":   tokens,
            "risk":     round(effective_risk, 4),
            "Ud":       Ud,
            "Ld":       Ld,
            "label":    ground_truth_label,
        })
    return results


def profile_samples(samples: List[Dict], chunk_tokens: int | None = None,
                    risk_mode: str | None = None) -> List[Dict]:
    """
    Given a list of dataset samples (from data_loader), produce a flat list
    of all chunk dicts with an extra "sample_id" key.

    Parameters
    ----------
    chunk_tokens : int or None
        Override for chunk size in tokens.  When None, reads the current value
        of ``CHUNK_TOKEN_SIZE`` from ``src.config`` at call-time (not import-time)
        so that runtime config changes (e.g. chunk-size ablation) take effect.
    risk_mode : str or None
        Override for risk mode ("oracle", "heuristic", "sast_sim").
        When None, reads from ``src.config.RISK_MODE``.
    """
    if chunk_tokens is None:
        # Read at call-time to pick up runtime config changes
        from src import config as _cfg
        chunk_tokens = _cfg.CHUNK_TOKEN_SIZE
    if risk_mode is None:
        from src import config as _cfg
        risk_mode = _cfg.RISK_MODE

    all_chunks = []
    for sample in samples:
        chunks = profile_code(
            code=sample["code"],
            ground_truth_label=sample["label"],
            chunk_tokens=chunk_tokens,
            risk_mode=risk_mode,
            sast_seed=sample.get("id", 0),  # reproducible per-sample SAST sim
        )
        for c in chunks:
            c["sample_id"] = sample["id"]
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    code = """
void vuln(char *src) {
    char buf[64];
    strcpy(buf, src);   /* no bounds check */
    system(buf);
}
"""
    chunks = profile_code(code, ground_truth_label=1)
    for c in chunks:
        print(f"[chunk {c['chunk_id']}] risk={c['risk']:.3f}  Ud={c['Ud']}  Ld={c['Ld']}  tokens={c['tokens']}")
        print("   ", c["text"][:80].replace("\n", " "))
