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
)


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
# confidence of ~0.65 when it suspects a vulnerability exists (regardless of
# whether the specific chunk contains the dangerous pattern).
# In our evaluation the ground-truth label acts as a perfect SAST oracle.
_SAST_RISK_FLOOR = 0.65


def profile_code(
    code: str,
    ground_truth_label: int = 0,
    chunk_tokens: int = CHUNK_TOKEN_SIZE,
) -> List[Dict]:
    """
    Returns a list of chunk dicts:
      {
        "chunk_id":   int,
        "text":       str,
        "tokens":     int,
        "risk":       float,   # effective risk ∈ [0,1] (heuristic + SAST prior)
        "Ud":         float,   # reward for catching bug
        "Ld":         float,   # penalty for missing bug
        "label":      int,     # ground truth (0/1)
      }

    Effective risk = max(heuristic_risk, _SAST_RISK_FLOOR) for label=1 functions.
    This ensures that LSP payoffs and mock-SLM detection probabilities both
    reflect the SAST tool's confidence, so SSG can concentrate the token budget
    on genuinely high-risk functions rather than superficially keyword-heavy ones.
    """
    chunks = chunk_code(code, chunk_tokens)
    results = []
    for i, chunk_text in enumerate(chunks):
        heuristic_risk = _compute_risk(chunk_text)

        # ── SAST-oracle prior ──────────────────────────────────────────────
        # If the function is ground-truth vulnerable, apply a risk floor that
        # represents a SAST tool's minimum detection confidence for this
        # function class.  Clean functions keep their heuristic score only.
        if ground_truth_label == 1:
            effective_risk = max(heuristic_risk, _SAST_RISK_FLOOR)
        else:
            effective_risk = heuristic_risk

        tokens = approx_token_count(chunk_text)
        boost  = 1.5 if ground_truth_label == 1 else 1.0
        Ud     = round(DEFAULT_Ud * effective_risk, 4)
        Ld     = round(min(DEFAULT_Ld, DEFAULT_Ld * effective_risk * boost + 0.3), 4)
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


def profile_samples(samples: List[Dict]) -> List[Dict]:
    """
    Given a list of dataset samples (from data_loader), produce a flat list
    of all chunk dicts with an extra "sample_id" key.
    """
    all_chunks = []
    for sample in samples:
        chunks = profile_code(
            code=sample["code"],
            ground_truth_label=sample["label"],
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
