"""
data_loader.py
Loads the Devign vulnerability dataset from HuggingFace Hub.

Primary source:  code_x_glue_cc_defect_detection  (21 854 real C/C++ functions)
  → columns: func (code), target (0/1), project, commit_id

Falls back to a small synthetic set if HF is unavailable.

Each sample returned is a dict:
  {
    "id":        int,
    "code":      str,   # raw C/C++ function code
    "label":     int,   # 1 = vulnerable, 0 = clean
    "project":   str,
  }
"""

import random
import json
import os
from pathlib import Path
from typing import List, Dict

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from src.config import RANDOM_SEED, NUM_SAMPLES, USE_DATASET

CACHE_DIR = Path("data")

# ─── HuggingFace dataset identifiers ────────────────────────────────────────
# Primary: CodeXGLUE Defect Detection = official Devign on HF Hub
HF_DEVIGN_NAME = "code_x_glue_cc_defect_detection"
HF_DEVIGN_SPLIT = "train"  # 21 854 samples; test/valid sets are small

# ─── Minimal synthetic fallback (used only when HF is unavailable) ───────────
_SYNTHETIC_FALLBACK = [
    (1, 'void copy(char *dst, char *src){char buf[64]; strcpy(buf,src); strcpy(dst,buf);}'),
    (0, 'int add(int a, int b){return a+b;}'),
    (1, 'void run(char *cmd){char s[256]; sprintf(s,cmd); system(s);}'),
    (0, 'void sort(int *a,int n){int i,j,t;for(i=0;i<n;i++)for(j=i+1;j<n;j++)if(a[i]>a[j]){t=a[i];a[i]=a[j];a[j]=t;}}'),
    (1, 'void auth(char *tok){char d[512]; jwt_decode(tok,d,NULL); if(strcmp(d,"admin")==0) grant_root();}'),
    (0, 'int bsearch(int *a,int n,int t){int lo=0,hi=n-1;while(lo<=hi){int m=lo+(hi-lo)/2;if(a[m]==t)return m;if(a[m]<t)lo=m+1;else hi=m-1;}return -1;}'),
    (1, 'void sql(char *user,char *pw){char q[512]; sprintf(q,"SELECT * FROM users WHERE user=\'%s\' AND pw=\'%s\'",user,pw); db_exec(q);}'),
    (0, 'float dot(float *a,float *b,int n){float s=0;for(int i=0;i<n;i++)s+=a[i]*b[i];return s;}'),
]


# ─── HuggingFace loader — code_x_glue_cc_defect_detection (Devign) ───────────

def _load_devign_hf(n: int) -> List[Dict]:
    """
    Load n samples from code_x_glue_cc_defect_detection (official Devign on HF).
    Columns: func (code), target (0/1), project, commit_id.

    Strategy: take all splits, pool them, then draw a BALANCED stratified sample
    (n/2 vulnerable + n/2 clean) so the evaluator always has equal class sizes.
    """
    rng = random.Random(RANDOM_SEED)

    print(f"[data_loader] Downloading 'code_x_glue_cc_defect_detection' from HF Hub …")
    all_splits = load_dataset(HF_DEVIGN_NAME)    # loads train+validation+test
    print(f"[data_loader] Splits: { {k: len(v) for k,v in all_splits.items()} }")

    # Concatenate all splits into one pool
    from datasets import concatenate_datasets
    full = concatenate_datasets(list(all_splits.values()))
    print(f"[data_loader] Total pool: {len(full)} samples")

    # Separate by label
    vuln_indices  = [i for i, row in enumerate(full) if row["target"] == 1]
    clean_indices = [i for i, row in enumerate(full) if row["target"] == 0]
    print(f"[data_loader] Pool: {len(vuln_indices)} vulnerable, {len(clean_indices)} clean")

    # Stratified sample: n/2 from each class
    half = n // 2
    sampled_vuln  = rng.sample(vuln_indices,  min(half, len(vuln_indices)))
    sampled_clean = rng.sample(clean_indices, min(half, len(clean_indices)))

    rows = full.select(sampled_vuln + sampled_clean)

    samples = []
    for idx, row in enumerate(rows):
        code = row["func"].strip()
        if len(code) < 30:      # skip degenerate near-empty entries
            continue
        samples.append({
            "id":        idx,
            "code":      code,
            "label":     int(row["target"]),
            "project":   row.get("project", "unknown"),
            "commit_id": row.get("commit_id", ""),
        })

    # Shuffle so vuln/clean are interleaved
    rng.shuffle(samples)
    # Re-assign sequential IDs
    for i, s in enumerate(samples):
        s["id"] = i

    print(f"[data_loader] Loaded {len(samples)} real Devign samples "
          f"({sum(1 for s in samples if s['label']==1)} vuln, "
          f"{sum(1 for s in samples if s['label']==0)} clean).")
    return samples


# ─── Public API ───────────────────────────────────────────────────────────────

def load_samples(n: int = NUM_SAMPLES, use_hf: bool = True) -> List[Dict]:
    """
    Return a list of n samples from the Devign vulnerability dataset.

    1. Checks local cache (data/samples_devign_<n>.json) to avoid re-downloading.
    2. Downloads from HuggingFace Hub (code_x_glue_cc_defect_detection).
    3. Falls back to a minimal synthetic set if HF is unavailable.

    Samples are balanced: ~50 % vulnerable, ~50 % clean.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"samples_devign_{n}.json"

    # ── Cache hit ──
    if cache_file.exists():
        print(f"[data_loader] Loading {n} samples from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── HuggingFace download ──
    samples: List[Dict] = []
    if use_hf and HF_AVAILABLE:
        try:
            samples = _load_devign_hf(n)
        except Exception as exc:
            print(f"[data_loader] HF download failed: {exc}")
            print("[data_loader] Falling back to synthetic data.")

    # ── Synthetic fallback ──
    if not samples:
        print("[data_loader] Using synthetic fallback dataset.")
        rng = random.Random(RANDOM_SEED)
        pool = _SYNTHETIC_FALLBACK * (n // len(_SYNTHETIC_FALLBACK) + 1)
        rng.shuffle(pool)
        samples = [
            {"id": i, "code": code, "label": lbl, "project": "synthetic", "commit_id": ""}
            for i, (lbl, code) in enumerate(pool[:n])
        ]

    # ── Cache to disk ──
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"[data_loader] Cached {len(samples)} samples → {cache_file}")
    return samples


def get_stats(samples: List[Dict]) -> Dict:
    """Return basic label statistics for a sample list."""
    total = len(samples)
    vuln  = sum(1 for s in samples if s["label"] == 1)
    return {
        "total":     total,
        "vulnerable": vuln,
        "clean":     total - vuln,
        "vuln_rate": round(vuln / total, 3) if total else 0,
    }


if __name__ == "__main__":
    # Quick smoke-test: download 200 real samples and print stats
    data = load_samples(n=200, use_hf=True)
    stats = get_stats(data)
    print(f"\nStats: {stats}")
    print(f"Sample[0] project={data[0]['project']}  label={data[0]['label']}")
    print(f"func[:300]:\n{data[0]['code'][:300]}")
