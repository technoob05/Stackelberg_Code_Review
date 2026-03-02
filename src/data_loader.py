"""
data_loader.py
Downloads and preprocesses the Devign vulnerability dataset from HuggingFace.
Each sample is a dict:
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

# ─── Synthetic fallback data ──────────────────────────────────────────────────
SYNTHETIC_SAMPLES = [
    {
        "id": i,
        "code": code,
        "label": label,
        "project": "synthetic",
    }
    for i, (code, label) in enumerate([
        ('void vuln(char *src) { char buf[64]; strcpy(buf, src); }', 1),
        ('int safe_add(int a, int b) { return a + b; }', 0),
        ('void exec_cmd(char *cmd) { system(cmd); }', 1),
        ('int max(int a, int b) { return a > b ? a : b; }', 0),
        ('char *get_user_input() { char *buf = malloc(256); gets(buf); return buf; }', 1),
        ('size_t strlen_safe(const char *s) { if (!s) return 0; return strlen(s); }', 0),
        ('void log_msg(char *fmt, char *msg) { char buf[128]; sprintf(buf, fmt, msg); }', 1),
        ('void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }', 0),
        ('void sql_query(char *user) { char q[256]; sprintf(q, "SELECT * FROM users WHERE name=\'%s\'", user); }', 1),
        ('int clamp(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }', 0),
        ('void deserialize(char *data) { pickle_loads(data); }', 1),
        ('float avg(float *arr, int n) { float s = 0; for(int i=0;i<n;i++) s+=arr[i]; return s/n; }', 0),
        ('void copy_buf(char *dst, char *src, int len) { memcpy(dst, src, len+1); }', 1),
        ('bool is_even(int n) { return n % 2 == 0; }', 0),
        ('void auth(char *token) { jwt_decode(token, NULL, NULL); }', 1),
        ('int gcd(int a, int b) { return b ? gcd(b, a%b) : a; }', 0),
        ('void run(char *cmd) { exec(cmd); }', 1),
        ('int factorial(int n) { return n <= 1 ? 1 : n * factorial(n-1); }', 0),
        ('void free_and_use(int *p) { free(p); *p = 42; }', 1),
        ('void print_greeting(const char *name) { printf("Hello, %s!\\n", name); }', 0),
    ] * 25   # repeat to have enough samples
    )
]


def _load_devign_hf(n: int) -> List[Dict]:
    """Load from HuggingFace Hub (claudios/Devign)."""
    ds = load_dataset("claudios/Devign", split="train", trust_remote_code=True)
    rng = random.Random(RANDOM_SEED)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    samples = []
    for idx, row in enumerate(ds.select(indices)):
        samples.append({
            "id": idx,
            "code": row.get("func", row.get("code", "")),
            "label": int(row.get("target", row.get("label", 0))),
            "project": row.get("project", "unknown"),
        })
    return samples


def _load_bigvul_hf(n: int) -> List[Dict]:
    """Load from HuggingFace Hub (benjaminjellis/bigvul)."""
    ds = load_dataset("benjaminjellis/bigvul", split="train", trust_remote_code=True)
    rng = random.Random(RANDOM_SEED)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    samples = []
    for idx, row in enumerate(ds.select(indices)):
        cwe   = row.get("CWE ID", "")
        label = 1 if cwe and cwe != "NVD-CWE-Other" else 0
        samples.append({
            "id": idx,
            "code": row.get("func_before", ""),
            "label": label,
            "project": row.get("project", "unknown"),
        })
    return samples


def load_samples(n: int = NUM_SAMPLES, use_hf: bool = True) -> List[Dict]:
    """
    Returns a list of at most `n` samples.
    Falls back to synthetic data if HuggingFace is unavailable.
    Caches results locally to data/samples_cache.json.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"samples_{USE_DATASET}_{n}.json"

    # Load from cache if present
    if cache_file.exists():
        print(f"[data_loader] Loading {n} samples from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    samples = []
    if use_hf and HF_AVAILABLE:
        try:
            print(f"[data_loader] Fetching '{USE_DATASET}' from HuggingFace Hub …")
            if USE_DATASET == "devign":
                samples = _load_devign_hf(n)
            else:
                samples = _load_bigvul_hf(n)
            print(f"[data_loader] Loaded {len(samples)} samples from HF.")
        except Exception as exc:
            print(f"[data_loader] HF download failed ({exc}). Falling back to synthetic data.")

    if not samples:
        print("[data_loader] Using synthetic fallback dataset.")
        rng = random.Random(RANDOM_SEED)
        samples = rng.sample(SYNTHETIC_SAMPLES, min(n, len(SYNTHETIC_SAMPLES)))
        for i, s in enumerate(samples):
            s["id"] = i

    # Cache to disk
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"[data_loader] Cached {len(samples)} samples -> {cache_file}")
    return samples


def get_stats(samples: List[Dict]) -> Dict:
    total = len(samples)
    vuln  = sum(1 for s in samples if s["label"] == 1)
    return {"total": total, "vulnerable": vuln, "clean": total - vuln,
            "vuln_rate": vuln / total if total else 0}


if __name__ == "__main__":
    data = load_samples(n=50, use_hf=True)
    print(f"Stats: {get_stats(data)}")
    print("Sample[0]:", data[0]["code"][:120])
