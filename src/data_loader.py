"""
data_loader.py
Multi-dataset loader for the Stackelberg Code Review experiment.

Supported datasets (all via HuggingFace Hub):
  1. devign    – code_x_glue_cc_defect_detection  (27 318 real C/C++ functions)
  2. bigvul    – bstee615/bigvul                   (217 007 C/C++ CVE functions)
  3. swebench  – princeton-nlp/SWE-bench           (2 294 Python PR patches)
  4. combined  – stratified pool from all three

Each sample returned:
  {
    "id":       int,
    "code":     str,   # source code (C/C++ or Python diff context)
    "label":    int,   # 1 = vulnerable / buggy, 0 = clean / fixed
    "project":  str,
    "source":   str,   # "devign" | "bigvul" | "swebench" | "synthetic"
  }
"""

import re
import random
import json
import os
from pathlib import Path
from typing import List, Dict

try:
    from datasets import load_dataset, concatenate_datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from src.config import RANDOM_SEED, NUM_SAMPLES, USE_DATASET

CACHE_DIR = Path("data")

# ─── HuggingFace dataset identifiers ─────────────────────────────────────────
HF_DEVIGN_NAME   = "code_x_glue_cc_defect_detection"
HF_BIGVUL_NAME   = "bstee615/bigvul"
HF_SWEBENCH_NAME = "princeton-nlp/SWE-bench"


# ─── Minimal synthetic fallback (offline only) ────────────────────────────────
_SYNTHETIC_FALLBACK = [
    (1, 'void copy(char *dst, char *src){char buf[64]; strcpy(buf,src); strcpy(dst,buf);}', 'synthetic'),
    (0, 'int add(int a, int b){return a+b;}', 'synthetic'),
    (1, 'void run(char *cmd){char s[256]; sprintf(s,cmd); system(s);}', 'synthetic'),
    (0, 'void sort(int *a,int n){int i,j,t;for(i=0;i<n;i++)for(j=i+1;j<n;j++)if(a[i]>a[j]){t=a[i];a[i]=a[j];a[j]=t;}}', 'synthetic'),
    (1, 'void auth(char *tok){char d[512]; jwt_decode(tok,d,NULL); if(strcmp(d,"admin")==0) grant_root();}', 'synthetic'),
    (0, 'int search(int *a,int n,int t){int lo=0,hi=n-1;while(lo<=hi){int m=lo+(hi-lo)/2;if(a[m]==t)return m;if(a[m]<t)lo=m+1;else hi=m-1;}return -1;}', 'synthetic'),
    (1, 'void sql(char *u,char *p){char q[512]; sprintf(q,"SELECT * FROM users WHERE u=\'%s\' AND p=\'%s\'",u,p); db_exec(q);}', 'synthetic'),
    (0, 'float dot(float *a,float *b,int n){float s=0;for(int i=0;i<n;i++)s+=a[i]*b[i];return s;}', 'synthetic'),
]


# ─── Dataset 1: Devign (C/C++) ────────────────────────────────────────────────

def _load_devign(n: int, rng: random.Random) -> List[Dict]:
    """Load n balanced samples from code_x_glue_cc_defect_detection."""
    print(f"[data_loader] Loading Devign ({HF_DEVIGN_NAME}) …")
    ds   = load_dataset(HF_DEVIGN_NAME)
    full = concatenate_datasets(list(ds.values()))
    print(f"[data_loader] Devign pool: {len(full)} rows")

    vuln_idx  = [i for i, r in enumerate(full) if r["target"] == 1]
    clean_idx = [i for i, r in enumerate(full) if r["target"] == 0]
    half     = n // 2
    selected = rng.sample(vuln_idx, min(half, len(vuln_idx))) + \
               rng.sample(clean_idx, min(half, len(clean_idx)))
    rows = full.select(selected)

    samples = []
    for row in rows:
        code = row["func"].strip()
        if len(code) < 30:
            continue
        samples.append({
            "code":    code,
            "label":   int(row["target"]),
            "project": row.get("project", "devign"),
            "source":  "devign",
        })
    return samples


# ─── Dataset 2: BigVul (C/C++) ────────────────────────────────────────────────

def _load_bigvul(n: int, rng: random.Random) -> List[Dict]:
    """
    Load n balanced samples from bstee615/bigvul.
    Vulnerable = function body before the CVE security patch.
    Clean      = function body after the patch.
    """
    print(f"[data_loader] Loading BigVul ({HF_BIGVUL_NAME}) …")
    ds = load_dataset(HF_BIGVUL_NAME, split="train", trust_remote_code=True)
    print(f"[data_loader] BigVul pool: {len(ds)} rows")

    cols = ds.column_names
    print(f"[data_loader] BigVul columns (first 20): {cols[:20]}")

    # Detect code columns – BigVul has had minor schema changes on HF
    code_col_vuln = code_col_clean = None
    for cv, cc in [
        ("func_before", "func_after"),
        ("func_before_change", "func_after_change"),
        ("before", "after"),
    ]:
        if cv in cols:
            code_col_vuln  = cv
            code_col_clean = cc if cc in cols else None
            break
    # Fallback: single code column with a label column
    if code_col_vuln is None:
        for c in ("func", "code", "function"):
            if c in cols:
                code_col_vuln = code_col_clean = c
                break

    label_col = next((c for c in ("vul", "target", "label", "vulnerable") if c in cols), None)

    if code_col_vuln is None:
        print("[data_loader] BigVul: cannot find code column, skipping.")
        return []

    vuln_rows, clean_rows = [], []
    for row in ds:
        vuln_code = (row.get(code_col_vuln) or "").strip()
        if len(vuln_code) < 30:
            continue
        lbl = int(row.get(label_col, 1)) if label_col else 1
        proj = row.get("project", row.get("repo", row.get("cve_id", "bigvul")))
        if lbl == 1:
            vuln_rows.append({"code": vuln_code, "label": 1, "project": proj, "source": "bigvul"})
        if code_col_clean and code_col_clean != code_col_vuln:
            clean_code = (row.get(code_col_clean) or "").strip()
            if len(clean_code) >= 30:
                clean_rows.append({"code": clean_code, "label": 0, "project": proj, "source": "bigvul"})
        if len(vuln_rows) >= n * 2 and len(clean_rows) >= n * 2:
            break

    half = n // 2
    chosen = rng.sample(vuln_rows, min(half, len(vuln_rows))) + \
             rng.sample(clean_rows, min(half, len(clean_rows)))
    print(f"[data_loader] BigVul: {len(chosen)} samples selected")
    return chosen


# ─── Dataset 3: SWE-bench (Python PRs) ───────────────────────────────────────

def _extract_patch_context(patch: str) -> tuple:
    """
    Parse a unified diff and extract (old_code, new_code) strings.
    old_code = context + removed lines (state before the fix — may contain the bug)
    new_code = context + added lines  (state after the fix — clean)
    """
    old_lines, new_lines = [], []
    for line in patch.splitlines():
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith("+"):
            new_lines.append(line[1:])
        else:
            ctx = line.lstrip(" ")
            old_lines.append(ctx)
            new_lines.append(ctx)
    return "\n".join(old_lines), "\n".join(new_lines)


def _load_swebench(n: int, rng: random.Random) -> List[Dict]:
    """
    Load n balanced samples from princeton-nlp/SWE-bench.
    Uses the `patch` field (unified diffs of real Python bug fixes).
    old state (before fix) → label=1,  new state (after fix) → label=0.
    """
    print(f"[data_loader] Loading SWE-bench ({HF_SWEBENCH_NAME}) …")
    ds = load_dataset(HF_SWEBENCH_NAME, split="test", trust_remote_code=True)
    print(f"[data_loader] SWE-bench pool: {len(ds)} rows")

    vuln_samples, clean_samples = [], []
    for row in ds:
        patch = (row.get("patch") or "").strip()
        repo  = row.get("repo", "swebench")
        if not patch or len(patch) < 80:
            continue
        diff_lines = sum(1 for l in patch.splitlines()
                         if l.startswith(("+", "-")) and len(l.strip()) > 2)
        if diff_lines < 5:
            continue
        old_code, new_code = _extract_patch_context(patch)
        if len(old_code.strip()) >= 50:
            vuln_samples.append({"code": old_code[:4000], "label": 1,
                                  "project": repo, "source": "swebench"})
        if len(new_code.strip()) >= 50:
            clean_samples.append({"code": new_code[:4000], "label": 0,
                                   "project": repo, "source": "swebench"})
        if len(vuln_samples) >= n and len(clean_samples) >= n:
            break

    half = n // 2
    chosen = rng.sample(vuln_samples,  min(half, len(vuln_samples))) + \
             rng.sample(clean_samples, min(half, len(clean_samples)))
    print(f"[data_loader] SWE-bench: {len(chosen)} samples selected")
    return chosen


# ─── Combined loader ──────────────────────────────────────────────────────────

def _load_combined(n: int, rng: random.Random) -> List[Dict]:
    """Pool all three datasets then stratified-sample n total."""
    per = max(n, 150)
    pool: List[Dict] = []
    for loader in (_load_devign, _load_bigvul, _load_swebench):
        try:
            pool.extend(loader(per, rng))
        except Exception as e:
            print(f"[data_loader] {loader.__name__} skipped: {e}")

    vuln  = [s for s in pool if s["label"] == 1]
    clean = [s for s in pool if s["label"] == 0]
    half  = n // 2
    chosen = rng.sample(vuln,  min(half, len(vuln))) + \
             rng.sample(clean, min(half, len(clean)))
    sources = set(s["source"] for s in chosen)
    print(f"[data_loader] Combined: {len(chosen)} samples from {sources}")
    return chosen


# ─── Public API ───────────────────────────────────────────────────────────────

def load_samples(
    n: int       = NUM_SAMPLES,
    use_hf: bool = True,
    dataset: str = USE_DATASET,   # "devign" | "bigvul" | "swebench" | "combined"
) -> List[Dict]:
    """
    Return n balanced samples from the chosen dataset(s).
    Checks local JSON cache first; downloads from HF on cache miss.
    Falls back to tiny synthetic set if HF is unavailable.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"samples_{dataset}_{n}.json"

    if cache_file.exists():
        print(f"[data_loader] Loading {n} samples from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    samples: List[Dict] = []
    rng = random.Random(RANDOM_SEED)

    if use_hf and HF_AVAILABLE:
        loaders = {
            "devign":   _load_devign,
            "bigvul":   _load_bigvul,
            "swebench": _load_swebench,
            "combined": _load_combined,
        }
        fn = loaders.get(dataset, _load_devign)
        try:
            samples = fn(n, rng)
        except Exception as exc:
            print(f"[data_loader] HF failed ({exc}), using synthetic fallback.")

    if not samples:
        print("[data_loader] Using 8-entry synthetic fallback.")
        samples = [
            {"id": i, "code": code, "label": lbl,
             "project": src, "source": "synthetic"}
            for i, (lbl, code, src) in enumerate(_SYNTHETIC_FALLBACK)
        ]

    rng.shuffle(samples)
    for i, s in enumerate(samples):
        s["id"] = i

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[data_loader] Cached {len(samples)} samples → {cache_file}")
    print(f"[data_loader] Loaded {len(samples)} samples "
          f"({sum(s['label']==1 for s in samples)} vuln, "
          f"{sum(s['label']==0 for s in samples)} clean).")
    return samples


def get_stats(samples: List[Dict]) -> Dict:
    total = len(samples)
    vuln  = sum(1 for s in samples if s["label"] == 1)
    return {
        "total":      total,
        "vulnerable": vuln,
        "clean":      total - vuln,
        "vuln_rate":  round(vuln / total, 3) if total > 0 else 0,
        "sources":    list(set(s.get("source", "?") for s in samples)),
    }


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "devign"
    n       = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(f"\n=== Smoke test: dataset={dataset!r}  n={n} ===")
    samples = load_samples(n=n, dataset=dataset)
    print(f"Stats: {get_stats(samples)}")
    s0 = samples[0]
    print(f"Sample[0] source={s0['source']}  project={s0['project']}  label={s0['label']}")
    print(f"code[:200]: {s0['code'][:200]}")
