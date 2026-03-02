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
# Each sample is a realistic multi-line C function so the chunker produces
# ≥ 2 chunks per sample, giving the budget-allocation strategies something
# meaningful to work with.
_RAW_SYNTHETIC = [
    # ── VULNERABLE ──────────────────────────────────────────────────────────
    (1, """
void process_input(char *user_input) {
    char buf[64];
    int  len;
    /* BUG: no bounds check before strcpy */
    strcpy(buf, user_input);
    len = strlen(buf);
    printf("Processed %d bytes: %s\n", len, buf);
    /* further processing ... */
    buf[0] = toupper(buf[0]);
    printf("Capitalised: %s\n", buf);
}
"""),
    (0, """
int vector_dot(const float *a, const float *b, int n) {
    float sum = 0.0f;
    int   i;
    if (!a || !b || n <= 0) return -1;
    for (i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return (int)sum;
}
"""),
    (1, """
void exec_shell(char *cmd, char *arg) {
    char full[256];
    /* BUG: sprintf with user-controlled format string */
    sprintf(full, cmd, arg);
    system(full);
    printf("Command executed\n");
}
"""),
    (0, """
char *str_join(const char *a, const char *b) {
    size_t la = strlen(a), lb = strlen(b);
    char  *out = malloc(la + lb + 1);
    if (!out) return NULL;
    memcpy(out,      a, la);
    memcpy(out + la, b, lb);
    out[la + lb] = '\0';
    return out;
}
"""),
    (1, """
void read_config(const char *path) {
    FILE *fp = fopen(path, "r");
    char  line[128];
    char  key[64], val[64];
    if (!fp) return;
    /* BUG: gets() inside loop — unbounded read */
    while (gets(line)) {
        sscanf(line, "%s = %s", key, val);
        printf("  %s -> %s\n", key, val);
    }
    fclose(fp);
}
"""),
    (0, """
void insertion_sort(int *arr, int n) {
    int i, j, key;
    for (i = 1; i < n; i++) {
        key = arr[i];
        j   = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}
"""),
    (1, """
void db_login(char *username, char *password) {
    char query[512];
    /* BUG: SQL injection — user input directly interpolated */
    sprintf(query,
        "SELECT id FROM users WHERE username='%s' AND password='%s'",
        username, password);
    db_exec(query);
    log_msg("Login attempt for user: %s", username);
}
"""),
    (0, """
int binary_search(const int *arr, int n, int target) {
    int lo = 0, hi = n - 1, mid;
    while (lo <= hi) {
        mid = lo + (hi - lo) / 2;
        if      (arr[mid] == target) return mid;
        else if (arr[mid]  < target) lo = mid + 1;
        else                         hi = mid - 1;
    }
    return -1;
}
"""),
    (1, """
void deserialize_payload(char *raw, size_t len) {
    struct Payload *p = malloc(sizeof(struct Payload));
    /* BUG: memcpy with attacker-controlled len+1 (off-by-one) */
    memcpy(p->data, raw, len + 1);
    p->checksum = crc32(p->data, len);
    process_payload(p);
    free(p);
}
"""),
    (0, """
float matrix_trace(float **m, int n) {
    float trace = 0.0f;
    int   i;
    if (!m || n <= 0) return 0.0f;
    for (i = 0; i < n; i++) {
        if (!m[i]) continue;
        trace += m[i][i];
    }
    return trace;
}
"""),
    (1, """
char *build_redirect(char *host, char *path) {
    char url[256];
    /* BUG: no validation — open redirect + buffer overflow */
    sprintf(url, "http://%s%s", host, path);
    char *tok = jwt_decode(url, NULL, NULL);
    return strdup(tok ? tok : url);
}
"""),
    (0, """
void trim_whitespace(char *s) {
    char *start = s;
    char *end;
    if (!s || !*s) return;
    while (isspace((unsigned char)*start)) start++;
    memmove(s, start, strlen(start) + 1);
    end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) *(end--) = '\0';
}
"""),
    (1, """
void auth_token(const char *token) {
    char decoded[512];
    /* BUG: no signature verification on JWT */
    jwt_decode(token, decoded, NULL);
    if (strncmp(decoded, "admin", 5) == 0) {
        grant_root_access();
    }
    log_access(decoded);
}
"""),
    (0, """
int safe_multiply(long a, long b, long *out) {
    if (a != 0 && b > LONG_MAX / a) {
        return -1;  /* overflow */
    }
    *out = a * b;
    return 0;
}
"""),
    (1, """
void load_plugin(char *name) {
    char path[256];
    void *handle;
    /* BUG: no path sanitisation — directory traversal */
    snprintf(path, sizeof(path), "/plugins/%s.so", name);
    handle = dlopen(path, RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen error: %s\n", dlerror());
    }
    /* exec plugin entry point */
    void (*init)() = dlsym(handle, "plugin_init");
    if (init) init();
}
"""),
    (0, """
uint32_t fnv1a_hash(const void *data, size_t len) {
    const uint8_t *p    = (const uint8_t *)data;
    uint32_t       hash = 2166136261u;
    size_t         i;
    for (i = 0; i < len; i++) {
        hash ^= p[i];
        hash *= 16777619u;
    }
    return hash;
}
"""),
    (1, """
void handle_request(int sock) {
    char buf[1024];
    int  n;
    /* BUG: recv result not bounds-checked before eval()-style dispatch */
    n = recv(sock, buf, sizeof(buf), 0);
    buf[n] = '\0';
    eval(buf);        /* dangerous eval of untrusted input */
    send(sock, "OK", 2, 0);
}
"""),
    (0, """
int parse_int(const char *s, int *out) {
    char *end;
    long  val;
    if (!s || !*s) return -1;
    errno = 0;
    val   = strtol(s, &end, 10);
    if (errno != 0 || *end != '\0') return -1;
    if (val < INT_MIN || val > INT_MAX)  return -1;
    *out = (int)val;
    return 0;
}
"""),
    (1, """
void run_subprocess(char *cmd) {
    /* BUG: os.system / subprocess with shell=True equivalent in C */
    char full_cmd[512];
    snprintf(full_cmd, sizeof(full_cmd), "bash -c \"%s\"", cmd);
    FILE *fp = popen(full_cmd, "r");
    char  line[256];
    while (fgets(line, sizeof(line), fp)) {
        fputs(line, stdout);
    }
    pclose(fp);
}
"""),
    (0, """
void linked_list_push(Node **head, int val) {
    Node *node = malloc(sizeof(Node));
    if (!node) return;
    node->val  = val;
    node->next = *head;
    *head      = node;
}
"""),
]

# Repeat to fill NUM_SAMPLES
SYNTHETIC_SAMPLES = [
    {"id": i % len(_RAW_SYNTHETIC), "code": code, "label": label, "project": "synthetic"}
    for i, (label, code) in enumerate(_RAW_SYNTHETIC * 30)
]


# Known working Devign dataset variants on HF Hub (tried in order)
_DEVIGN_HF_NAMES = [
    ("celinelee/devign",  "train", "func",   "target"),
    ("mr-abims0/devign",  "train", "func",   "target"),
    ("d4n1/Devign",       "train", "func",   "target"),
]


def _load_devign_hf(n: int) -> List[Dict]:
    """Try each known Devign variant on HF Hub; return first that works."""
    rng = random.Random(RANDOM_SEED)
    for ds_name, split, code_col, label_col in _DEVIGN_HF_NAMES:
        try:
            ds = load_dataset(ds_name, split=split)
            indices = rng.sample(range(len(ds)), min(n, len(ds)))
            samples = []
            for idx, row in enumerate(ds.select(indices)):
                code = row.get(code_col, row.get("code", ""))
                lbl  = int(row.get(label_col, row.get("label", 0)))
                samples.append({
                    "id":      idx,
                    "code":    code,
                    "label":   lbl,
                    "project": row.get("project", "unknown"),
                })
            print(f"[data_loader] Loaded {len(samples)} samples from '{ds_name}'.")
            return samples
        except Exception as exc:
            print(f"[data_loader] '{ds_name}' failed: {exc}")
    return []


def _load_bigvul_hf(n: int) -> List[Dict]:
    """Load from HuggingFace Hub (benjaminjellis/bigvul)."""
    ds = load_dataset("benjaminjellis/bigvul", split="train")
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
