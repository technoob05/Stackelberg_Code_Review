"""
config.py
Global configuration and hyperparameters for the
Resource-Constrained Stackelberg Games for Code Review project.
"""

# ─── Dataset ─────────────────────────────────────────────────────────────────
DATASET_NAME   = "google/bigbench"   # Hugging Face dataset identifier (overridden in data_loader)
DEVIGN_DATASET = "claudios/Devign"   # Alternative Devign dataset on HF Hub
# We use a lightweight variant for reproducibility; can be swapped to BigVul:
USE_DATASET    = "devign"            # "devign" | "bigvul"

NUM_SAMPLES    = 500                 # How many PR samples to use in experiments
RANDOM_SEED    = 42

# ─── Risk Profiler ────────────────────────────────────────────────────────────
CHUNK_TOKEN_SIZE = 80                # Approximate tokens per code chunk (keep small so multi-line functions produce ≥2 chunks)
# Danger keywords that inflate the Ud (reward for catching) score
DANGER_KEYWORDS = [
    "strcpy", "strcat", "gets", "sprintf", "system(",
    "exec(", "eval(", "pickle.loads", "subprocess",
    "os.system", "deserializ", "jwt.decode",
    "SQL", "SELECT", "INSERT", "DROP TABLE",
    "crypto", "password", "token", "secret",
    "malloc", "free(", "memcpy", "buffer",
]

# Payoff defaults (can be overridden by vulnerability ground-truth)
DEFAULT_Ud = 1.0   # reward for catching a vulnerable chunk
DEFAULT_Ld = 2.0   # penalty for missing a vulnerable chunk (loss)
DEFAULT_U0 = 0.0   # reward for catching a safe chunk (no-ops)
DEFAULT_L0 = 0.0   # penalty for missing a safe chunk

# ─── Stackelberg Solver ───────────────────────────────────────────────────────
# Token budget as fraction of total chunks tokes (0 < BUDGET_RATIO <= 1)
BUDGET_RATIO = 0.40   # Defender can read at most 40% of total tokens

# ─── SLM Audit Agent ─────────────────────────────────────────────────────────
SLM_MODE     = "mock"         # "mock" | "hf_pipeline" | "vllm"
SLM_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"   # HF model if SLM_MODE == "hf_pipeline"
MAX_NEW_TOKENS = 256

# ─── Evaluation ───────────────────────────────────────────────────────────────
RESULTS_DIR  = "results"
RESULTS_FILE = "results/evaluation_results.csv"
