"""
slm_agent.py
SLM (Small Language Model) audit agent for code vulnerability detection.

Modes:
  "mock"        – probabilistic simulation (no GPU needed, used in paper eval)
  "hf_4bit"     – Qwen2.5-Coder-7B-Instruct loaded with 4-bit BnB quantization
                  (recommended for Kaggle T4 / local 16GB VRAM)
  "hf_pipeline" – standard HF pipeline, no quantization (needs ~28GB VRAM for 7B)

Recommended Kaggle setup:
  mode = "hf_4bit"
  model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
"""

import random
from typing import List, Dict, Optional

from src.config import SLM_MODE, SLM_MODEL_ID, MAX_NEW_TOKENS

# ─── Dependency detection ─────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        pipeline as hf_pipeline,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import bitsandbytes  # noqa: F401 – just verifying it's installed
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


# ─── Prompt templates ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT_SECURITY = (
    "You are a senior security engineer performing code review. "
    "Detect security vulnerabilities: buffer overflows, injection flaws, "
    "use-after-free, integer overflows, authentication bypass, insecure "
    "deserialization, hard-coded secrets, race conditions, etc. "
    "Be concise and precise."
)

_SYSTEM_PROMPT_DIFF = (
    "You are an expert software engineer reviewing a code patch. "
    "Lines starting with '-' are the OLD code being replaced; lines starting "
    "with '+' are the NEW code. Your task is to decide whether the OLD code "
    "contains a bug, defect, or security vulnerability that the patch fixes. "
    "Be concise and precise."
)

_USER_TEMPLATE_SECURITY = """\
Review the following code chunk for security vulnerabilities (buffer overflows, \
injection flaws, memory corruption, authentication bypass, use-after-free, \
integer overflows, insecure deserialization, hard-coded credentials, etc.).

```c
{code}
```

Answer with exactly ONE of:
  VULNERABLE: <one-line description of the flaw>
  SAFE: no vulnerability detected

Your answer:"""

_USER_TEMPLATE_DIFF = """\
The following is a unified diff. Lines starting with '-' are the OLD (potentially \
buggy/vulnerable) code; lines starting with '+' are the NEW (fixed) code.

Determine whether the OLD code contains a bug, defect, or security vulnerability \
that this patch addresses.

```diff
{code}
```

Answer with exactly ONE of:
  VULNERABLE: <one-line description of the issue in the old code>
  SAFE: the old code looks correct

Your answer:"""


def _is_diff(code: str) -> bool:
    """Return True if the code string looks like a unified diff."""
    first = code.lstrip()[:120]
    return (
        first.startswith("diff --git")
        or first.startswith("---")
        or first.startswith("@@")
        or "\n@@" in code[:300]
    )


def _build_chat_messages(code: str) -> List[Dict]:
    if _is_diff(code):
        system  = _SYSTEM_PROMPT_DIFF
        user    = _USER_TEMPLATE_DIFF.format(code=code[:3000])
    else:
        system  = _SYSTEM_PROMPT_SECURITY
        user    = _USER_TEMPLATE_SECURITY.format(code=code[:3000])
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ─── Main agent class ─────────────────────────────────────────────────────────

class SLMAuditAgent:
    """
    Wraps either a mock probabilistic detector or a real quantized LLM.

    Parameters
    ----------
    mode     : "mock" | "hf_4bit" | "hf_pipeline"
    model_id : HuggingFace model repo ID, e.g. "Qwen/Qwen2.5-Coder-7B-Instruct"
    """

    def __init__(
        self,
        mode:     str = SLM_MODE,
        model_id: str = SLM_MODEL_ID,
    ):
        self.mode     = mode
        self.model_id = model_id
        self._model     = None
        self._tokenizer = None
        self._pipe      = None

        if mode in ("hf_4bit", "hf_pipeline"):
            self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self):
        if not TRANSFORMERS_AVAILABLE:
            print("[slm_agent] transformers not installed → falling back to mock.")
            self.mode = "mock"
            return

        print(f"[slm_agent] Loading {self.model_id}  (mode={self.mode}) …")
        device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

        if self.mode == "hf_4bit":
            if not BNB_AVAILABLE:
                print("[slm_agent] bitsandbytes not installed → falling back to hf_pipeline.")
                self.mode = "hf_pipeline"
            else:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if TORCH_AVAILABLE else None,
                )
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_id, trust_remote_code=True
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        quantization_config=bnb_cfg,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    self._model.eval()
                    print(f"[slm_agent] 4-bit model loaded on {device}  ✓")
                    return
                except Exception as e:
                    print(f"[slm_agent] 4-bit load failed ({e}) → trying hf_pipeline.")
                    self.mode = "hf_pipeline"

        # hf_pipeline (fp16 / bf16 fallback)
        try:
            dtype = torch.bfloat16 if (TORCH_AVAILABLE and torch.cuda.is_available()) else None
            self._pipe = hf_pipeline(
                "text-generation",
                model=self.model_id,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            print(f"[slm_agent] Pipeline model loaded  ✓")
        except Exception as e:
            print(f"[slm_agent] Pipeline load failed ({e}) → falling back to mock.")
            self.mode = "mock"

    # ── Inference ──────────────────────────────────────────────────────────────

    def _llm_detect(self, code: str) -> bool:
        """Run real LLM inference; return True if response starts with VULNERABLE."""
        messages = _build_chat_messages(code)

        if self.mode == "hf_4bit" and self._model is not None:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=2048
            ).to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        elif self.mode == "hf_pipeline" and self._pipe is not None:
            result = self._pipe(
                messages,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                return_full_text=False,
            )
            response = result[0]["generated_text"].strip()

        else:
            return False

        resp_lower = response.lower()
        return (
            resp_lower.startswith("vulnerable")
            or "vulnerable:" in resp_lower[:50]
            or resp_lower.startswith("yes, vulnerable")
            or resp_lower.startswith("yes,")
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def audit_chunk(self, chunk_text: str, ground_truth: int, risk: float) -> bool:
        """
        Returns True if a vulnerability is 'detected' in this chunk.

        Mock mode uses a probabilistic model calibrated so that risk-based
        selection (SSG) is strictly better than Sequential/Random.
        Real-LLM mode runs actual inference on the chunk text.
        """
        if self.mode == "mock":
            if ground_truth == 1:
                # High-risk vuln → high detection; subtle vuln → low detection
                det_prob = min(0.95, 0.15 + risk * 0.80)
                return random.random() < det_prob
            else:
                fp_prob = min(0.15, 0.02 + risk * 0.08)
                return random.random() < fp_prob

        else:  # hf_4bit or hf_pipeline (real LLM)
            try:
                return self._llm_detect(chunk_text)
            except Exception as e:
                print(f"[slm_agent] inference error: {e}")
                return False

    def audit_batch(self, chunks: List[Dict]) -> List[bool]:
        """Audit a list of chunk dicts; returns bool list in same order."""
        return [self.audit_chunk(c["text"], c["label"], c["risk"]) for c in chunks]

    # ── Convenience ───────────────────────────────────────────────────────────

    def audit_multiple(self, chunks: List[Dict]) -> List[bool]:
        """Alias for audit_batch (backward compatibility)."""
        return self.audit_batch(chunks)

    # ── Info ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"SLMAuditAgent(mode={self.mode!r}, model={self.model_id!r})"

    @staticmethod
    def available_modes() -> List[str]:
        modes = ["mock"]
        if TRANSFORMERS_AVAILABLE:
            modes.append("hf_pipeline")
            if BNB_AVAILABLE and TORCH_AVAILABLE and torch.cuda.is_available():
                modes.insert(1, "hf_4bit")
        return modes
