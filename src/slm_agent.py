"""
slm_agent.py
Interacts with the Small Language Model (SLM) to audit code chunks.
Supports:
  - "mock": Simulates detection based on probability/heuristics.
  - "hf_pipeline": Uses HuggingFace transformers pipeline.
"""

import random
from typing import List, Dict

try:
    import torch
    from transformers import pipeline
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False

from src.config import SLM_MODE, SLM_MODEL_ID, MAX_NEW_TOKENS

class SLMAuditAgent:
    def __init__(self, mode: str = SLM_MODE, model_id: str = SLM_MODEL_ID):
        self.mode = mode
        self.model_id = model_id
        self.pipeline = None
        
        if self.mode == "hf_pipeline" and HF_TRANSFORMERS_AVAILABLE:
            print(f"[slm_agent] Loading model {model_id} ...")
            self.pipeline = pipeline(
                "text-generation",
                model=model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
        elif self.mode == "hf_pipeline":
            print("[slm_agent] transformers not installed. Falling back to mock mode.")
            self.mode = "mock"

    def audit_chunk(self, chunk_text: str, ground_truth: int, risk: float) -> bool:
        """
        Returns True if a vulnerability is 'detected', False otherwise.
        """
        if self.mode == "mock":
            # Mock detection logic:
            # If there is a bug (ground_truth=1), probability of detection is based on risk.
            # If there is no bug (ground_truth=0), probability of false positive is low.
            if ground_truth == 1:
                # Higher risk heuristic increases detection probability
                det_prob = 0.5 + (risk * 0.4) 
                return random.random() < det_prob
            else:
                # False positive rate
                return random.random() < 0.05
        
        elif self.mode == "hf_pipeline":
            prompt = self._build_prompt(chunk_text)
            outputs = self.pipeline(
                prompt, 
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            response = outputs[0]["generated_text"][len(prompt):].lower()
            return "vulnerable" in response or "security issue" in response or "bug" in response
            
        return False

    def _build_prompt(self, code: str) -> str:
        return f"""Audit the following code for security vulnerabilities. 
If you find a vulnerability, explain it and start your response with 'VULNERABLE'. 
Otherwise, start with 'SAFE'.

Code:
{code}

Output:"""

    def audit_multiple(self, chunks: List[Dict]) -> List[bool]:
        results = []
        for c in chunks:
            detected = self.audit_chunk(c["text"], c["label"], c["risk"])
            results.append(detected)
        return results
