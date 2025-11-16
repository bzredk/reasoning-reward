# app/analyze_model.py
"""
Analyze and report internal structure of a HuggingFace causal LM:
- prints config (hidden_size, n_layers, n_heads, rope settings, vocab)
- counts total / trainable parameters
- per-layer module names and shapes (summary)
- saves a Markdown report to /workspace/ckpt/model_report.md
"""

import os, json, math
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/models/deepseek_r1d8b")
REPORT = "/workspace/ckpt/model_report.md"

print(f"Loading config from: {MODEL_DIR}")
cfg = AutoConfig.from_pretrained(MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

# 4-bit is enough for inspection on 16GB
bnb = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb
)

def human(n):
    return f"{n/1e9:.2f}B" if n>=1e9 else (f"{n/1e6:.2f}M" if n>=1e6 else f"{n}")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
vocab = getattr(cfg, "vocab_size", None)
hsize = getattr(cfg, "hidden_size", None)
layers = getattr(cfg, "num_hidden_layers", None)
n_heads = getattr(cfg, "num_attention_heads", None)
n_kv = getattr(cfg, "num_key_value_heads", None)
rope_theta = getattr(cfg, "rope_theta", None)
rope_scaling = getattr(cfg, "rope_scaling", None)
max_pos = getattr(cfg, "max_position_embeddings", None)
intermediate = getattr(cfg, "intermediate_size", None)
eps = getattr(cfg, "rms_norm_eps", None)

print("\n=== Model Config ===")
print(f"vocab_size: {vocab}")
print(f"hidden_size: {hsize}")
print(f"num_hidden_layers: {layers}")
print(f"num_attention_heads: {n_heads}  (KV heads: {n_kv})")
print(f"intermediate_size: {intermediate}")
print(f"rope_theta: {rope_theta}  rope_scaling: {rope_scaling}")
print(f"max_position_embeddings: {max_pos}")
print(f"rms_norm_eps: {eps}")

print("\n=== Parameter Counts ===")
print(f"Total params: {human(total)}  | Trainable: {human(trainable)}")

# Per-layer brief
names = []
for n, m in model.named_modules():
    # show only key blocks to keep it readable
    if any(k in n for k in [".layers.", ".embed_tokens", ".norm", ".lm_head"]) and all(
        bad not in n for bad in ["rotary_emb", "dropout"]
    ):
        names.append(n)
print(f"\nKey modules (sample {min(40,len(names))}/{len(names)}):")
for n in names[:40]:
    print(" -", n)

# Save Markdown report
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("# Model Internal Report\n\n")
    f.write(f"- **Path**: `{MODEL_DIR}`\n")
    f.write(f"- **vocab_size**: {vocab}\n")
    f.write(f"- **hidden_size**: {hsize}\n")
    f.write(f"- **num_hidden_layers**: {layers}\n")
    f.write(f"- **num_attention_heads**: {n_heads} (KV: {n_kv})\n")
    f.write(f"- **intermediate_size**: {intermediate}\n")
    f.write(f"- **rope_theta**: {rope_theta}  **rope_scaling**: {rope_scaling}\n")
    f.write(f"- **max_position_embeddings**: {max_pos}\n")
    f.write(f"- **rms_norm_eps**: {eps}\n\n")
    f.write(f"**Total params**: {human(total)}  | **Trainable**: {human(trainable)}\n\n")
    f.write("## Key modules (first 40)\n")
    for n in names[:40]:
        f.write(f"- `{n}`\n")
print(f"\nSaved report -> {REPORT}")
