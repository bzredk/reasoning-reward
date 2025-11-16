"""
Trace DeepSeek-R1-Distill-Llama-8B internals step-by-step.

What you get per generated token:
- final logits/logprob/entropy
- hidden_states for each Transformer layer
- per-layer "logit lens": top-k next-token guesses at each layer
- (optional) attentions (off by default; very memory heavy)

Outputs:
- /workspace/ckpt/traces/<run_id>_steps.csv   (compact per-step summary)
- /workspace/ckpt/traces/<run_id>_layers.json (per-layer top-k, per step)
"""

import os, time, json, csv
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# -------------------- Config --------------------
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
PROMPT   = os.getenv("PROMPT", "Explain 12*13 and end with: Answer: <number>")
MAX_NEW  = int(os.getenv("MAX_NEW_TOKENS", "1024"))
GREEDY   = True                # False = sample with temperature/top_p
TEMPERATURE = 0.7
TOP_P       = 0.9

TRACE_ATTNS  = False           # True to record attentions (heavy)
TRACE_LAYERS = None            # e.g. [0, 4, 8, 12, ...]; None = all
TOPK_PER_LAYER = 5

RUN_ID  = time.strftime("%Y%m%d-%H%M%S")
OUT_DIR = "/workspace/ckpt/traces"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- Load model --------------------
print(f"[trace] Loading tokenizer: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

print(f"[trace] Loading model (4-bit): {MODEL_ID}")
bnb = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb,
)
model.eval()
model.config.output_hidden_states = True
model.config.output_attentions   = TRACE_ATTNS
device = model.device
print(f"[trace] Device: {device}")

# -------------------- Helpers --------------------
def topk_tokens(logits: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    probs = F.softmax(logits, dim=-1)
    vals, idx = probs.topk(k)
    return [{"token_id": i, "token": tok.decode([i]), "prob": float(p)}
            for p, i in zip(vals.tolist(), idx.tolist())]

def decode(ids):
    return tok.decode(ids, skip_special_tokens=True)

# -------------------- Prepare inputs --------------------
inputs    = tok(PROMPT, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
attn_mask = inputs["attention_mask"]
generated = input_ids.clone()
text_so_far = decode(generated[0].tolist())

use_cache = True
past_kv   = None

rows, layers_dump = [], []

print("\n[trace] Start generation\n")
for step in range(MAX_NEW):
    with torch.no_grad():
        out = model(
            input_ids = generated if past_kv is None else generated[:, -1:],
            attention_mask = attn_mask if past_kv is None else None,
            use_cache = use_cache,
            past_key_values = past_kv,
            output_hidden_states = True,
            output_attentions = TRACE_ATTNS,
        )

    logits        = out.logits[:, -1, :]     # [B,V]
    logprobs      = F.log_softmax(logits, dim=-1)
    probs         = torch.exp(logprobs)
    entropy       = float(-(probs * logprobs).sum(dim=-1).item())
    hidden_states = out.hidden_states        # tuple(len=layers+1)
    past_kv       = out.past_key_values if use_cache else None

    # choose next token
    if GREEDY:
        next_id = torch.argmax(logits, dim=-1)
    else:
        next_id = torch.multinomial(
            F.softmax(logits / TEMPERATURE, dim=-1), num_samples=1
        ).squeeze(-1)

    next_logprob = float(logprobs[0, next_id.item()].item())

    # per-layer "logit lens" (top-k predictions from each layer)
    trace_layers = (list(range(len(hidden_states)))
                    if TRACE_LAYERS is None else
                    [l for l in TRACE_LAYERS if l < len(hidden_states)])
    per_layer = []
    for li in trace_layers:
        hs = hidden_states[li][:, -1, :]  # last position
        # Llama-like heads usually expect final norm:
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            hproj = model.lm_head(model.model.norm(hs))
        else:
            hproj = model.lm_head(hs)
        per_layer.append({"layer": li, "topk": topk_tokens(hproj[0], TOPK_PER_LAYER)})

    # append token
    generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)
    new_text  = tok.decode(next_id.tolist(), skip_special_tokens=True)
    text_so_far += new_text

    rows.append({
        "step": step,
        "token_id": int(next_id.item()),
        "token_text": new_text,
        "next_logprob": next_logprob,
        "entropy": entropy,
        "text_len": len(text_so_far),
    })
    layers_dump.append({"step": step, "per_layer": per_layer})

    if next_id.item() == tok.eos_token_id:
        break

# -------------------- Save --------------------
csv_path  = os.path.join(OUT_DIR, f"{RUN_ID}_steps.csv")
json_path = os.path.join(OUT_DIR, f"{RUN_ID}_layers.json")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "trace_layers": trace_layers,
        "topk_per_layer": TOPK_PER_LAYER,
        "steps": layers_dump,
        "final_text": text_so_far,
    }, f, ensure_ascii=False, indent=2)

print(f"\n[trace] Wrote:\n - {csv_path}\n - {json_path}")
print(f"[trace] Final text:\n{text_so_far}")
