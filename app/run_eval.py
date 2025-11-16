# app/grpo/run_eval.py
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL="meta-llama/Llama-3.1-8B"
ADAPTER="/workspace/ckpt/base-grpo/final"

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def infer(prompt):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.9,
                         max_new_tokens=256, pad_token_id=tok.eos_token_id,
                         eos_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

print(infer("Explain 12*13 and end with 'Final Answer: <number>'"))
