"""
Simple inference script for DeepSeek models
Runs 4-bit quantized inference to save GPU memory (works on 16GB GPUs)
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# You can change this to another base model (e.g., Llama 8B)
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# --- Load tokenizer ---
print(f"Loading tokenizer for {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# --- Load model (4-bit quantized, fits in 16GB GPU) ---
print(f"Loading model {MODEL_ID} (4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,     # key for small GPU
)

# --- Create pipeline for text generation ---
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# --- Simple test prompt ---
prompt = "Explain who is steph curry. Answer in two sentences."

# --- Run generation ---
print("\nRunning inference...\n")
result = pipe(
    prompt,
    max_new_tokens=500,
    temperature=0.7,
    top_p=0.9,
)
print(result[0]["generated_text"])
