# app/rm/use_rm_rerank.py
import os
import argparse
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel


def load_rm(base_model_id: str, rm_dir: str):
    """Load tokenizer + base classification model + LoRA RM head."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_cls = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=1,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )

    # IMPORTANT: set pad_token_id on config as well
    if getattr(base_cls.config, "pad_token_id", None) is None:
        base_cls.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_cls, rm_dir)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def score_texts(tokenizer, model, texts, max_length: int = 512):
    """Return a list of scalar scores (higher = better)."""
    if isinstance(texts, str):
        texts = [texts]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    outputs = model(**enc)
    # logits shape: [batch_size, 1]
    scores = outputs.logits.view(-1).tolist()
    return scores


def demo_rerank(args):
    # Example prompt + 3 candidate continuations
    prompt = (
        "Beginning: Al was taking a class for college. "
        "He needed a certain book for his class."
    )

    candidates = [
        "Al went to the campus library and asked the librarian for help. "
        "She showed him where the book was, and he checked it out before class.",
        "Al thought about the book for a while. He wondered if books were even "
        "necessary anymore and decided to play games instead.",
        "Al first checked the library catalog online, then visited the desk "
        "to see if the book was on reserve. When he found it, he thanked the "
        "librarian and hurried to finish his reading before the next lecture.",
    ]

    # If RM was trained on "prompt + completion", keep this pattern.
    texts = [
        prompt + "\n\nContinuation:\n" + c
        for c in candidates
    ]

    tokenizer, model = load_rm(args.base_model, args.rm_dir)

    scores = score_texts(tokenizer, model, texts, max_length=args.max_length)

    print("Prompt:")
    print(prompt)
    print("=" * 80)

    for i, (cand, s) in enumerate(zip(candidates, scores)):
        print(f"[{i}] score = {s:.4f}")
        print(cand)
        print("-" * 80)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    print(f"Best idx = {best_idx}, score = {scores[best_idx]:.4f}")
    print("Best candidate:")
    print(candidates[best_idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Base model ID used when training RM.",
    )
    parser.add_argument(
        "--rm_dir",
        type=str,
        default="/workspace/models/rm_ex2",
        help="Directory of the trained RM LoRA adapter.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length for RM scoring.",
    )
    args = parser.parse_args()
    demo_rerank(args)


if __name__ == "__main__":
    main()
