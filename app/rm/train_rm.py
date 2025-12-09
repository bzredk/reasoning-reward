# Train a reward model (regression) from strict_scored_train1000_reward.jsonl
# Compatible with 16GB VRAM using 4-bit + LoRA.

import os
import json
import math
import argparse
from typing import Dict, Any, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def extract_beginning_from_prompt(prompt: str) -> str:
    if not prompt:
        return ""

    lines = prompt.splitlines()
    out = []
    found = False

    for ln in lines:
        s = ln.strip()

        if not found:
            if s.lower().startswith("beginning:"):
                found = True
                after = ln.split(":", 1)[1].strip()
                if after:
                    out.append(after)
            continue

        # found == True
        if s == "":
            break

        # Stop hard if instruction section begins
        low = s.lower()
        if low.startswith("write a continuation") or low.startswith("write a continuation of") \
           or low.startswith("do not repeat") or low.startswith("output only") \
           or low.startswith("no analysis"):
            break

        out.append(ln)

    return "\n".join(out).strip()



def load_reward_jsonl(path: str, score_field: str = "overall_raw") -> Dataset:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            prompt = rec.get("prompt", "")
            completion = rec.get("completion", "")

            # Some files may store dims at top-level
            ic = rec.get("information_completeness", None)
            fa = rec.get("factual_accuracy", None)
            rel = rec.get("relevance", None)
            lc = rec.get("logical_coherence", None)
            ce = rec.get("creativity_expression", None)

            # Determine label
            if score_field in rec and rec[score_field] is not None:
                label = float(rec[score_field])
            else:
                # fallback: average of 5 dims if available
                dims = [ic, fa, rel, lc, ce]
                dims = [float(x) for x in dims if x is not None]
                label = float(sum(dims) / len(dims)) if dims else float(rec.get("overall_quality", 0.0))

            beginning = extract_beginning_from_prompt(prompt)

            # Fallback: if extraction fails, keep prompt short by using first 2 sentences
            if not beginning:
                # small safe fallback
                beginning = prompt.strip()

            text = (
                "### BEGINNING\n"
                f"{beginning}\n\n"
                "### CONTINUATION\n"
                f"{completion}"
            ).strip()

            records.append({
                "text": text,
                "label": label,
            })

    return Dataset.from_list(records)


def tokenize_fn(tokenizer, max_length: int):
    def _fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    return _fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to reward JSONL.")
    parser.add_argument("--eval_path", type=str, default=None,
                        help="Optional eval JSONL.")
    parser.add_argument("--base_model", type=str,
                        default=os.environ.get("RM_BASE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
                        help="HF base model id.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save RM LoRA adapter and tokenizer.")
    parser.add_argument("--score_field", type=str, default="overall_raw",
                        help="Which field to use as regression label.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bf16", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit load
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Sequence classification head for regression
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        num_labels=1,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.problem_type = "regression"
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare for k-bit training + LoRA
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_cfg)

    # Datasets
    train_ds = load_reward_jsonl(args.train_path, score_field=args.score_field)
    train_ds = train_ds.map(tokenize_fn(tokenizer, args.max_length), batched=True)
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in ("input_ids", "attention_mask", "label")])

    eval_ds = None
    if args.eval_path:
        eval_ds = load_reward_jsonl(args.eval_path, score_field=args.score_field)
        eval_ds = eval_ds.map(tokenize_fn(tokenizer, args.max_length), batched=True)
        eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in ("input_ids", "attention_mask", "label")])

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        # Simple regression metrics
        mse = float(((preds - labels) ** 2).mean())
        mae = float((abs(preds - labels)).mean())
        return {"mse": mse, "mae": mae}

    # Training args
    fp16 = not args.use_bf16
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        save_total_limit=1,
        bf16=args.use_bf16,
        fp16=not args.use_bf16,
    )

    # Enable gradient checkpointing for memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Write a small manifest
    manifest = {
        "base_model": args.base_model,
        "train_path": args.train_path,
        "eval_path": args.eval_path,
        "score_field": args.score_field,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
    }
    with open(os.path.join(args.output_dir, "rm_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] saved RM LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
