import os
import json
import argparse
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    from trl import DPOTrainer, DPOConfig
except Exception as e:
    raise ImportError("trl is required for DPO training.") from e


def load_dpo_jsonl(path: str) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            prompt = obj.get("prompt", "")
            chosen = obj.get("chosen", "")
            rejected = obj.get("rejected", "")

            if not prompt or not chosen or not rejected:
                continue

            rows.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "story_id": obj.get("story_id", ""),
                "score_chosen": obj.get("score_chosen", None),
                "score_rejected": obj.get("score_rejected", None),
                "source_chosen": obj.get("source_chosen", "unknown"),
                "source_rejected": obj.get("source_rejected", "unknown"),
            })

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=384)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bf16", action="store_true")

    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=20)

    parser.add_argument("--load_in_4bit", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quant config
    quant_cfg = None
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    # Base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # QLoRA prep
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Dataset
    train_ds = load_dpo_jsonl(args.train_path)

    # DPOConfig
    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.use_bf16,
        fp16=not args.use_bf16,
        report_to=[],
        remove_unused_columns=False,

        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    # --------- THIS IS THE IMPORTANT COMPAT FIX ----------
    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
    )

    try:
        # New TRL (0.2x)
        trainer = DPOTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        # Old TRL fallback
        trainer = DPOTrainer(**trainer_kwargs, tokenizer=tokenizer)
    # -----------------------------------------------------

    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    manifest = {
        "base_model": args.base_model,
        "train_path": args.train_path,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "load_in_4bit": bool(args.load_in_4bit),
    }
    with open(os.path.join(args.output_dir, "dpo_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] saved DPO LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
