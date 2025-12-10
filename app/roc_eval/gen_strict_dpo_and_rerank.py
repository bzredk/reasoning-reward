# app/roc_eval/gen_strict_dpo_and_rerank.py
# Input: narrative_with_prompts_test500.csv
# Filter strict rows, then generate:
#   1) strict + DPO single sample
#   2) strict + DPO + RM rerank (N=3)
# Output two new CSVs under data/rocstories/
#
# IMPORTANT:
# - Keep prompt_type == "strict" (do NOT change to strict_dpo_xxx)
# - Use setting column to distinguish experiment arms
# - Use prompt_text (already strict row's prompt_text); fallback to prompt_strict

import os
import argparse
from typing import List, Tuple

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel


def load_dpo_generator(base_model: str, dpo_dir: str, load_in_4bit: bool):
    bnb = None
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, dpo_dir)
    model.eval()
    return tok, model


def load_reward_model(base_model: str, rm_dir: str, load_in_4bit: bool):
    bnb = None
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_rm = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        problem_type="regression",
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    base_rm.config.pad_token_id = tok.pad_token_id

    rm = PeftModel.from_pretrained(base_rm, rm_dir)
    rm.eval()
    return tok, rm


@torch.no_grad()
def generate_n(
    tok,
    model,
    prompt: str,
    n: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        do_sample=True,
        num_return_sequences=n,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens if min_new_tokens and min_new_tokens > 0 else None,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    texts = []
    for i in range(out.size(0)):
        full = tok.decode(out[i], skip_special_tokens=True)
        if full.startswith(prompt):
            full = full[len(prompt):].lstrip()
        texts.append(full)
    return texts


@torch.no_grad()
def rm_score_texts(
    tok_rm,
    rm,
    prompt: str,
    completions: List[str],
    max_length: int,
) -> List[float]:
    texts = [prompt + "\n\n" + c for c in completions]
    enc = tok_rm(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(rm.device)

    out = rm(**enc)
    scores = out.logits.squeeze(-1).detach().float().cpu().tolist()
    if isinstance(scores, float):
        scores = [scores]
    return scores


def pick_best_by_rm(
    tok_rm,
    rm,
    prompt: str,
    completions: List[str],
    max_length: int,
) -> Tuple[str, float, int]:
    scores = rm_score_texts(tok_rm, rm, prompt, completions, max_length=max_length)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return completions[best_idx], float(scores[best_idx]), int(best_idx)


def main():
    parser = argparse.ArgumentParser()

    # INPUT SHOULD BE narrative_with_prompts_test500.csv
    parser.add_argument("--input_csv", type=str, required=True)

    parser.add_argument("--output_dpo_csv", type=str, required=True)
    parser.add_argument("--output_dpo_rm_csv", type=str, required=True)

    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--dpo_dir", type=str, required=True)
    parser.add_argument("--rm_dir", type=str, required=True)

    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--rm_max_length", type=int, default=512)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    df = pd.read_csv(args.input_csv)

    # sanity check for your schema
    required = {"id", "prompt_type", "prompt_text", "beginning", "prompt_strict"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    # filter strict only
    strict_df = df[df["prompt_type"].astype(str).str.lower() == "strict"].copy()
    strict_df.reset_index(drop=True, inplace=True)

    if len(strict_df) == 0:
        raise ValueError("No strict rows found in input CSV.")

    # load DPO generator and RM
    gen_tok, gen_model = load_dpo_generator(
        args.base_model, args.dpo_dir, load_in_4bit=args.load_in_4bit
    )
    rm_tok, rm_model = load_reward_model(
        args.base_model, args.rm_dir, load_in_4bit=args.load_in_4bit
    )

    out_rows_dpo = []
    out_rows_dpo_rm = []

    total = len(strict_df)
    for i, row in strict_df.iterrows():
        item_id = row["id"]

        # use strict row's prompt_text first
        prompt_text = str(row.get("prompt_text", "") or "").strip()
        if not prompt_text:
            prompt_text = str(row.get("prompt_strict", "") or "").strip()

        # last-resort fallback
        if not prompt_text:
            beginning = str(row.get("beginning", "") or "")
            prompt_text = f"Beginning:\n{beginning}\n\nWrite a continuation in 3 to 5 sentences."

        print(f"[{i+1}/{total}] DPO strict id={item_id}")

        # 1) strict + DPO single sample
        single = generate_n(
            gen_tok, gen_model, prompt_text,
            n=1,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )[0]

        r1 = dict(row)
        r1["prompt_type"] = "strict"
        r1["setting"] = "strict_dpo_sample1"
        r1["model_answer"] = single
        out_rows_dpo.append(r1)

        # 2) strict + DPO + RM rerank N=3
        cands = generate_n(
            gen_tok, gen_model, prompt_text,
            n=args.num_candidates,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        best_text, best_score, best_idx = pick_best_by_rm(
            rm_tok, rm_model, prompt_text, cands, max_length=args.rm_max_length
        )

        r2 = dict(row)
        r2["prompt_type"] = "strict"
        r2["setting"] = f"strict_dpo_rm_rerank{args.num_candidates}"
        r2["model_answer"] = best_text
        r2["rm_score_best"] = best_score
        r2["rm_best_index"] = best_idx
        out_rows_dpo_rm.append(r2)

    out_df_dpo = pd.DataFrame(out_rows_dpo)
    out_df_dpo_rm = pd.DataFrame(out_rows_dpo_rm)

    os.makedirs(os.path.dirname(args.output_dpo_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_dpo_rm_csv), exist_ok=True)

    out_df_dpo.to_csv(args.output_dpo_csv, index=False, encoding="utf-8")
    out_df_dpo_rm.to_csv(args.output_dpo_rm_csv, index=False, encoding="utf-8")

    print(f"[done] strict + DPO sample1 -> {args.output_dpo_csv}")
    print(f"[done] strict + DPO + RM rerank{args.num_candidates} -> {args.output_dpo_rm_csv}")


if __name__ == "__main__":
    main()
