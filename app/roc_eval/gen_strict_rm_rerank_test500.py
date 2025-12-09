# app/roc_eval/gen_strict_rm_rerank_test500.py
# Generate N candidates for strict prompts on test_500.csv
# Rerank with a trained RM LoRA and save the best answers.
#
# Expected input CSV columns:
# - id (optional but recommended)
# - prompt_strict (required)
# Other columns will be preserved in output.

import os
import json
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel


def load_base_generator(model_id: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


def load_reward_model(base_model_id: str, rm_dir: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_rm = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
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
def generate_candidates(
    tok,
    model,
    prompt: str,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    inputs = tok(prompt, return_tensors="pt", truncation=True).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_candidates,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    texts = []
    for i in range(out.size(0)):
        full = tok.decode(out[i], skip_special_tokens=True)
        if full.startswith(prompt):
            full = full[len(prompt):].lstrip()
        texts.append(full.strip())
    return texts


@torch.no_grad()
def score_candidates(
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

    outputs = rm(**enc)
    scores = outputs.logits.squeeze(-1).detach().float().cpu().tolist()
    if isinstance(scores, float):
        scores = [scores]
    return scores


def pick_best(completions: List[str], scores: List[float]):
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx, completions[best_idx], scores[best_idx]


def load_done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        if "id" in df.columns:
            return set(df["id"].astype(str).tolist())
    except Exception:
        pass
    return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/rocstories/test_500.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/rocstories/test_500_strict_rm_rerank3.csv",
    )
    parser.add_argument(
        "--rm_dir",
        type=str,
        required=True,
        help="Path to trained RM LoRA directory.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    )
    parser.add_argument(
        "--prompt_col",
        type=str,
        default="prompt_strict",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="id",
    )
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--save_candidates", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if args.prompt_col not in df.columns:
        raise ValueError(f"Missing column '{args.prompt_col}' in {args.input_csv}")

    if args.id_col not in df.columns:
        df[args.id_col] = [str(i) for i in range(len(df))]

    df[args.id_col] = df[args.id_col].astype(str)

    done_ids = load_done_ids(args.output_csv)

    gen_tok, gen_model = load_base_generator(args.base_model)
    rm_tok, rm_model = load_reward_model(args.base_model, args.rm_dir)

    out_rows: List[Dict[str, Any]] = []
    if os.path.exists(args.output_csv):
        try:
            old = pd.read_csv(args.output_csv)
            out_rows = old.to_dict(orient="records")
        except Exception:
            out_rows = []

    total = len(df)
    processed_now = 0

    for idx, row in df.iterrows():
        item_id = str(row[args.id_col])
        if item_id in done_ids:
            continue

        prompt = str(row[args.prompt_col])

        print(f"[{idx+1}/{total}] strict+RM rerank id={item_id}...")

        completions = generate_candidates(
            gen_tok,
            gen_model,
            prompt,
            num_candidates=args.num_candidates,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        scores = score_candidates(
            rm_tok,
            rm_model,
            prompt,
            completions,
            max_length=args.max_length,
        )

        best_idx, best_text, best_score = pick_best(completions, scores)

        out = dict(row)
        out["id"] = item_id
        out["prompt_type"] = "strict"
        out["setting"] = f"strict_rm_rerank{args.num_candidates}"
        out["model_answer"] = best_text
        out["rm_score_best"] = float(best_score)
        out["rm_best_index"] = int(best_idx)

        if args.save_candidates:
            out["rm_candidates_json"] = json.dumps(
                [{"i": i, "score": float(scores[i]), "text": completions[i]} for i in range(len(completions))],
                ensure_ascii=False
            )

        out_rows.append(out)
        processed_now += 1

        if processed_now % 20 == 0:
            pd.DataFrame(out_rows).to_csv(args.output_csv, index=False, encoding="utf-8")

    pd.DataFrame(out_rows).to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"[done] wrote {len(out_rows)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
