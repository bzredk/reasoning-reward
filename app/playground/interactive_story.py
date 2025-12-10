# /workspace/app/playground/interactive_story.py
# Interactive playground (memory-safe):
# - base single
# - base + RM rerank (N=3)
# - DPO single (adapter on same base)
# - DPO + RM rerank (N=3)

import argparse
from typing import List, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel


def make_bnb(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_generator(base_model: str, load_in_4bit: bool):
    bnb = make_bnb(load_in_4bit)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


def load_reward_model(base_model: str, rm_dir: str, load_in_4bit: bool):
    bnb = make_bnb(load_in_4bit)

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
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=n,
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
def score_texts(
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


def rerank(
    tok_rm,
    rm,
    prompt: str,
    completions: List[str],
    max_length: int,
) -> List[Tuple[float, str]]:
    scores = score_texts(tok_rm, rm, prompt, completions, max_length=max_length)
    pairs = list(zip(scores, completions))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs


def print_menu(has_rm: bool, has_dpo: bool):
    print("\nChoose a mode:")
    print("  1) base (single sample)")
    print("  2) base + RM rerank (N=3)" if has_rm else "  2) base + RM rerank (N=3)  [RM not loaded]")
    print("  3) DPO (single sample)" if has_dpo else "  3) DPO (single sample)    [DPO not loaded]")
    print("  4) DPO + RM rerank (N=3)" if (has_dpo and has_rm) else "  4) DPO + RM rerank (N=3)  [RM/DPO not loaded]")
    print("  q) quit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--rm_dir", type=str, default=None)
    parser.add_argument("--dpo_dir", type=str, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")

    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()

    # 1) load base generator ONCE
    print("[INFO] Loading BASE generator (single instance)...")
    gen_tok, gen_model = load_generator(args.base_model, args.load_in_4bit)

    # 2) attach DPO adapter on SAME base instance
    dpo_tok = dpo_model = None
    if args.dpo_dir:
        print("[INFO] Attaching DPO adapter on the SAME BASE instance...")
        dpo_tok = gen_tok
        dpo_model = PeftModel.from_pretrained(gen_model, args.dpo_dir)
        dpo_model.eval()

    # 3) load RM (separate seq-cls model)
    rm_tok = rm_model = None
    if args.rm_dir:
        print("[INFO] Loading RM (sequence classification + LoRA)...")
        rm_tok, rm_model = load_reward_model(args.base_model, args.rm_dir, args.load_in_4bit)

    has_rm = rm_model is not None
    has_dpo = dpo_model is not None

    print("\n[READY] Interactive playground.")
    print("Enter a prompt. Empty input will be skipped.\n")

    while True:
        prompt = input("\n=== Enter prompt (or 'q' to quit) ===\n> ").strip()
        if not prompt:
            continue
        if prompt.lower() in ("q", "quit", "exit"):
            break

        print_menu(has_rm, has_dpo)
        mode = input("> ").strip().lower()
        if mode in ("q", "quit", "exit"):
            break

        if mode == "1":
            outs = generate_n(
                gen_tok, gen_model, prompt,
                n=1,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            print("\n=== BASE OUTPUT ===")
            print(outs[0])

        elif mode == "2":
            if not has_rm:
                print("[WARN] RM not loaded. Provide --rm_dir.")
                continue
            cands = generate_n(
                gen_tok, gen_model, prompt,
                n=args.num_candidates,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            ranked = rerank(rm_tok, rm_model, prompt, cands, max_length=args.max_length)

            print("\n=== Candidates (generation order) ===")
            for i, c in enumerate(cands):
                print(f"\n[{i}]\n{c}")

            print("\n=== RM Ranked (high -> low) ===")
            for i, (s, c) in enumerate(ranked):
                print(f"\n[{i}] RM_score={s:.4f}\n{c}")

            print("\n=== BEST (BASE + RM) ===")
            print(f"RM_score={ranked[0][0]:.4f}")
            print(ranked[0][1])

        elif mode == "3":
            if not has_dpo:
                print("[WARN] DPO not loaded. Provide --dpo_dir.")
                continue
            outs = generate_n(
                dpo_tok, dpo_model, prompt,
                n=1,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            print("\n=== DPO OUTPUT ===")
            print(outs[0])

        elif mode == "4":
            if not (has_dpo and has_rm):
                print("[WARN] Need both --dpo_dir and --rm_dir.")
                continue
            cands = generate_n(
                dpo_tok, dpo_model, prompt,
                n=args.num_candidates,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            ranked = rerank(rm_tok, rm_model, prompt, cands, max_length=args.max_length)

            print("\n=== Candidates (generation order) ===")
            for i, c in enumerate(cands):
                print(f"\n[{i}]\n{c}")

            print("\n=== RM Ranked (high -> low) ===")
            for i, (s, c) in enumerate(ranked):
                print(f"\n[{i}] RM_score={s:.4f}\n{c}")

            print("\n=== BEST (DPO + RM) ===")
            print(f"RM_score={ranked[0][0]:.4f}")
            print(ranked[0][1])

        else:
            print("[WARN] Unknown mode. Please choose 1/2/3/4/q.")

    print("\n[bye]")


if __name__ == "__main__":
    main()
