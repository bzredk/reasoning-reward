# app/rm/gen_and_rerank.py
# Generate candidates with a base LLM and rerank them with a trained RM LoRA.

import os
import json
import argparse
from typing import List, Tuple
import re

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel


# -----------------------------
# Meta cleaning utilities
# -----------------------------

META_PATTERNS = [
    r"^\s*no meta\.?\s*$",
    r"^\s*keep it clean\.?\s*$",
    r"^\s*keep it flowing.*$",
    r"^\s*no extra info\.?\s*$",
    r"^\s*no extra sentences.*$",
    r"^\s*only the story\.?\s*$",
    r"^\s*just the story\.?\s*$",
    r"^\s*no markdown\.?\s*$",
    r"^\s*no preface\.?\s*$",
    r"^\s*no quotes\.?.*$",
    r"^\s*you are a story continuation writer\.?.*$",
    r"^\s*write only the continuation.*$",
    r"^\s*do not include analysis.*$",
    r"^\s*do not mention.*$",
    r"^\s*beginning:\s*.*$",
    r"^\s*write a continuation.*$",
    r"^\s*example:\s*.*$",
    r"^\s*prompt type:\s*.*$",
]

REASONING_LEADS = (
    "okay, so",
    "alright, so",
    "let me think",
    "wait, but",
    "so, the continuation is",
    "the continuation is",
    "first, i should",
    "i need to",
)

INSTRUCTION_MARKERS = [
    "write a continuation",
    "write exactly",
    "do not repeat",
    "do not include",
    "output only",
    "no analysis",
]


def clean_meta_text(text: str) -> str:
    if not text:
        return text

    lines = text.splitlines()
    cleaned = []

    for ln in lines:
        s = ln.strip()
        low = s.lower()

        drop = False
        for pat in META_PATTERNS:
            if re.match(pat, low):
                drop = True
                break

        if not drop:
            for lead in REASONING_LEADS:
                if low.startswith(lead):
                    drop = True
                    break

        if not drop:
            cleaned.append(ln)

    out = "\n".join(cleaned).strip()

    # Hard cut if instruction markers still appear
    low_out = out.lower()
    for m in INSTRUCTION_MARKERS:
        idx = low_out.find(m)
        if idx != -1:
            out = out[:idx].strip()
            break

    # If too short after cleaning, treat as empty
    if len(out.split()) < 6:
        return ""

    return out


def take_first_story_sentences(text: str, n_max: int = 5) -> str:
    """
    Keep at most n_max sentences. Helps prevent long rambling meta.
    """
    if not text:
        return text

    # naive sentence split
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    parts = [p for p in parts if p.strip()]
    if not parts:
        return text.strip()

    kept = parts[:n_max]
    return " ".join(kept).strip()


# -----------------------------
# RM input building (match training format)
# -----------------------------

def extract_beginning_from_prompt(prompt: str) -> str:
    """
    Extract the 'Beginning' block from a ROCStories-style prompt.
    Designed to avoid leaking instruction text into RM inputs.
    """
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

        if s == "":
            break

        low = s.lower()
        if any(low.startswith(m) for m in INSTRUCTION_MARKERS):
            break

        out.append(ln)

    beginning = "\n".join(out).strip()

    low_b = beginning.lower()
    if any(m in low_b for m in INSTRUCTION_MARKERS):
        # extra safety fallback to first line only
        beginning = beginning.split("\n")[0].strip()

    return beginning.strip()


def safe_beginning_fallback(prompt: str) -> str:
    """
    Conservative fallback: never return the full prompt.
    Prefer:
    - text after 'Beginning:' on the same line
    - otherwise first non-empty line
    """
    if not prompt:
        return ""

    for ln in prompt.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith("beginning:"):
            after = ln.split(":", 1)[1].strip()
            return after if after else ""
        return s

    return ""


def build_rm_text(prompt: str, completion: str) -> str:
    beginning = extract_beginning_from_prompt(prompt)
    if not beginning:
        beginning = safe_beginning_fallback(prompt)

    return (
        "### BEGINNING\n"
        f"{beginning}\n\n"
        "### CONTINUATION\n"
        f"{completion}"
    ).strip()


# -----------------------------
# Model loading
# -----------------------------

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


def load_reward_model(base_model_id: str, rm_dir: str, debug_rm_head: bool = False):
    """
    RM was trained as a LoRA adapter on top of a sequence classification head.
    The base model does not provide 'score.weight' by default, so transformers
    may warn that the head is newly initialized. This is expected.
    """
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

    if debug_rm_head:
        try:
            score_module = None
            if hasattr(rm, "base_model") and hasattr(rm.base_model, "model"):
                score_module = getattr(rm.base_model.model, "score", None)
            if score_module is not None and hasattr(score_module, "weight"):
                norm = score_module.weight.data.float().norm().item()
                print(f"[debug] RM score weight norm: {norm:.6f}")
            else:
                print("[debug] RM score module not found on expected path.")
        except Exception as e:
            print(f"[debug] RM head check failed: {e}")

    return tok, rm


# -----------------------------
# Generation + scoring
# -----------------------------

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
    clean_meta: bool = False,
) -> List[str]:
    """
    Generate N candidates. Optionally apply meta-cleaning.
    """

    anti_meta = (
        "You are a story continuation writer.\n"
        "Write ONLY the continuation in 3 to 5 sentences.\n"
        "Do NOT include analysis, planning, rules, or meta text.\n"
        "Do NOT mention 'the user', 'the prompt', or instructions.\n\n"
    )
    gen_prompt = anti_meta + prompt

    inputs = tok(gen_prompt, return_tensors="pt").to(model.device)

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

        if full.startswith(gen_prompt):
            cand = full[len(gen_prompt):].lstrip()
        elif full.startswith(prompt):
            cand = full[len(prompt):].lstrip()
        else:
            cand = full.strip()

        if clean_meta:
            cand = clean_meta_text(cand)
            cand = take_first_story_sentences(cand, n_max=5)

        texts.append(cand.strip())

    return texts


@torch.no_grad()
def score_texts(
    tok,
    rm,
    prompt: str,
    completions: List[str],
    max_length: int,
) -> List[float]:
    texts = [build_rm_text(prompt, c) for c in completions]

    enc = tok(
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

    pairs = []
    for s, c in zip(scores, completions):
        wc = len(c.split()) if c else 0
        # Hard penalty for empty/meta-like ultra short outputs
        if wc < 6:
            s = float(s) - 10.0
        pairs.append((float(s), c))

    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text. If not set, read from --prompt_file.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to a txt file containing the prompt.")
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save ranked results as JSON.")
    parser.add_argument("--print_raw", action="store_true", help="Print raw candidates before reranking.")
    parser.add_argument("--clean_meta", action="store_true", help="Apply meta-cleaning to generated candidates.")
    parser.add_argument("--debug_rm_head", action="store_true", help="Print a small debug signal for RM head.")
    args = parser.parse_args()

    if args.prompt is None and args.prompt_file is None:
        raise ValueError("Provide --prompt or --prompt_file.")

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt.strip()

    # 1) load base generator
    gen_tok, gen_model = load_base_generator(args.base_model)

    # 2) load reward model
    rm_tok, rm_model = load_reward_model(
        args.base_model,
        args.rm_dir,
        debug_rm_head=args.debug_rm_head,
    )

    # 3) generate candidates
    completions = generate_candidates(
        gen_tok,
        gen_model,
        prompt,
        num_candidates=args.num_candidates,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        clean_meta=args.clean_meta,
    )

    if args.print_raw:
        print("\n=== Raw candidates (generation order, after optional clean) ===")
        for i, c in enumerate(completions):
            print(f"\n[{i}]\n{c if c else '[EMPTY AFTER CLEAN]'}")

    # 4) rerank
    ranked = rerank(
        rm_tok,
        rm_model,
        prompt,
        completions,
        max_length=args.max_length,
    )

    print("\n=== Prompt ===")
    print(prompt)
    print("\n=== Ranked completions (high -> low) ===")
    for i, (s, c) in enumerate(ranked):
        show = c if c else "[EMPTY AFTER CLEAN]"
        print(f"\n[{i}] RM_score={s:.4f}\n{show}")

    best_score, best_text = ranked[0]
    print("\n=== Best (RM top-1) ===")
    print(f"RM_score={best_score:.4f}")
    print(best_text if best_text else "[EMPTY AFTER CLEAN]")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        payload = {
            "base_model": args.base_model,
            "rm_dir": args.rm_dir,
            "prompt": prompt,
            "num_candidates": args.num_candidates,
            "generation": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
                "clean_meta": bool(args.clean_meta),
            },
            "ranked": [
                {"rank": i, "rm_score": float(s), "completion": c}
                for i, (s, c) in enumerate(ranked)
            ],
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[done] saved ranked results to {args.save_json}")


if __name__ == "__main__":
    main()
