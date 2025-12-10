import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def load_tokenizer(prefer_dir: str, fallback_id: str):
    """
    Prefer tokenizer saved in adapter dir if present,
    because DPO training may have updated special tokens.
    """
    try:
        # Heuristic: if dir contains tokenizer files, load from dir
        files = set(os.listdir(prefer_dir)) if os.path.isdir(prefer_dir) else set()
        has_tok = any(
            f in files for f in [
                "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                "special_tokens_map.json"
            ]
        )
        if has_tok:
            tok = AutoTokenizer.from_pretrained(prefer_dir, use_fast=True, trust_remote_code=True)
            return tok
    except Exception:
        pass

    tok = AutoTokenizer.from_pretrained(fallback_id, use_fast=True, trust_remote_code=True)
    return tok


def load_base(model_id: str, load_in_4bit: bool = True):
    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quant,
        trust_remote_code=True,
    )
    model.eval()
    return model


@torch.no_grad()
def gen(tok, model, prompt, max_new_tokens=120, min_new_tokens=8,
        temperature=0.7, top_p=0.9, top_k=50):
    inputs = tok(prompt, return_tensors="pt", truncation=True).to(model.device)

    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,  # <-- 防止一上来 EOS
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)

    if text.startswith(prompt):
        cont = text[len(prompt):].lstrip()
        return cont if cont.strip() else text

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--dpo_dir", type=str, required=True)
    ap.add_argument("--prompt_file", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--min_new_tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    # base tokenizer
    tok_base = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tok_base.pad_token is None:
        tok_base.pad_token = tok_base.eos_token

    # dpo tokenizer (prefer adapter dir)
    tok_dpo = load_tokenizer(args.dpo_dir, args.base_model)
    if tok_dpo.pad_token is None:
        tok_dpo.pad_token = tok_dpo.eos_token

    # load two separate base models to avoid any shared state weirdness
    base_model = load_base(args.base_model, load_in_4bit=args.load_in_4bit)
    base_for_dpo = load_base(args.base_model, load_in_4bit=args.load_in_4bit)

    dpo_model = PeftModel.from_pretrained(base_for_dpo, args.dpo_dir)
    dpo_model.eval()

    print("\n=== PROMPT ===")
    print(prompt)

    print("\n=== BASE OUTPUT ===")
    base_out = gen(
        tok_base, base_model, prompt,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
    )
    print(base_out)

    print("\n=== DPO OUTPUT ===")
    dpo_out = gen(
        tok_dpo, dpo_model, prompt,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
    )
    print(dpo_out)


if __name__ == "__main__":
    main()
