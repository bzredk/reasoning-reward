# app/roc_eval/build_prompts.py
# 1) Read ROCStories-style data.
# 2) Use OpenAI to generate loose/moderate/strict prompts.
# 3) Call local base model with each prompt.
# 4) Save expanded rows to narrative_with_prompts.csv.

import os
import json
import argparse
import pandas as pd

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# ---------- OpenAI client ----------

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    return OpenAI(api_key=api_key)


PROMPT_SYSTEM_TEXT = """
You are helping construct an evaluation dataset for narrative story completion.

Given:
- a full gold story (background),
- a gold continuation (reference),
- and the beginning text shown to the model,

you must generate three English instructions:

1. prompt_loose:
   - Very open-ended, creative.
   - Just ask for a continuation in the model’s own words.
   - No strict format.

2. prompt_moderate:
   - Some structure and planning (e.g. mention narrative moves).
   - Still natural language, but with mild constraints and a simple format.

3. prompt_strict:
   - Highly constrained template.
   - Explicit formatting requirements (lists, slots, tags, etc.).
   - Strongly controls what the model is allowed to output.

All three prompts MUST:
- Be in English.
- Explicitly include the “Beginning: <beginning text>”.
- Ask the model to write a continuation (not to choose between options).
- Be self-contained.

Return a JSON object with exactly these fields:
{
  "prompt_loose": "...",
  "prompt_moderate": "...",
  "prompt_strict": "..."
}
Do not add any extra fields or commentary.
"""


def extract_json_block(text: str):
    """Robustly parse JSON even if the model wraps it in ```json ...```."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # pick the middle block if format is ```json ... ```
        if len(parts) >= 3:
            text = parts[1]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def build_prompts_for_row(client, openai_model, background: str, reference: str, beginning: str):
    """Call OpenAI once to get loose/moderate/strict prompts for a single item."""
    instructions = PROMPT_SYSTEM_TEXT

    user_input = f"""
background:
{background}

reference:
{reference}

beginning:
{beginning}
"""

    resp = client.responses.create(
        model=openai_model,
        instructions=instructions,
        input=user_input,
        temperature=0.0,  # deterministic prompt templates
    )

    text = resp.output_text
    data = extract_json_block(text)

    return data["prompt_loose"], data["prompt_moderate"], data["prompt_strict"]


# ---------- Local base model (DeepSeek / Qwen) ----------

_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None
_LOCAL_PIPELINE = None


def init_local_model(local_model_id: str, max_new_tokens: int):
    """Lazy-load local base model in 4-bit."""
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_PIPELINE
    if _LOCAL_MODEL is not None:
        return

    print(f"[local-model] Loading base model: {local_model_id}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
    )
    _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
        local_model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(local_model_id, use_fast=True, trust_remote_code=True)
    if _LOCAL_TOKENIZER.pad_token is None:
        _LOCAL_TOKENIZER.pad_token = _LOCAL_TOKENIZER.eos_token

    _LOCAL_PIPELINE = pipeline(
        "text-generation",
        model=_LOCAL_MODEL,
        tokenizer=_LOCAL_TOKENIZER,
        device_map="auto",
    )
    print("[local-model] Loaded.")


def call_local_model(prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a continuation with local base model."""
    if _LOCAL_PIPELINE is None:
        raise RuntimeError("Local model is not initialized. Call init_local_model() first.")
    out = _LOCAL_PIPELINE(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=_LOCAL_TOKENIZER.eos_token_id,
        eos_token_id=_LOCAL_TOKENIZER.eos_token_id,
    )[0]["generated_text"]
    return out


# ---------- Main pipeline ----------

def guess_beginning(background: str, reference: str) -> str:
    """Try to recover the 'beginning' by removing the reference suffix."""
    bg = background.strip()
    ref = reference.strip()
    if ref and ref in bg:
        return bg.replace(ref, "").strip()
    # fallback: use the first 2 sentences as beginning
    parts = bg.split(".")
    if len(parts) > 2:
        return ".".join(parts[:2]).strip() + "."
    return bg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to narrative_raw.csv")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to narrative_with_prompts.csv")
    parser.add_argument("--openai_model", type=str, default="gpt-4.1",
                        help="OpenAI model name for prompt generation.")
    parser.add_argument("--local_model_id", type=str,
                        default=os.environ.get("LOCAL_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
                        help="Local base model id (Hugging Face).")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Optional limit for debugging.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens for local model generation.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    # We expect columns: id, background, reference (you can add more)
    if "background" not in df.columns or "reference" not in df.columns:
        raise ValueError("Input CSV must contain 'background' and 'reference' columns.")

    client = get_openai_client()
    init_local_model(args.local_model_id, args.max_new_tokens)

    rows = []
    total = len(df)
    for idx, row in df.iterrows():
        item_id = row.get("id", idx)
        background = str(row["background"])
        reference = str(row["reference"])
        beginning = guess_beginning(background, reference)

        print(f"[{idx+1}/{total}] id={item_id} – building prompts via OpenAI...")
        prompt_loose, prompt_mod, prompt_strict = build_prompts_for_row(
            client, args.openai_model, background, reference, beginning
        )

        # Call local model for each prompt type
        for p_type, p_text in [
            ("loose", prompt_loose),
            ("moderate", prompt_mod),
            ("strict", prompt_strict),
        ]:
            print(f"  -> generating local answer for {p_type}...")
            model_answer = call_local_model(p_text, max_new_tokens=args.max_new_tokens)

            out_row = {
                "id": item_id,
                "prompt_type": p_type,
                "prompt_text": p_text,
                "beginning": beginning,
                "reference": reference,
                "background": background,
                "model_answer": model_answer,
            }

            # Keep any other metadata columns if present
            for col in df.columns:
                if col not in out_row:
                    out_row[col] = row[col]

            rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[done] Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
