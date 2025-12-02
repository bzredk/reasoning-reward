# app/roc_eval/gen_strict_completions.py
# 1) Read ROCStories-style train.csv (background, reference, etc.)
# 2) Use OpenAI (e.g. gpt-4.1-mini) to generate ONLY a strong "prompt_strict"
# 3) Call local base model (e.g. DeepSeek-R1) with prompt_strict to generate 2 completions
# 4) Save to JSONL for later judging / DPO / reward modeling

import os
import json
import argparse
import pandas as pd

from openai import OpenAI
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# ---------- OpenAI: generate strong prompt only ----------

PROMPT_SYSTEM_TEXT = """
You are helping construct an evaluation dataset for narrative story completion.

Given:
- a full gold story (background),
- a gold continuation (reference),
- and the beginning text shown to the model,

you must generate one English instruction:

prompt_strict:
  - Highly constrained template.
  - Explicit formatting requirements (lists, slots, tags, etc.).
  - Strongly controls what the model is allowed to output.

The prompt MUST:
- Be in English.
- Explicitly include the line: "Beginning: <beginning text>".
- Ask the model to write a continuation (not to choose between options).
- Be self-contained.

Return a JSON object with exactly this field:
{
  "prompt_strict": "..."
}
Do not add any extra fields or commentary.
"""


def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    return OpenAI(api_key=api_key)


def extract_json_block(text: str):
    """
    Robust JSON parse helper, compatible with ```json ... ``` wrapping.
    """
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def build_strict_prompt_for_item(
    client,
    openai_model: str,
    background: str,
    reference: str,
    beginning: str,
) -> str:
    """
    Use OpenAI once to get a single strong prompt: prompt_strict.
    """
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
        instructions=PROMPT_SYSTEM_TEXT,
        input=user_input,
        temperature=0.0,  # deterministic prompt template
    )

    # Keep the same pattern as other scripts (build_prompts.py / judge_llm.py)
    text = resp.output_text
    data = extract_json_block(text)
    return data["prompt_strict"]


# ---------- Local model (DeepSeek / Qwen etc.) ----------

_LOCAL_PIPELINE = None
_LOCAL_TOKENIZER = None


def init_local_pipeline(local_model_id: str):
    """Lazy-load local base model in 4-bit."""
    global _LOCAL_PIPELINE, _LOCAL_TOKENIZER
    if _LOCAL_PIPELINE is not None:
        return

    print(f"[local] loading base model: {local_model_id}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(
        local_model_id,
        use_fast=True,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",
    )

    _LOCAL_PIPELINE = pipe
    _LOCAL_TOKENIZER = tok
    print("[local] model loaded.")


def call_local_model(
    prompt: str,
    max_new_tokens: int = 256,
    num_return_sequences: int = 2,
):
    """
    Generate multiple completions with the strong prompt.
    Sampling config matches your requirement.
    """
    if _LOCAL_PIPELINE is None or _LOCAL_TOKENIZER is None:
        raise RuntimeError("Local pipeline is not initialized.")

    pipe = _LOCAL_PIPELINE
    tok = _LOCAL_TOKENIZER

    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        num_return_sequences=num_return_sequences,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return [o["generated_text"] for o in outputs]


# ---------- Helpers: guess beginning from background + reference ----------

def guess_beginning(background: str, reference: str) -> str:
    """
    Try to recover 'beginning' by removing the reference suffix if possible.
    Fallback: first two sentences.
    """
    bg = background.strip()
    ref = reference.strip()
    if ref and ref in bg:
        idx = bg.rfind(ref)
        if idx != -1:
            cand = bg[:idx].strip()
            if cand:
                return cand
    parts = bg.split(".")
    if len(parts) > 2:
        return ".".join(parts[:2]).strip() + "."
    return bg


# ---------- Main: generate strict completions ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="data/rocstories/train.csv",
        help="ROCStories csv file with 'background' and 'reference' columns.",
    )
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default="data/rocstories/strict_gen_train1000.jsonl",
        help="Output JSONL path for strict-prompt completions.",
    )
    parser.add_argument(
        "--local_model_id",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Local base model id (Hugging Face).",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name for strict prompt generation.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=1000,
        help="Max number of rows to process from CSV (for budget).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens per local completion.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    if "background" not in df.columns or "reference" not in df.columns:
        raise ValueError("Input CSV must contain 'background' and 'reference' columns.")

    client = get_openai_client()
    init_local_pipeline(args.local_model_id)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        total = len(df)
        for idx, row in df.iterrows():
            story_id = row.get("id", idx)
            background = str(row["background"])
            reference = str(row["reference"])
            beginning = guess_beginning(background, reference)

            print(f"[{idx+1}/{total}] story_id={story_id} -> generating strict prompt...")

            prompt_strict = build_strict_prompt_for_item(
                client=client,
                openai_model=args.openai_model,
                background=background,
                reference=reference,
                beginning=beginning,
            )

            print("  -> generating 2 completions with local model...")
            completions = call_local_model(
                prompt=prompt_strict,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=2,
            )

            # Each completion is one line in JSONL
            for comp_idx, comp in enumerate(completions):
                rec = {
                    "story_id": story_id,
                    "completion_id": comp_idx,
                    "prompt_strict": prompt_strict,
                    "prompt": prompt_strict,  # unified field name for later
                    "background": background,
                    "reference": reference,
                    "beginning": beginning,
                    "completion": comp,
                }
                # Keep any extra metadata from CSV (category, split, etc.)
                for col in df.columns:
                    if col not in rec:
                        rec[col] = row[col]

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote strict completions to {args.out_jsonl}")


if __name__ == "__main__":
    main()
