# app/roc_eval/gen_strict_completions.py
# 1) Read ROCStories-style train.csv
# 2) Use OpenAI to generate ONLY a strong "prompt_strict"
# 3) Use OpenAI to generate 2 completions with prompt_strict
# 4) Save to JSONL for later judging / DPO / reward modeling

import os
import json
import argparse
import re
import pandas as pd
from openai import OpenAI

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
- Explicitly require: "Write exactly 3 to 5 sentences."
- Explicitly require: "Do not repeat the beginning or the instructions."
- Explicitly require: "Do not include any preface like 'Just the story', 'No markdown', or similar."
- Explicitly require: "Output only the continuation. No analysis."
- Add a failure mode: If you cannot comply, output exactly: INVALID

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
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def build_strict_prompt_for_item(client, openai_model: str, background: str, reference: str, beginning: str) -> str:
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
        temperature=0.0,
    )
    data = extract_json_block(resp.output_text)
    return data["prompt_strict"]


# ---------- Helpers ----------

def guess_beginning(background: str, reference: str) -> str:
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


def clean_completion_text(text: str) -> str:
    if not text:
        return text
    t = text.strip()

    # Remove common prefatory junk
    t = re.sub(r'^(no quotes\.?|no markdown\.?|just the story\.?|no analysis\.?)\s*', '', t, flags=re.I)

    # Hard remove if model echoed instructions
    t = re.sub(r'(?is)beginning:\s.*?\n\n', '', t).strip()

    return t


def truncate_to_3_5_sentences(text: str, max_sents: int = 5) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s for s in sents if s]
    if len(sents) <= max_sents:
        return text.strip()
    return " ".join(sents[:max_sents]).strip()


# ---------- OpenAI: generate completions ----------

def call_openai_model(
    client,
    model: str,
    prompt: str,
    num_return_sequences: int = 2,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_output_tokens: int = 180,
):
    # We call the model multiple times to mimic num_return_sequences.
    # This is predictable and easy to debug.
    outs = []
    for _ in range(num_return_sequences):
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        outs.append(resp.output_text.strip())
    return outs


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/rocstories/train.csv")
    parser.add_argument("--out_jsonl", type=str, default="data/rocstories/strict_gen_train1000.jsonl")
    parser.add_argument("--openai_prompt_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--openai_gen_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--max_output_tokens", type=int, default=180)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    if "background" not in df.columns or "reference" not in df.columns:
        raise ValueError("Input CSV must contain 'background' and 'reference' columns.")

    client = get_openai_client()
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
                openai_model=args.openai_prompt_model,
                background=background,
                reference=reference,
                beginning=beginning,
            )

            if prompt_strict.strip() == "INVALID":
                print("  -> got INVALID prompt, skip.")
                continue

            print("  -> generating 2 completions with OpenAI...")
            raw_comps = call_openai_model(
                client=client,
                model=args.openai_gen_model,
                prompt=prompt_strict,
                num_return_sequences=2,
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=args.max_output_tokens,
            )

            # Clean + truncate
            comps = []
            for c in raw_comps:
                c = clean_completion_text(c)
                c = truncate_to_3_5_sentences(c, max_sents=5)
                comps.append(c)

            for comp_idx, comp in enumerate(comps):
                rec = {
                    "story_id": story_id,
                    "completion_id": comp_idx,
                    "prompt_strict": prompt_strict,
                    "prompt": prompt_strict,
                    "background": background,
                    "reference": reference,
                    "beginning": beginning,
                    "completion": comp,
                }
                for col in df.columns:
                    if col not in rec:
                        rec[col] = row[col]

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote strict completions to {args.out_jsonl}")


if __name__ == "__main__":
    main()
