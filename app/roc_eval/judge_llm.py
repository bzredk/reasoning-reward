# app/roc_eval/judge_llm.py
# Read narrative_with_prompts.csv, call OpenAI as LLM-as-a-judge,
# and save scores to narrative_scores.csv.

import os
import json
import argparse
import pandas as pd
from openai import OpenAI

JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for narrative story continuations.

You will be given:
- Beginning: the initial part of a short story.
- Model continuation: a candidate ending produced by a model.
- Reference continuation: a human-written gold ending.

Judge ONLY the model continuation, but you may use the reference
as a guide to what is plausible and complete.

Important:
- There can be multiple valid continuations.
- Treat the reference as a strong example, NOT the only truth.
- Reward candidates that are logically consistent with the beginning
  and internally coherent, even if they choose a different specific
  outcome from the reference.

Additional guidance by prompt type:
- For "loose" prompts: shorter but coherent continuations are acceptable.
- For "moderate" prompts: prefer answers that follow the requested
  structure and provide several concrete details.
- For "strict" prompts: strongly reward exact template adherence and
  fully filled slots; very short or underfilled outputs should get
  lower completeness and overall scores.

Score each dimension from 0 to 5 (integers only):

1) Information Completeness
- 0: irrelevant or empty
- 1: very short (<20 words), only one tiny conclusion
- 2: 20–50 words; covers only 1 key point
- 3: covers about half of the important events
- 4: covers most important events (>=75%)
- 5: covers almost all important events (>=90%), typically requires a
  multi-sentence, well-developed continuation (not just 1–2 short lines).

2) Factual Accuracy
- 0: contradicts core facts or basic physics
- 1: 2+ major factual/causal errors
- 2: 1 major + minor errors
- 3: at most one small error
- 4: only tiny ambiguities; essentially correct
- 5: all claims consistent with prompt, world knowledge, and causal logic

3) Relevance
- 0: completely off-topic
- 1: >50% off-topic
- 2: 30–50% off-topic
- 3: mostly relevant, some fluff
- 4: very minor redundancy (<10%)
- 5: fully on-topic, no unnecessary content

4) Logical Coherence & Clarity
- 0: self-contradictory or no chain of events
- 1: almost no reasoning; just a final line
- 2: many jumps; key steps missing
- 3: mostly coherent; a few leaps
- 4: clear, reproducible narrative logic
- 5: exceptionally clear, stepwise and consistent

5) Creativity & Expression
- 0: dull, template-like, no originality
- 1: minimal variation; very plain
- 2: one simple example or image
- 3: some vivid language or interesting detail
- 4: multiple fresh angles or details; fluent style
- 5: highly original, engaging, strong “wow” factor

6) Overall Quality
- 0: unreadable or seriously misleading
- 1: most other dimensions <=1
- 2: most other dimensions <=2
- 3: roughly medium quality
- 4: good quality; no dimension <=1
- 5: strong story; at least two dimensions =5

Think carefully but DO NOT output your reasoning.
Return ONLY a JSON object:

{
  "information_completeness": 0-5,
  "factual_accuracy": 0-5,
  "relevance": 0-5,
  "logical_coherence": 0-5,
  "creativity_expression": 0-5,
  "overall_quality": 0-5,
  "comments": "<short free-text comment in <=30 words>"
}
"""


def extract_json_block(text: str):
    """Same helper as in build_prompts.py."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to narrative_with_prompts.csv")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to narrative_scores.csv")
    parser.add_argument("--openai_model", type=str, default="gpt-4.1-mini",
                        help="OpenAI model name for judging.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(args.input)

    required_cols = {"id", "prompt_type", "beginning", "reference", "model_answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    rows = []
    total = len(df)

    for idx, row in df.iterrows():
        item_id = row["id"]
        prompt_type = str(row["prompt_type"])
        beginning = str(row["beginning"])
        reference = str(row["reference"])
        candidate = str(row["model_answer"])
        length_words = len(candidate.split())

        print(f"[{idx+1}/{total}] judging id={item_id}, type={prompt_type}...")

        user_input = f"""
Prompt type: {prompt_type}

Beginning:
{beginning}

Reference continuation:
{reference}

Model continuation:
{candidate}
"""

        resp = client.responses.create(
            model=args.openai_model,
            instructions=JUDGE_SYSTEM_PROMPT,
            input=user_input,
            temperature=0.0,  # deterministic judging
        )

        scores = extract_json_block(resp.output_text)

        base_overall = scores.get("overall_quality", 0)

        # Small length-based bonus for moderate/strict, capped at +1.0
        length_bonus = 0.0
        if prompt_type in ("moderate", "strict"):
            if length_words > 80:
                length_bonus = min(1.0, (length_words - 80) / 80.0)

        adjusted_overall = min(5, base_overall + length_bonus)

        out_row = {
            "id": item_id,
            "prompt_type": prompt_type,
            "information_completeness": scores.get("information_completeness"),
            "factual_accuracy": scores.get("factual_accuracy"),
            "relevance": scores.get("relevance"),
            "logical_coherence": scores.get("logical_coherence"),
            "creativity_expression": scores.get("creativity_expression"),
            "overall_quality": adjusted_overall,  # adjusted score
            "overall_raw": base_overall,          # original judge score
            "answer_length_words": length_words,
            "judge_comments": scores.get("comments", ""),
        }

        # Also keep any other columns from input, for easier analysis.
        for col in df.columns:
            if col not in out_row:
                out_row[col] = row[col]

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[done] wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
