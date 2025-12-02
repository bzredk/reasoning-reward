# app/roc_eval/judge_completions.py
# 读取 strict_gen_train1000.jsonl
# 用 OpenAI LLM-as-a-judge 打 6 维分
# 输出 strict_scored_train1000.jsonl

import os
import json
import argparse
from openai import OpenAI

JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for narrative story continuations.

You will be given:
- Beginning (prompt): the initial part or instructions.
- Model continuation: the candidate ending.

Judge ONLY the model continuation.

Score each dimension from 0 to 5 (integers or decimals allowed):

1) Information Completeness
- 0: irrelevant or empty
- 1: very short (<20 words), only one tiny conclusion
- 2: 20–50 words; covers only 1 key point
- 3: covers about half of the important events
- 4: covers most important events (>=75%)
- 5: covers almost all important events (>=90%)

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
- 1: most other dimensions ≤1
- 2: most other dimensions ≤2
- 3: roughly medium quality
- 4: good quality; no dimension ≤1
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
    """Handle ```json ... ``` wrapping if present."""
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
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="data/rocstories/strict_gen_train1000.jsonl",
        help="Generated completions JSONL.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="data/rocstories/strict_scored_train1000.jsonl",
        help="Output JSONL with scores.",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model for judging.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output_jsonl, "w", encoding="utf-8") as fout:

        lines = fin.readlines()
        total = len(lines)

        for i, line in enumerate(lines):
            rec = json.loads(line)
            story_id = rec["story_id"]
            prompt = rec["prompt"]
            completion = rec["completion"]

            print(f"[{i+1}/{total}] scoring story_id={story_id}...")

            user_input = f"""
Beginning (prompt):
{prompt}

Model continuation:
{completion}
"""

            resp = client.responses.create(
                model=args.openai_model,
                instructions=JUDGE_SYSTEM_PROMPT,
                input=user_input,
            )

            scores = extract_json_block(resp.output_text)

            out_record = {
                "story_id": story_id,
                "prompt": prompt,
                "completion": completion,
                "scores": {
                    "information_completeness": float(scores.get("information_completeness", 0.0)),
                    "factual_accuracy": float(scores.get("factual_accuracy", 0.0)),
                    "relevance": float(scores.get("relevance", 0.0)),
                    "logical_coherence": float(scores.get("logical_coherence", 0.0)),
                    "creativity_expression": float(scores.get("creativity_expression", 0.0)),
                    "overall_quality": float(scores.get("overall_quality", 0.0)),
                },
                "judge_comments": scores.get("comments", ""),
            }

            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"[done] wrote scored records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
