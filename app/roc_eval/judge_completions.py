# app/roc_eval/judge_completions.py
# Read strict_gen_train1000.jsonl
# Use OpenAI LLM-as-a-judge to score 5 dimensions
# Compute overall_raw locally (5-dim average)
# Output strict_scored_train1000.jsonl

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

This is a story continuation faithfulness task.
The continuation should be semantically consistent with the context and should not introduce hallucinated
core entities/events that are unsupported by the prompt.

Score each dimension from 0 to 5.
Use decimals with two digits when helpful (e.g., 3.20, 4.75).
A score of 5 should be rare and reserved for truly exceptional cases.

1) Information Completeness (0–5)
Measures whether the continuation meaningfully completes the narrative implied by the prompt
given the requested length/format.
- 0: empty, nonsensical, or unrelated continuation.
- 1: extremely thin ending; only a minimal closure with little narrative content.
- 2: partial ending that addresses only one small aspect of the situation; feels underdeveloped.
- 3: reasonably completes the main thread but leaves key implied outcomes vague or abrupt.
- 4: provides a well-rounded ending that resolves most salient issues or tensions introduced by the prompt.
- 5: exceptionally complete and satisfying resolution; fully addresses the prompt’s implied narrative arc
     without unnecessary filler.

2) Factual Accuracy & Context Faithfulness (0–5)
This dimension includes faithfulness to the prompt’s entities, events, and causal state.
Penalize hallucinations here when they introduce unsupported core facts or contradictions.
- 0: multiple severe contradictions with the prompt or impossible events that break basic logic/physics.
- 1: clear contradiction with prompt facts/causal chain, or introduces a major impossible claim.
- 2: introduces a major unsupported event/entity that alters the story state or meaning
     (even if not an explicit contradiction).
- 3: adds unsupported detail that noticeably changes interpretation but is not a direct contradiction.
- 4: minor unverifiable detail that does not affect core story logic; overall faithful.
- 5: fully consistent with the prompt and commonsense; no invented facts that change the story state.

Guideline:
If the continuation introduces a new core entity/event not supported by the prompt,
deduct at least 1.0 point in this dimension.

3) Relevance & Constraint Following (0–5)
Measures how tightly the continuation stays on-topic and follows the prompt’s instructions.
Penalize unnecessary or off-topic additions, especially new major plot elements not grounded in context.
- 0: completely off-topic or ignores key instructions.
- 1: largely off-topic or heavily violates constraints; more than half of content is irrelevant.
- 2: significant drift; introduces multiple unnecessary elements or ignores important constraints.
- 3: mostly relevant but includes noticeable fluff, redundancy, or mild instruction slippage.
- 4: strongly on-topic with only minor extra detail; follows constraints well.
- 5: fully on-topic, concise, and strictly follows constraints; no unnecessary content.

Guideline:
If the continuation introduces major new content unrelated to the provided context,
deduct in this dimension even if the writing is fluent.

4) Logical Coherence & Clarity (0–5)
Measures whether the continuation forms a clear, stepwise narrative logic with sensible transitions.
- 0: incoherent, self-contradictory, or no discernible chain of events.
- 1: almost no narrative reasoning; disconnected statements or an abrupt one-liner.
- 2: multiple logical jumps; missing key steps; hard to follow causal progression.
- 3: generally coherent with a few leaps or unclear transitions.
- 4: clear, plausible progression; easy to follow cause-effect and timeline.
- 5: exceptionally clear and well-structured narrative logic with smooth transitions and strong readability.

5) Creativity & Expression (0–5)
Measures originality and engaging writing while remaining faithful to the context.
Creativity should not excuse hallucination or contradiction.
- 0: dull, template-like, or unnatural phrasing; no originality.
- 1: very plain and repetitive; minimal stylistic effort.
- 2: somewhat acceptable but generic; few distinctive details.
- 3: includes some vivid language or interesting detail that fits the story.
- 4: multiple fresh but context-appropriate details; fluent and engaging style.
- 5: highly original yet perfectly context-consistent; memorable and emotionally effective.

Calibration rules:
- Do not default to 5s for well-formed text.
- Use 3.5–4.6 for good but not outstanding continuations.
- Reflect small but real differences with decimals when appropriate.
- Avoid giving identical overall impressions across many samples unless genuinely indistinguishable.

Think carefully but DO NOT output your reasoning.
Return ONLY a JSON object:

{
  "information_completeness": 0-5,
  "factual_accuracy": 0-5,
  "relevance": 0-5,
  "logical_coherence": 0-5,
  "creativity_expression": 0-5,
  "comments": "<short free-text comment in <=30 words>"
}
"""


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


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def clip_score(x: float, lo: float = 0.0, hi: float = 5.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


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
                temperature=0.0,
            )

            scores = extract_json_block(resp.output_text)

            ic = clip_score(safe_float(scores.get("information_completeness", 0.0)))
            fa = clip_score(safe_float(scores.get("factual_accuracy", 0.0)))
            rel = clip_score(safe_float(scores.get("relevance", 0.0)))
            lc = clip_score(safe_float(scores.get("logical_coherence", 0.0)))
            ce = clip_score(safe_float(scores.get("creativity_expression", 0.0)))

            overall_raw = (ic + fa + rel + lc + ce) / 5.0

            out_record = {
                "story_id": story_id,
                "prompt": prompt,
                "completion": completion,
                "information_completeness": ic,
                "factual_accuracy": fa,
                "relevance": rel,
                "logical_coherence": lc,
                "creativity_expression": ce,
                "overall_quality": overall_raw,
                "overall_raw": overall_raw,
                "judge_comments": scores.get("comments", ""),
            }

            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"[done] wrote scored records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
