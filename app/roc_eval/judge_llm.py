# app/roc_eval/judge_llm.py
# Read narrative_with_prompts.csv, call OpenAI as LLM-as-a-judge,
# and save scores to narrative_scores.csv.
#
# Scoring:
# - overall_raw = average of 5 dimensions
# - overall_quality = overall_raw + gated length bonus - verbosity penalty
#
# Length bonus (v2):
# - < 30 words: 0
# - 30–80 words: linear up to +1.0
# - > 80 words: capped at +1.0
# - bonus 再乘以 lambda_gate(ic, rel) 做门控
# - 长且低相关/低连贯会有 verbosity_penalty
# - overall_quality clipped to [0, 5]

import os
import json
import time
import argparse
import pandas as pd
from typing import Any, Dict, Optional
from openai import OpenAI


JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for narrative story continuations.

You will be given:
- Beginning: the initial context for the story.
- Model continuation: the candidate continuation.

Judge ONLY the model continuation.

This is a story continuation faithfulness task.
The continuation should be semantically consistent with the context and should not introduce hallucinated
core entities/events that are unsupported by the beginning.

Score each dimension from 0 to 5.
Use decimals with two digits when helpful (e.g., 3.20, 4.75).
A score of 5 should be rare and reserved for truly exceptional cases.

1) Information Completeness (0–5)
Measures whether the continuation meaningfully completes the narrative implied by the context
given the requested length/format.
- 0: empty, nonsensical, or unrelated continuation.
- 1: extremely thin ending; only a minimal closure with little narrative content.
- 2: partial ending that addresses only one small aspect of the situation; feels underdeveloped.
- 3: reasonably completes the main thread but leaves key implied outcomes vague or abrupt.
- 4: provides a well-rounded ending that resolves most salient issues or tensions introduced by the context.
- 5: exceptionally complete and satisfying resolution; fully addresses the implied narrative arc
     without unnecessary filler.

2) Factual Accuracy & Context Faithfulness (0–5)
This dimension includes faithfulness to the context’s entities, events, and causal state.
Penalize hallucinations here when they introduce unsupported core facts or contradictions.
- 0: multiple severe contradictions with the context or impossible events that break basic logic/physics.
- 1: clear contradiction with context facts/causal chain, or introduces a major impossible claim.
- 2: introduces a major unsupported event/entity that alters the story state or meaning
     (even if not an explicit contradiction).
- 3: adds unsupported detail that noticeably changes interpretation but is not a direct contradiction.
- 4: minor unverifiable detail that does not affect core story logic; overall faithful.
- 5: fully consistent with the context and commonsense; no invented facts that change the story state.

Guideline:
If the continuation introduces a new core entity/event not supported by the context,
deduct at least 1.0 point in this dimension.

3) Relevance & Constraint Following (0–5)
Measures how tightly the continuation stays on-topic and follows the prompt’s intent.
Penalize unnecessary or off-topic additions, especially new major plot elements not grounded in context.
- 0: completely off-topic or ignores key constraints.
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
""".strip()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def clip_score(x: float, lo: float = 0.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, x))


def extract_json_block(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction:
    - Accept raw JSON
    - Accept ```json ... ```
    - Try to locate first '{' ... last '}' if extra text appears
    """
    if text is None:
        raise ValueError("Empty judge response.")

    s = text.strip()

    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1].strip()
            if s.startswith("json"):
                s = s[4:].strip()

    # If still not valid JSON, attempt brace slicing
    if not s.startswith("{"):
        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            s = s[l:r+1].strip()

    return json.loads(s)


# ---- new length-related helpers ----

def length_bonus_v2(length_words: int) -> float:
    """
    Length bonus:
    - < 30 words: 0
    - 30–80 words: linear up to +1.0
    - > 80 words: capped at +1.0
    """
    if length_words < 30:
        return 0.0
    if length_words <= 80:
        return (length_words - 30) / 50.0
    return 1.0


def lambda_gate(ic: float, rel: float) -> float:
    """
    Gate for length bonus based on Information Completeness + Relevance.
    ic, rel in [0,5].
    """
    lam = (ic + rel - 4.0) / 6.0
    return max(0.0, min(1.0, lam))


def verbosity_penalty(length_words: int, rel: float, lc: float,
                      alpha: float = 0.25, beta: float = 0.25) -> float:
    """
    Penalize long but low-quality answers:
    - only trigger when length > 100
    - if relevance < 4.0 -> +alpha
    - if logical_coherence < 3.5 -> +beta
    """
    if length_words <= 100:
        return 0.0

    penalty = 0.0
    if rel < 4.0:
        penalty += alpha
    if lc < 3.5:
        penalty += beta
    return penalty


# ---- main judge glue ----

def build_user_input(
    beginning: str,
    candidate: str,
    prompt_type: str,
    reference: Optional[str] = None,
    include_reference: bool = False,
) -> str:
    # Keep judge input minimal to reduce bias.
    if include_reference and reference is not None and str(reference).strip():
        return f"""
Prompt type: {prompt_type}

Beginning:
{beginning}

Reference continuation:
{reference}

Model continuation:
{candidate}
""".strip()

    return f"""
Prompt type: {prompt_type}

Beginning:
{beginning}

Model continuation:
{candidate}
""".strip()


def judge_one(
    client: OpenAI,
    openai_model: str,
    user_input: str,
    max_retries: int = 3,
    sleep_sec: float = 1.0,
) -> Dict[str, Any]:
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=openai_model,
                instructions=JUDGE_SYSTEM_PROMPT,
                input=user_input,
                temperature=0.0,
            )
            data = extract_json_block(resp.output_text)
            return data
        except Exception as e:
            last_err = e
            print(f"[WARN] judge failed attempt={attempt}: {e}")
            time.sleep(sleep_sec * attempt)

    raise RuntimeError(f"Judge failed after {max_retries} attempts: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to narrative_with_prompts.csv")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to narrative_scores.csv")
    parser.add_argument("--openai_model", type=str, default="gpt-4.1-mini",
                        help="OpenAI model name for judging.")
    parser.add_argument("--include_reference", action="store_true",
                        help="Include reference continuation in judge input (default: False).")
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(args.input)

    required_cols = {"id", "prompt_type", "beginning", "model_answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    has_reference = "reference" in df.columns

    rows = []
    total = len(df)

    for idx, row in df.iterrows():
        item_id = row["id"]
        prompt_type = str(row.get("prompt_type", "")).strip().lower()
        beginning = str(row.get("beginning", ""))
        candidate = str(row.get("model_answer", ""))

        reference = None
        if has_reference:
            reference = row.get("reference", None)

        length_words = len(candidate.split())

        print(f"[{idx+1}/{total}] judging id={item_id}, type={prompt_type}...")

        user_input = build_user_input(
            beginning=beginning,
            candidate=candidate,
            prompt_type=prompt_type,
            reference=str(reference) if reference is not None else None,
            include_reference=args.include_reference,
        )

        scores = judge_one(
            client=client,
            openai_model=args.openai_model,
            user_input=user_input,
            max_retries=args.max_retries,
        )

        ic = clip_score(safe_float(scores.get("information_completeness", 0.0)))
        fa = clip_score(safe_float(scores.get("factual_accuracy", 0.0)))
        rel = clip_score(safe_float(scores.get("relevance", 0.0)))
        lc = clip_score(safe_float(scores.get("logical_coherence", 0.0)))
        ce = clip_score(safe_float(scores.get("creativity_expression", 0.0)))

        overall_raw = (ic + fa + rel + lc + ce) / 5.0

        # new length-aware scoring
        bonus_raw = length_bonus_v2(length_words)
        lam = lambda_gate(ic, rel)
        penalty = verbosity_penalty(length_words, rel, lc)
        bonus_effective = lam * bonus_raw

        overall_quality = clip_score(overall_raw + bonus_effective - penalty)

        out_row = {
            "id": item_id,
            "prompt_type": prompt_type,
            "information_completeness": ic,
            "factual_accuracy": fa,
            "relevance": rel,
            "logical_coherence": lc,
            "creativity_expression": ce,
            "overall_raw": overall_raw,
            "length_bonus_raw": bonus_raw,
            "length_bonus": bonus_effective,
            "verbosity_penalty": penalty,
            "overall_quality": overall_quality,
            "answer_length_words": length_words,
            "judge_comments": scores.get("comments", ""),
        }

        # Keep original columns for easier analysis
        for col in df.columns:
            if col not in out_row:
                out_row[col] = row[col]

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[done] wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
