import os
import re
import json
import argparse
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
- 1: clear contradiction with prompt facts/causal chain, or introduces a major impossible event.
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
    if text is None:
        raise ValueError("Empty judge response text (None).")

    text = text.strip()
    if not text:
        raise ValueError("Empty judge response text.")

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Judge response is not valid JSON: {text[:200]}")
    return json.loads(m.group(0))


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


def load_base_model(model_name_or_path: str, load_in_4bit: bool = False):
    kwargs = {"device_map": "auto"}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        **kwargs
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int,
                 do_sample: bool, temperature: float, top_p: float, top_k: int):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    output_ids = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()


def score_with_openai(client: OpenAI, model_name: str, prompt: str, completion: str):
    user_input = f"""
Beginning (prompt):
{prompt}

Model continuation:
{completion}
"""
    resp = client.responses.create(
        model=model_name,
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

    return {
        "information_completeness": ic,
        "factual_accuracy": fa,
        "relevance": rel,
        "logical_coherence": lc,
        "creativity_expression": ce,
        "overall_raw": overall_raw,
        "comments": scores.get("comments", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Local base model name or path."
    )
    parser.add_argument(
        "--input_scored_jsonl",
        type=str,
        default="data/rocstories/strict_scored_train1000.jsonl",
        help="Existing scored file with candidates per story."
    )
    parser.add_argument(
        "--output_local_scored_jsonl",
        type=str,
        default="data/rocstories/strict_scored_train1000_local.jsonl",
        help="Output scored file for local single completions."
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model for judging."
    )
    parser.add_argument(
        "--limit_stories",
        type=int,
        default=1000,
        help="Number of unique stories to process."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling for local model generation."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    os.makedirs(os.path.dirname(args.output_local_scored_jsonl), exist_ok=True)

    with open(args.input_scored_jsonl, "r", encoding="utf-8") as f:
        existing_rows = [json.loads(line) for line in f if line.strip()]

    existing_by_story = defaultdict(list)
    for r in existing_rows:
        existing_by_story[r["story_id"]].append(r)

    story_ids = sorted(existing_by_story.keys())
    story_ids = story_ids[:args.limit_stories]

    tokenizer, model = load_base_model(args.base_model, load_in_4bit=args.load_in_4bit)

    done_ids = set()
    if os.path.exists(args.output_local_scored_jsonl):
        with open(args.output_local_scored_jsonl, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sid = obj.get("story_id")
                    if sid:
                        done_ids.add(sid)
                except Exception:
                    continue

    remaining_ids = [sid for sid in story_ids if sid not in done_ids]
    print(f"[INFO] Found {len(done_ids)} completed. Remaining {len(remaining_ids)}/{len(story_ids)}.")

    with open(args.output_local_scored_jsonl, "a", encoding="utf-8") as fout:
        for i, sid in enumerate(remaining_ids, start=1):
            items = existing_by_story[sid]
            prompt = items[0]["prompt"]

            completion = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            print(f"[{i}/{len(remaining_ids)}] local-generate + judge story_id={sid}...")

            s = None
            for attempt in range(5):
                try:
                    s = score_with_openai(
                        client=client,
                        model_name=args.openai_model,
                        prompt=prompt,
                        completion=completion
                    )
                    break
                except Exception as e:
                    print(f"[WARN] judge failed story_id={sid} attempt={attempt+1}: {e}")

            if s is None:
                continue

            out = {
                "story_id": sid,
                "prompt": prompt,
                "completion": completion,
                "information_completeness": s["information_completeness"],
                "factual_accuracy": s["factual_accuracy"],
                "relevance": s["relevance"],
                "logical_coherence": s["logical_coherence"],
                "creativity_expression": s["creativity_expression"],
                "overall_quality": s["overall_raw"],
                "overall_raw": s["overall_raw"],
                "judge_comments": s["comments"],
                "source": "local_base",
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[done] wrote local scored -> {args.output_local_scored_jsonl}")


if __name__ == "__main__":
    main()
