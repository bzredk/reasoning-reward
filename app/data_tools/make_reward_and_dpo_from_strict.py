import os
import json
import argparse
from collections import defaultdict


def normalize_score(s: float, score_min: float, score_max: float) -> float:
    if score_max <= score_min:
        return s
    s = max(score_min, min(score_max, s))
    return (s - score_min) / (score_max - score_min)


def get_raw_score(obj, preferred_key: str):
    if preferred_key and preferred_key in obj and obj[preferred_key] is not None:
        return float(obj[preferred_key])
    for k in ["overall_quality", "overall_raw"]:
        if k in obj and obj[k] is not None:
            return float(obj[k])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/rocstories/strict_scored_train1000.jsonl",
        help="Input scored jsonl."
    )
    parser.add_argument(
        "--out_reward",
        type=str,
        default="train/strict_scored_train1000_reward.jsonl",
        help="Output reward jsonl."
    )
    parser.add_argument(
        "--out_dpo",
        type=str,
        default="train/strict_scored_train1000_dpo.jsonl",
        help="Output dpo jsonl."
    )
    parser.add_argument(
        "--score_key",
        type=str,
        default="overall_quality",
        help="Primary score key to use when available."
    )
    parser.add_argument("--score_min", type=float, default=1.0)
    parser.add_argument("--score_max", type=float, default=5.0)
    parser.add_argument(
        "--epsilon_equal",
        type=float,
        default=0.0,
        help="Treat scores as equal if abs(diff) <= epsilon."
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_reward), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_dpo), exist_ok=True)

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"[INFO] Loaded {len(rows)} lines from {args.input}")

    reward_written = 0
    with open(args.out_reward, "w", encoding="utf-8") as w:
        for obj in rows:
            story_id = obj.get("story_id")
            prompt = obj.get("prompt", "")
            completion = obj.get("completion", "")

            raw = get_raw_score(obj, args.score_key)
            if raw is None:
                continue

            out = {
                "story_id": story_id,
                "prompt": prompt,
                "completion": completion,
                "score": round(normalize_score(raw, args.score_min, args.score_max), 6),
                "score_raw": raw,
                "information_completeness": obj.get("information_completeness"),
                "factual_accuracy": obj.get("factual_accuracy"),
                "relevance": obj.get("relevance"),
                "logical_coherence": obj.get("logical_coherence"),
                "creativity_expression": obj.get("creativity_expression"),
                "judge_comments": obj.get("judge_comments", obj.get("comments", "")),
                "source": obj.get("source", "unknown"),
            }
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            reward_written += 1

    print(f"[INFO] Wrote {reward_written} reward lines -> {args.out_reward}")

    groups = defaultdict(list)
    for obj in rows:
        sid = obj.get("story_id")
        if sid is not None:
            groups[sid].append(obj)

    def score_of(o):
        s = get_raw_score(o, args.score_key)
        return float(s) if s is not None else 0.0

    pair_count = 0
    skip_single = 0
    skip_equal = 0

    with open(args.out_dpo, "w", encoding="utf-8") as w:
        for sid, items in groups.items():
            if len(items) < 2:
                skip_single += 1
                continue

            items_sorted = sorted(items, key=score_of, reverse=True)
            best = items_sorted[0]
            worst = items_sorted[-1]

            s_best = score_of(best)
            s_worst = score_of(worst)

            if abs(s_best - s_worst) <= args.epsilon_equal:
                skip_equal += 1
                continue

            out = {
                "story_id": sid,
                "prompt": best.get("prompt", ""),
                "chosen": best.get("completion", ""),
                "rejected": worst.get("completion", ""),
                "score_chosen": s_best,
                "score_rejected": s_worst,
                "source_chosen": best.get("source", "unknown"),
                "source_rejected": worst.get("source", "unknown"),
            }
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            pair_count += 1

    print(f"[INFO] Wrote {pair_count} DPO pairs -> {args.out_dpo}")
    print(f"[INFO] DPO stats: skipped_single={skip_single}, skipped_equal={skip_equal}")


if __name__ == "__main__":
    main()
