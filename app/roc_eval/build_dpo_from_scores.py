# app/roc_eval/build_dpo_from_scores.py
# 从 strict_scored_train1000.jsonl 生成：
# 1) dpo_pairs_train1000.jsonl
# 2) reward_samples_train2000.jsonl

import os
import json
import argparse
from collections import defaultdict
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="data/rocstories/strict_scored_train1000.jsonl",
        help="Scored completions JSONL (6-dim scores).",
    )
    parser.add_argument(
        "--out_dpo",
        type=str,
        default="data/rocstories/dpo_pairs_train1000.jsonl",
        help="Output DPO pairs JSONL.",
    )
    parser.add_argument(
        "--out_reward",
        type=str,
        default="data/rocstories/reward_samples_train2000.jsonl",
        help="Output reward single-sample JSONL.",
    )
    args = parser.parse_args()

    # 1) 读所有样本，按 story_id 分组
    groups = defaultdict(list)
    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            sid = rec["story_id"]
            groups[sid].append(rec)

    print(f"[info] loaded {len(groups)} story groups")

    os.makedirs(os.path.dirname(args.out_dpo), exist_ok=True)

    # 2) reward: 每个 completion 一条（2000 条左右）
    with open(args.out_reward, "w", encoding="utf-8") as frew:
        for sid, recs in groups.items():
            for r in recs:
                s = r["scores"]
                reward_record = {
                    "story_id": sid,
                    "prompt": r["prompt"],
                    "completion": r["completion"],
                    "information_completeness": s["information_completeness"],
                    "factual_accuracy": s["factual_accuracy"],
                    "relevance": s["relevance"],
                    "logical_coherence": s["logical_coherence"],
                    "creativity_expression": s["creativity_expression"],
                    "overall_quality": s["overall_quality"],
                }
                frew.write(json.dumps(reward_record, ensure_ascii=False) + "\n")

    # 3) DPO 对：chosen / rejected 用 overall_quality 决定
    dpo_count = 0
    with open(args.out_dpo, "w", encoding="utf-8") as fdpo:
        for sid, recs in groups.items():
            if len(recs) < 2:
                continue
            if len(recs) > 2:
                # 如果以后你生成 >2 个，可以在这里挑最高的两个
                recs = recs[:2]

            r1, r2 = recs[0], recs[1]
            s1, s2 = r1["scores"], r2["scores"]

            o1, o2 = s1["overall_quality"], s2["overall_quality"]

            if o1 > o2:
                chosen, rejected = r1, r2
            elif o2 > o1:
                chosen, rejected = r2, r1
            else:
                # overall_quality 打平，用其他 5 维的和做 tie-break
                aux1 = (
                    s1["information_completeness"]
                    + s1["factual_accuracy"]
                    + s1["relevance"]
                    + s1["logical_coherence"]
                    + s1["creativity_expression"]
                )
                aux2 = (
                    s2["information_completeness"]
                    + s2["factual_accuracy"]
                    + s2["relevance"]
                    + s2["logical_coherence"]
                    + s2["creativity_expression"]
                )
                if aux1 > aux2:
                    chosen, rejected = r1, r2
                elif aux2 > aux1:
                    chosen, rejected = r2, r1
                else:
                    chosen, rejected = random.sample([r1, r2], 2)

            cd = chosen["scores"]
            rd = rejected["scores"]

            dpo_record = {
                "story_id": sid,
                "prompt": chosen["prompt"],
                "chosen": chosen["completion"],
                "rejected": rejected["completion"],
                "score_chosen": cd["overall_quality"],
                "score_rejected": rd["overall_quality"],
            }
            fdpo.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")
            dpo_count += 1

    print(f"[done] wrote {dpo_count} DPO pairs to {args.out_dpo}")
    print("[done] wrote reward samples to", args.out_reward)


if __name__ == "__main__":
    main()
