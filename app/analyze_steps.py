import sys, csv
from collections import deque

if len(sys.argv) < 2:
    print("Usage: python analyze_steps.py /workspace/ckpt/traces/<run>_steps.csv")
    sys.exit(1)

path = sys.argv[1]
rows = []
with open(path, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for x in r:
        rows.append({
            "step": int(x["step"]),
            "token_text": x["token_text"],
            "logp": float(x["next_logprob"]),
            "ent": float(x["entropy"]),
        })

def rolling_mean(vals, w=15):
    q, s = deque(), 0.0
    out = []
    for v in vals:
        q.append(v); s += v
        if len(q) > w: s -= q.popleft()
        out.append(s/len(q))
    return out

logps = [r["logp"] for r in rows]
ents  = [r["ent"]  for r in rows]
ma_lp = rolling_mean(logps, 21)
ma_en = rolling_mean(ents, 21)

scores = []
for i,(r,lp,ep) in enumerate(zip(rows, ma_lp, ma_en)):
    s = (lp - r["logp"]) + max(0.0, r["ent"] - ep)
    scores.append((s, i))

scores.sort(reverse=True)
TOPN = 10
print(f"Top {TOPN} suspicious steps (higher=more suspicious):")
for s,i in scores[:TOPN]:
    ctx = " ".join(x["token_text"] for x in rows[max(0,i-3):i+4])
    print(f"- step={i:>4}  score={s:.3f}  token='{rows[i]['token_text']}'  context: {ctx}")
