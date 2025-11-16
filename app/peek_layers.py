import sys, json, csv

if len(sys.argv) < 4:
    print("Usage: python peek_layers.py <layers.json> <steps.csv> <step_index>")
    sys.exit(1)

jpath, cpath, step_str = sys.argv[1], sys.argv[2], sys.argv[3]
S = int(step_str)

with open(cpath, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    rows = list(r)
actual_token = rows[S]["token_text"]

with open(jpath, "r", encoding="utf-8") as f:
    J = json.load(f)

step_rec = next(s for s in J["steps"] if s["step"] == S)
per_layer = step_rec["per_layer"]

def rank_in_topk(topk, token_text):
    for rank, item in enumerate(topk, 1):
        if item["token"] == token_text:
            return rank
    return None

hits, first_miss = 0, None
ranks = []
for L in per_layer:
    rk = rank_in_topk(L["topk"], actual_token)
    ranks.append((L["layer"], rk, L["topk"]))
    if rk is not None:
        hits += 1
    elif first_miss is None:
        first_miss = L["layer"]

print(f"[step {S}] actual token = {repr(actual_token)}")
print(f"coverage: {hits}/{len(per_layer)} layers contain it in top-k")
if first_miss is not None:
    print(f"first layer excluding it: {first_miss}")
else:
    print("all traced layers include it in top-k.")
    
HEAD = 3
print("\nSample layers around the first miss:")
for L, rk, topk in ranks[max(0,(first_miss or 0)-HEAD): (first_miss or 0)+HEAD+1]:
    show = [(x['token'], round(x['prob'],4)) for x in topk]
    print(f"layer {L:>2}: rk={rk}  topk={show}")
