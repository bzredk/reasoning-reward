# app/grpo/rewards.py
import re, math

HEDGES = ("yeah", "so,", "alternatively", "wait", "actually", "i think")
NUMOPS = r"[0-9]"
OPS    = r"[\*\+\-\/×÷=]"

def extract_final_answer(txt: str):
    m = re.search(r"Final Answer:\s*([^\n\r]+)", txt, flags=re.I)
    return m.group(1).strip() if m else None

def orm_score(
    text: str,
    gold: str | None,
    information_completeness: float | None = None,
    factual_accuracy: float | None = None,
    relevance: float | None = None,
    logical_coherence: float | None = None,
    creativity_expression: float | None = None,
):
    """
    Outcome reward.

    Two modes:
    1) If judge scores are provided (all five dims not None):
       use their normalized sum as ORM:
         orm = (ic + fa + rel + lc + ce) / (5 * 5)

    2) Otherwise fall back to the old exact-match rule:
       - parse 'Final Answer: <...>' from text
       - compare with gold (numeric or string)
    """

    # ---------- Mode 1: use LLM-as-a-judge scores if available ----------
    judge_dims = [
        information_completeness,
        factual_accuracy,
        relevance,
        logical_coherence,
        creativity_expression,
    ]

    if all(d is not None for d in judge_dims):
        ic, fa, rel, lc, ce = [float(d) for d in judge_dims]
        sum_dims = ic + fa + rel + lc + ce
        # each dim is 0–5, there are 5 dims → max sum = 25
        orm = sum_dims / (5.0 * 5.0)
        # safety clamp into [0,1]
        return max(0.0, min(1.0, orm))

    # ---------- Mode 2: original exact-match fallback ----------
    if not gold:
        return 0.0

    pred = extract_final_answer(text)
    if pred is None:
        return 0.0

    # numeric exact match if possible
    try:
        return 1.0 if float(pred) == float(gold) else 0.0
    except Exception:
        return 1.0 if pred.strip() == str(gold).strip() else 0.0


def prm_score(text: str):
    """Process reward: short, math-dense, low-hedges, has equations, ends cleanly."""
    body = text.split("Final Answer:")[0]
    low  = body.lower()
    length = len(body)

    nums = len(re.findall(NUMOPS, body))
    ops  = len(re.findall(OPS, body))
    hed  = sum(h in low for h in HEDGES)
    has_eq = ("=" in body) or ("→" in body) or ("⇒" in body)

    # Normalize pieces into [0,1] ish
    math_density = min(1.0, (nums + ops) / max(1, length/20))
    hedge_pen    = min(1.0, hed * 0.25)
    len_pen      = max(0.0, (length - 400) / 400.0)  # soft penalty after 400 chars

    score = 0.35*math_density + (0.2 if has_eq else 0.0) - 0.25*hedge_pen - 0.2*len_pen
    return max(0.0, min(1.0, score))

def total_reward(text: str, gold: str | None, alpha=0.5, beta=0.5, length_lambda=0.0005):
    """Final reward for GRPO: α*ORM + β*PRM − λ*len."""
    o = orm_score(text, gold)
    p = prm_score(text)
    L = len(text)
    return alpha*o + beta*p - length_lambda*L, {"orm": o, "prm": p, "len": L}
