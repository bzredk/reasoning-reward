# Data Description & Evaluation Protocol

> This file is a technical, structured description of the dataset and how it is evaluated.  
> It contains what the data includes, why these parts are used, how they are formatted/split,  
> and how we score model outputs. No claims beyond what is provided here.

---

## 1) Overview

We construct a **composite evaluation suite** with **four independent parts**; results are reported per part and (optionally) aggregated.

1. **Narrative (story) continuation** — prompts from **ROCStories**  
   • Use the first 1–2 sentences as the **prompt**; the 5th sentence is the **reference**.

2. **Lateral reasoning** — stems **adapted** (paraphrased) from **Paul Sloane & Des MacHale**.

3. **Curiosity-driven planning** — open-ended scenarios **reframed** from **Reddit r/AskReddit** and **_The Book of Questions_**.

4. **Role-aware dialogue** — persona cards/scenes **adapted** from **PersonaChat (Kaggle mirror)**.

**Purpose.** We want to see how models behave **in each sub-domain** under different prompt tightness (Loose / Moderate / Strict), with emphasis on reasoning quality and hallucination control.

**Current status.** We **provide the Narrative subset** (train/dev/test).  
The other three subsets require extra processing (paraphrasing, provenance). Some items **do not have standard labels** (e.g., Curiosity, Role) and are intended for **evaluation/diagnosis** rather than supervised training.

---

## 2) Sources & Licensing

- **ROCStories** (five-sentence micro-stories) — [https://cs.rochester.edu/nlp/rocstories](https://cs.rochester.edu/nlp/rocstories)  
- **Lateral puzzles** by Paul Sloane & Des MacHale — stems paraphrased from published puzzle collections (no direct redistribution).  
- **r/AskReddit** — [https://www.reddit.com/r/AskReddit](https://www.reddit.com/r/AskReddit)  
- **_The Book of Questions_** by Gregory Stock — commercial source, cited but not redistributed.  
- **PersonaChat** (Kaggle mirror; research use) — [https://www.kaggle.com/datasets/athavale/personachat](https://www.kaggle.com/datasets/athavale/personachat)


All target datasets listed above have been located and verified for research use.  
However, most require further preprocessing or paraphrasing before integration into the unified evaluation suite.  
These ongoing curation and refinement steps will be maintained and updated in our shared Drive repository:  
[https://drive.google.com/drive/folders/1zCDUV7PSqC7BuSy7hsKIowQN2zZMjqjJ?usp=sharing](https://drive.google.com/drive/folders/1zCDUV7PSqC7BuSy7hsKIowQN2zZMjqjJ?usp=sharing)


---

## 3) Format & Fields

All files use **JSONL**, one example per line.

**Prompts-only (used for GRPO or inference-time evaluation):**
```json
{"task":"story","prompt":"Beginning: ..."}
{"task":"lateral","prompt":"A man pushes his car to a hotel and loses his fortune. Why?"}
{"task":"curiosity","prompt":"I want to switch careers within six months."}
{"task":"roleplay","prompt":"ROLE: terse spaceship engineer.\nSCENARIO: a coolant leak mid-flight."}
```

### Field Dictionary
Required:
- `task` : `story | lateral | curiosity | roleplay`
- `prompt` : input shown to the model

Optional:
- `reference` : gold text (Narrative only)
- `mode` : `loose | moderate | strict`
- `constraints` : hard requirements for `strict`
- `meta` : `source`, `license`, `lang`, `hash`
- `id` : stable identifier

---

## 4) Splits
Internal prompts follow **80/10/10** into `train/dev/test`, strictly de-duplicated (no cross-split overlap).  
External/official tests (e.g., ROCStories test) are held out for blind evaluation and not used for tuning.

| Part | Train | Dev | Test | Notes |
|---|---:|---:|---:|---|
| Narrative (ROCStories) | 62748 | 7844 | 7843 | references present |
| Lateral reasoning | TBD | TBD | TBD | answers present |
| Curiosity planning | TBD | TBD | TBD | prompts only |
| Role-aware dialogue | TBD | TBD | TBD | prompts only |

---

## 5) Three Prompt Levels
Each example is queried **three times**:
- **Loose** – open-ended, minimal constraints
- **Moderate** – guided structure (e.g., three narrative moves)
- **Strict** – hard template + entity reuse + causal/closure checks

Record decoding settings (`seed`, `temperature`, `top_p`, `max_new_tokens`). If multi-sampling is used, record the selection rule.

---

## 6) Scoring Rubric (API)

The judge API returns **0–5** per dimension. The rubric below is the **operational scale used by the evaluator**.

### Dimensions & Point Scale

| Dimension | 0 pts | 1 pt | 2 pts | 3 pts | 4 pts | 5 pts | Collection / Verification Method |
|---|---|---|---|---|---|---|---|
| **Information Completeness** | Irrelevant or no answer | <20 words; only one very brief conclusion | 20–50 words; covers only 1 key point | ≥50% of “golden key points” covered | ≥75% of golden key points covered | ≥90% of golden key points covered | Pre-define a checklist of golden key points; annotators tick off coverage; auto-compute coverage rate |
| **Factual Accuracy** | Core facts directly contradict prompt or real-world/physics | ≥2 core factual errors (incl. physics laws, unit errors) | 1 core error + ≥1 minor detail error | ≤1 ignorable small error; rest aligns with real-world/causal logic | Only minor wording ambiguity or peripheral data inaccuracy | All fact claims (incl. scientific, historical, physical/math relations) are correct | Annotators mark “error spans” + classify; automated exact-match/unit-test scripts; custom physics-formula checks where needed |
| **Relevance** | Completely off-topic | >50% of content unrelated to the prompt | 30–50% off-topic | ≤30% off-topic | Very minimal redundancy (<10%); overall on-point | Fully on-point; zero unnecessary content | Annotators highlight off-topic sentences; compute n-gram Jaccard overlap to flag anomalies |
| **Logical Coherence & Clarity** | Self-contradictory or no reasoning chain | Almost no chain-of-thought; only final answer | Chain has many jumps; key steps missing | Chain mostly complete; minor leaps | Clear, structured chain; reproducible reasoning | Exceptionally rigorous; each step explained & justified | Only scored when CoT is shown; annotators focus on logical jumps/missing steps; spot-check samples for consistency |
| **Creativity & Expression** | Pure template with no originality | Minimal variation; only 1–2 word tweaks | Contains a simple example or analogy | 1–2 examples/analogies; somewhat vivid language | Multiple fresh angles/analogies; fluent style | Highly original insights; surprising “wow” factor; engaging tone | Annotators add a short “highlight” comment (≤8 words); optional pairwise-ELO ranking to surface the most creative outputs |
| **Overall Quality** | Unreadable or seriously misleading | Majority of dimensions ≤1 pt | Majority of dimensions ≤2 pts | Median score ≈3 pts | Median ≥4 pts and no dimension ≤1 pt | All dimensions ≥4 pts, and at least 2 dimensions =5 pts | Auto-aggregate per-dimension scores via median or weighted average; spot-check a sample for alignment |

> **Narrative-specific checks (when background/reference exist):**  
> • Semantic overlap (e.g., BERTScore/ROUGE if used)  
> • Consistency (NLI entail vs. contradict on background→continuation; **HHEM** factual-consistency using the background as source)  
> • Strict constraints: sentence count, explicit callback, **no new named entities**

**Efficiency (reported separately):** CoT usage ratio, output tokens, latency.

---

## 7) Judge API Schema


**Request (per run)**
```json
{
  "id": "<sample_id>",
  "subset": "narrative|lateral|curiosity|role",
  "level": "loose|moderate|strict",
  "background": "...",
  "prompt": "...",
  "prediction": "..."
}
```

**Response**
```json
{
  "scores": {
    "information_completeness": 0-5,
    "factual_accuracy": 0-5,
    "relevance": 0-5,
    "logical_coherence": 0-5,
    "creativity_expression": 0-5,
    "overall_quality": 0-5
  },
  "notes": "optional brief comment"
}
```