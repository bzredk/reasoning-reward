# Reasoning Project

**Final Presentation Slides:**  
https://docs.google.com/presentation/d/1Lem_2CszmDKFT3NH2uDBCEG7Dk8QRoTtezvs2DWa2HE/edit?usp=sharing

---

# Submissions

All submission materials will be updated in the repository at:

**`submit/milestoneX/`**

Please refer to this directory for the latest submission files and documentation.

---
# Milestone 2

## Experiments & Evaluation Pipeline

All experiments use **DeepSeek-R1-Distill-Llama-8B (4-bit
quantization)** for local inference.\
Evaluation is conducted using **GPT-4.1-mini** as the judge model.

The complete pipeline includes:

-   **`build_prompts.py`** --- Generates three prompt variants per
    story\
-   **`judge_llm.py`** --- Implements rubric-based evaluation\
-   **`simple-baseline.py`** --- Direct generation without CoT\
-   **`strong-baseline.py`** --- Structured CoT generation

Full scoring details and rubric definitions are provided in
**`scoring.md`**.

### Running the Evaluation

``` bash
python -m app.roc_eval.judge_llm \
    --input narrative_with_prompts.csv \
    --output narrative_scores.csv \
    --openai_model gpt-4.1-mini
```

# Milestone 3 & Final Pre Warmup

---

##  Project Overview — Narrative Continuation Evaluation & Reward Model Training

This project evaluates narrative-continuation performance across multiple prompting setups and introduces a Reward Model (RM) to improve generation quality through reranking.

We use **ROCStories**, a dataset of short 5-sentence stories with a missing ending.  
The goal is to evaluate how well an LLM can produce a coherent continuation given different prompt strictness levels.

---

## 1.  Training Data Pipeline

We construct several JSONL datasets from 1000 ROCStories items.

---

### **1.1 strict_scored_train1000.jsonl**
- For each of 1000 story beginnings, a high-quality **LLM-as-Judge** scores **two completions**.
- Each completion is rated across five dimensions:
  - *Information Completeness*
  - *Factual Accuracy*
  - *Relevance*
  - *Logical Coherence*
  - *Creativity*
- Produces **2000 high-quality training samples**.

Example record (single scored completion):

```json
{
  "story_id": 12,
  "prompt": "...",
  "completion": "...",
  "information_completeness": 4.0,
  "factual_accuracy": 3.5,
  "relevance": 4.5,
  "logical_coherence": 4.0,
  "creativity_expression": 3.5,
  "overall_quality": 4.0,
  "overall_raw": 3.9
}
```
---

### **1.2 strict_scored_train1000_local.jsonl**
- A **small local LLM** generates an answer for each of the 1000 stories.
- These answers tend to be low quality → useful as **negative samples** for DPO.
- Also scored by LLM-as-Judge.

---

### **1.3 strict_scored_train1000_all.jsonl**
A combined dataset:

| Source | Count |
|-------|-------|
| Judge-scored answer A | 1000 |
| Judge-scored answer B | 1000 |
| Local model answer | 1000 |
| **Total** | **3000** |

Used for:
- **Reward Model (RM) training**  
- **DPO training** (preference pairs: Good vs Local)

---

## 2.  Reward Model (RM)

Training script:

```
app/rm/train_rm.py
```

- Uses a **regression head** on top of Llama-8B (LoRA + 4-bit quantization).
- Trains on all 3000 entries.
- Target = `overall_raw` (average of the five judge dimensions).
- The RM learns to assign higher scores to higher-quality continuations.

---

## 3.  RM-Guided Reranking

Script:

```
app/rm/gen_and_rerank.py
```

Pipeline:

1. Base LLM generates **N = 3 sampled continuations** (temperature + top-p).
2. RM scores each candidate.
3. Return the **highest-scoring** candidate as the selected answer.

This produces much stronger strict-prompt outputs.

---

## 4.  Evaluation Setup — Three Baselines + RM Rerank

We evaluate 500 ROCStories test examples using:

1. **Loose prompt + Base model**
2. **Moderate prompt + Base model**
3. **Strict prompt + Base model**
4. **Strict prompt + Base model + RM rerank (N=3)** ← *our improved method*
5. **Strict prompt + Base model + RM rerank (N=3) +DPO** ← *in progress*

Outputs judged by LLM-as-Judge with the **new rubric**.

### Result files:
```
data/rocstories/narrative_scores_test500_newrubric_base3_newlen.csv
data/rocstories/narrative_scores_test500_strict_rm_rerank3_newlen.csv
```

---

## 5.  Judging System (LLM-as-a-Judge)

Judging script:

```
app/roc_eval/judge_llm.py
```

Computes:

- Information Completeness  
- Factual Accuracy  
- Relevance  
- Logical Coherence  
- Creativity & Expression  

The judge outputs:

- **overall_raw** = mean of five scores  
- **overall_quality** = overall_raw + (quality-gated length bonus − verbosity penalty)

### New scoring rubric includes:

#### ✔️ Quality-gated length bonus  
Only applies when:
- prompt_type ∈ {moderate, strict}
- answer length > threshold
- weighted by (IC + Relevance)

#### ✔️ Verbosity penalty  
Applied when:
- answer is long  
- relevance or coherence is low  

This discourages "fluff padding" in loose answers.

---

## 6.  Final Files Used in Analysis

| Purpose | File |
|--------|------|
| Baseline results | `narrative_scores_test500_newrubric_base3_newlen.csv` |
| RM rerank results | `narrative_scores_test500_strict_rm_rerank3_newlen.csv` |

---

## 7. DPO Data Format

Example DPO pair:

```json
  {
  "story_id": 12,
  "prompt": "...",
  "chosen": "ending with higher score...",
  "rejected": "ending with lower score...",
  "score_chosen": 4.0,
  "score_rejected": 3.2
  }
```

This is built by pairing:
 - one high-quality judge answer
 - with the local answer for the same prompt.



---

##  Summary

- Built a high-quality RM from LLM-as-Judge scored data.  
- Applied RM for reranking sampled strict-prompt generations.  
- Compared four model configurations (loose / moderate / strict / strict+RM).  
- Introduced a more robust evaluation rubric emphasizing relevance and coherence over verbosity.  
- Final report will contain DPO part!
- Final results used in the presentation deck.

---

# Milestone 4

This milestone evaluates how preference optimization (DPO) and reward-model reranking improve **strict** story continuation quality compared with earlier baselines.

## Experiment Overview

We compare six settings on the ROCStories test set:

1) **loose + base (single-sample)**  
   The base model answers the loose prompt once.

2) **moderate + base (single-sample)**  
   The base model answers the moderate prompt once.

3) **strict + base (single-sample)**  
   The base model answers the strict prompt once.

4) **strict + base + RM rerank (N=3)**  
   The base model samples 3 candidates using temperature/top_p.  
   A trained Reward Model scores the 3 and selects the best.

5) **strict + DPO (single-sample)**  
   The strict prompt is answered once by the DPO-finetuned model.

6) **strict + DPO + RM rerank (N=3)**  
   The DPO model samples 3 candidates; RM reranks and selects the best.

All generations use the same sampling parameters where applicable to ensure fair comparison.

## Evaluation

We use an LLM-as-Judge to score each answer on five dimensions:

- information_completeness  
- factual_accuracy  
- relevance  
- logical_coherence  
- creativity_expression  

We report:
- **overall_raw**: mean of the five dimensions  
- **overall_quality**: overall_raw adjusted by the updated length/verbosity rule (new rubric)

Aggregations are computed by `setting` over the strict subset and compared against earlier baseline runs.

## What’s New in Milestone 4

Milestone 4 adds two new strict-condition improvements:

### A) Strict + DPO (single-sample)
- Trains a DPO LoRA adapter using strict preference pairs:
  - `chosen` = high-scoring judge outputs
  - `rejected` = lower-quality (often local) outputs
- Goal: directly shift the base model toward strict-format, instruction-following continuations.

### B) Strict + DPO + RM rerank (N=3)
- Combines preference-optimized generation with RM selection.
- Goal: test whether RM provides additional gains after DPO,
  especially by filtering remaining verbose/meta or weak-structure samples.

## Expected Interpretation

- **Base strict vs. strict + RM** isolates the benefit of RM selection.  
- **Strict + DPO vs. strict base** isolates the benefit of preference optimization.  
- **Strict + DPO + RM vs. strict + DPO** tests whether RM still adds value after DPO.

In short, Milestone 4 focuses on whether **DPO** improves strict adherence and
whether **RM reranking** remains a useful second-stage filter on top of DPO. 

**More details see in PPT and report!**



---

#  Quick File Glossary

- train/strict_scored_train1000.jsonl  
  1000 prompts, each scored twice by LLM-as-Judge (~2000 records).

- train/strict_scored_train1000_local.jsonl  
  Local model answers for the same 1000 prompts.  
  Generally low quality; used as the “rejected” side for DPO.

- data/rocstories/strict_scored_train1000_all.jsonl  
  Merged judge A + judge B + local answers (~3000 records).

- train/strict_scored_train1000_reward.jsonl  
  Reward-model training format derived from `*_all.jsonl` (~3000 records).  
  Each line contains prompt + completion + score fields.

- train/strict_scored_train1000_dpo.jsonl  
  DPO training pairs derived from `*_all.jsonl` (~1000 pairs).  
  Each record contains `prompt`, `chosen`, `rejected`, and their scores.

- app/rm/train_rm.py  
  Trains the Reward Model (regression) with 4-bit + LoRA.

- app/rm/gen_and_rerank.py  
  Generates N=3 sampled candidates from the base model and reranks using the trained RM.

- app/dpo/train_dpo.py  
  Trains a DPO LoRA adapter from `strict_scored_train1000_dpo.jsonl` (supports 4-bit).

- app/roc_eval/judge_llm.py  
  LLM-as-Judge scoring tool.  
  Produces 5-dim scores + `overall_raw` (mean of 5 dims) + adjusted `overall_quality` (your length rule).

- data/rocstories/narrative_with_prompts_test500.csv  
  Test set with loose/moderate/strict prompt variants and metadata.  
  Used as the generation input.

- data/rocstories/narrative_scores_test500_newrubric_base3_newlen.csv  
  Scores for loose/moderate/strict base-model runs under the new rubric/length rule.

- data/rocstories/narrative_scores_test500_strict_rm_rerank3_newlen.csv  
  Scores for strict + RM rerank (N=3) under the new rubric/length rule.

- data/rocstories/narrative_scores_test500_strict_dpo_sample1.csv  
  Scores for strict + DPO single-sample generation.

- data/rocstories/narrative_scores_test500_strict_dpo_rm_rerank3.csv  
  Scores for strict + DPO + RM rerank (N=3).



---




# Deployment Notes

Because the base and fine-tuned checkpoints are large, they are **not** stored in the repo.  
The repo only provides:

- `Dockerfile`
- `docker-compose.yml`
- Training / evaluation scripts

You should host model weights separately (HF cache, local disk, or an internal model store) and mount them into the container.

## Recommended Layout

On the host machine:

- `/models/base/` – base 8B model (e.g. `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)
- `/models/rm/` – RM LoRA checkpoints (`rm_ex2_beginonly`, etc.)
- `/models/dpo/` – DPO LoRA checkpoints (`dpo_ex1_strict1000`, etc.)

Then in `docker-compose.yml`:

- Mount `/models` → `/workspace/models`
- Set env vars like:
  - `RM_DIR=/workspace/models/rm_ex2_beginonly`
  - `DPO_DIR=/workspace/models/dpo_ex1_strict1000`
  - `BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

The scripts already take `--base_model`, `--rm_dir`, `--dpo_dir` as CLI flags, so you can point them to these paths.

## GPU / VRAM Guidelines

All fine-tuning uses **4-bit + LoRA** to keep memory modest.

**Training (RM / DPO)**  
- 16 GB VRAM is comfortable for the current configs (batch size = 1, grad accum, 4-bit).

**Inference only**

- **Base 8B model, single strict/loose/moderate generation (4-bit)**  
  → _8–10 GB VRAM_ is usually enough.

- **Strict + RM rerank (N=3)**  
  - Load generator + RM (sequence classification head).  
  - 3 samples per prompt.  
  → Recommend _10–12 GB VRAM_. (More is safer if you plan to batch requests.)

- **Strict + DPO (single-sample)**  
  - Same as base, with a LoRA adapter on top.  
  → Also _8–10 GB VRAM_ is typically fine.

- **Strict + DPO + RM rerank (N=3)**  
  - Load DPO-tuned generator + RM, sample 3 candidates and rerank.  
  → Recommend at least _12 GB VRAM_ for comfort; _16 GB_ if you want room for larger context or batch serving.

## Suggested Deployment Flow

1. **Pull base model** on the host (e.g. `huggingface-cli download` or first run in a dev container).
2. **Train RM / DPO** once on a 16 GB GPU box, saving LoRA adapters under `/models/rm` and `/models/dpo`.
3. **Build the Docker image** from the repo (`docker build ...`).
4. **Use `docker-compose up`** with:
   - `./models:/workspace/models` volume
   - `NVIDIA_VISIBLE_DEVICES` / `--gpus all` configured
5. Expose a simple CLI or HTTP endpoint that:
   - Calls base / DPO model for generation.
   - Optionally calls RM for rerank (N=3) in the strict settings.

This way, the repo stays lightweight, and you can swap in newer checkpoints just by updating the mounted `/models` directory.


---

# Playground 

`app/playground/interactive_story.py` is a small interactive CLI for quick qualitative checks across the Milestone 4 setups (for all experiments setting).

## What it does
Loads the base 8B model once (4-bit optional), then optionally attaches:
- **DPO LoRA** on top of the same base instance
- **RM LoRA** on a separate sequence-classification head

You can repeatedly enter a prompt and choose one mode:
1) **base (single sample)**  
2) **base + RM rerank (N=3)**  
3) **DPO (single sample)**  
4) **DPO + RM rerank (N=3)**  

This design avoids loading multiple full generator instances and is meant for fast local sanity checks.

## Output
For rerank modes, the script prints:
- raw candidates (generation order)
- RM-ranked list (high → low)
- the selected best completion

## Demo video
A short demonstration of the interactive workflow is included at:
- `video/simple_show.mp4`  
  or  
- `submit/milestone4/video/simple_show.mp4`

## Usage
```bash
python /workspace/app/playground/interactive_story.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --rm_dir /workspace/models/rm_ex2_beginonly \
  --dpo_dir /workspace/models/dpo_ex1_strict1000 \
  --load_in_4bit \
  --num_candidates x \
  --max_new_tokens xxx \
  --min_new_tokens xxx \
  --temperature xxx \
  --top_p x \
  --top_k x

---