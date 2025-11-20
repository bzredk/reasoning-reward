# scoring.md

## 1. Task and Evaluation Setting
We evaluate narrative story continuation on a ROCStories-style dataset.  
Each example includes:
- `beginning` — starting sentences of the story
- `reference` — human gold ending
- `model_answer` — system-generated continuation
- `prompt_type` — `loose`, `moderate`, or `strict`

Goal: assess continuation quality with a unified rubric.

---

## 2. Evaluation: LLM-as-a-Judge

We use a rubric-based LLM-as-a-judge to evaluate all narrative continuations.  
The evaluation logic is implemented in:

**app/roc_eval/judge_llm.py**

For each example, the judge model (`gpt-4.1-mini`) receives:
- the story beginning  
- the human reference ending (as guidance only)  
- the model-generated continuation  
- the prompt type (`loose`, `moderate`, `strict`)

The judge returns a strict JSON object with six integer scores (0–5):
- `information_completeness`  
- `factual_accuracy`  
- `relevance`  
- `logical_coherence`  
- `creativity_expression`  
- `overall_quality`

The reference ending is *not* treated as the only correct answer; the judge evaluates the continuation’s plausibility and coherence relative to the beginning.

A detailed description of each scoring dimension and its 0–5 criteria is provided in **data.md** (submitted in Milestone 1).  
The scoring script directly applies those criteria through the system prompt embedded in `judge_llm.py`.

---

## 3. Final Per-Example Score
The script outputs all rubric scores plus:
- `overall_raw`
- `answer_length_words`
- `overall_quality` (final score used)

A small, controlled smoothing may apply to `overall_quality` for structured prompts (`moderate`, `strict`) to avoid extremely short template outputs. This adjustment is capped and does not dominate evaluation. `loose` scores are unmodified.

Full results for Milestone 2 use:
`data/rocstories/narrative_scores_test500.csv`.

---

## 4. System-Level Metrics

Let N be the number of evaluation examples.

For each example i, let:
- o_i = final overall quality score
- f_i = factual accuracy score
- r_i = relevance score

The primary system-level metric is the average overall score:

OverallScore(S) = (1 / N) * sum over i of o_i

Auxiliary metrics:

FactualScore(S)   = (1 / N) * sum over i of f_i  
RelevanceScore(S) = (1 / N) * sum over i of r_i

`OverallScore` is the primary metric.

---

## 5. Scripts and Workflow
### Prompt Generation
**app/roc_eval/build_prompts.py**  
Outputs:
`id, prompt_type, prompt_text, beginning, reference, background, model_answer, ...`

### Evaluation Script (score.py)
**app/roc_eval/judge_llm.py**

#### Input CSV Requirements
`id, prompt_type, beginning, reference, model_answer`

#### Output CSV
All input fields plus rubric scores and derived metadata.

---

## 6. Running Evaluation

```bash
python -m app.roc_eval.judge_llm   --input data/rocstories/narrative_with_prompts_test500.csv   --output data/rocstories/narrative_scores_test500.csv   --openai_model gpt-4.1-mini
```

Final system metrics are computed as column means from  
`data/rocstories/narrative_scores_test500.csv`.

---

## 7. End-to-End Example (from prompt construction to judged scores)

This section illustrates the end-to-end flow from prompt construction, local generation, and LLM-as-a-judge scoring, through to the final CSV output.

### 7.0 Prompt construction (`build_prompts.py`)

We first construct three instruction prompts per story using:

```bash
python -m app.roc_eval.build_prompts   --input data/rocstories/narrative_raw_test500.csv   --output data/rocstories/narrative_with_prompts_test500.csv
```

For each input row with `background` (full story) and `reference` (gold ending), `build_prompts.py`:

1. Derives `beginning` by removing the reference from the full story or taking the first 1–2 sentences.
2. Calls an OpenAI model (`gpt-4.1` by default) with a fixed system instruction (`PROMPT_SYSTEM_TEXT`) to generate a JSON object:

   ```json
   {
     "prompt_loose":   "...",
     "prompt_moderate":"...",
     "prompt_strict":  "..."
   }
   ```

   Each prompt:
   - is in English,
   - explicitly includes `Beginning: <beginning>`,
   - asks the model to write a continuation (not to select an option),
   - differs only in the level of structural constraints (loose / moderate / strict).

3. For each `prompt_type ∈ {loose, moderate, strict}`, it:
   - stores the corresponding `prompt_text`,
   - calls the local base model (DeepSeek-R1-Distill-Llama-8B) in 4‑bit to generate `model_answer`,
   - writes one expanded row to the output CSV.

### 7.1 Row from `narrative_with_prompts_test500.csv`

After running `build_prompts.py`, a typical row in `data/rocstories/narrative_with_prompts_test500.csv` has the form:

```csv
id,prompt_type,beginning,reference,background,prompt_text,model_answer,...
42,strict,"The baby shook from the cold room. His mother turned up the heat.",
"The nurses kept turning it back down. She was angry that they were making him uncomfortable. She told the doctor she wanted to go home.",
"...full background story...",
"Provide a chronologically ordered continuation that follows the template exactly...
Beginning:
The baby shook from the cold room. His mother turned up the heat.

Your output:",
"...model continuation text...",...
```

- `prompt_text` is the strict instruction generated by `build_prompts.py`.
- `model_answer` is the continuation generated locally from that prompt.

### 7.2 Judge call (`judge_llm.py`)

Running:

```bash
python -m app.roc_eval.judge_llm   --input data/rocstories/narrative_with_prompts_test500.csv   --output data/rocstories/narrative_scores_test500.csv   --openai_model gpt-4.1-mini
```

for each row constructs the judge input:

```text
Prompt type: strict

Beginning:
The baby shook from the cold room. His mother turned up the heat.

Reference continuation:
The nurses kept turning it back down. She was angry that they were making him uncomfortable. She told the doctor she wanted to go home.

Model continuation:
...model_answer...
```

The judge model, using `JUDGE_SYSTEM_PROMPT`, returns a JSON object:

```json
{
  "information_completeness": 3,
  "factual_accuracy": 4,
  "relevance": 5,
  "logical_coherence": 3,
  "creativity_expression": 4,
  "overall_quality": 4,
  "comments": "Coherent and relevant; could include more detail."
}
```

### 7.3 Row in `narrative_scores_test500.csv`

`judge_llm.py` then:

1. Computes `answer_length_words` from `model_answer`.
2. Applies the length bonus (for `prompt_type` = `moderate`/`strict`).
3. Sets `overall_quality` to the length-adjusted score and stores the raw score as `overall_raw`.
4. Copies all original columns from the input CSV.

The corresponding row in `data/rocstories/narrative_scores_test500.csv` looks like:

```csv
id,prompt_type,information_completeness,factual_accuracy,relevance,logical_coherence,
creativity_expression,overall_quality,overall_raw,answer_length_words,judge_comments,
prompt_text,beginning,reference,background,model_answer,...
42,strict,3,4,5,3,4,4.25,4,112,"Coherent and relevant; could include more detail.",
"Provide a chronologically ordered continuation that follows the template exactly...",
"The baby shook from the cold room. His mother turned up the heat.",
"The nurses kept turning it back down. ...",
"...full background story...",
"...model continuation text...",...
```

System-level metrics (e.g., `OverallScore`, `FactualScore`, `RelevanceScore`) are computed as column-wise means over all rows in `narrative_scores_test500.csv`.

---