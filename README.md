# Reasoning Project

---

## Submissions

All submission materials will be updated in the repository at:

**`submit/milestoneX/`**

Please refer to this directory for the latest submission files and documentation.

---

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
---