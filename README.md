# Diagnosing Evaluation Instability via Deep Linguistic Fingerprinting

**Beyond "My Answer is C": Unpacking the Linguistic Mechanics of LLM Evaluation Instability**

## Overview

Standard LLM evaluation often forces models into a rigid format — picking a multiple-choice letter or reading off the highest first-token probability. Prior work shows this constrained setup can disagree with what the model actually says when allowed to answer in open-ended text, with mismatch rates as high as 60%. This project asks *why*: what changes in the model's language itself when the constraint is lifted?

We combine [elfen](https://github.com/mmilbig/elfen) (1,061 linguistic features across 11 areas), opinion/knowledge benchmarks (MMLU, OpinionQA, TruthfulQA), and stance-shift analysis to build **linguistic fingerprints** of LLM outputs under varying prompt constraints — and test whether those fingerprints can predict when a model is about to flip its answer or refuse.

## Research Questions

1. How do syntactic complexity, hedging, readability, and sentiment in open-ended LLM generations correlate with decision shifts (e.g., agreeing vs. disagreeing, answering vs. refusing)?
2. Can a classifier trained on elfen linguistic features predict when a model is about to experience a **first-token vs. text mismatch**?

## Method

**Models:** Llama-3-8B-Instruct, Mistral-7B-Instruct, Qwen2.5-1.5B-Instruct (run locally via vLLM)

**Datasets:** MMLU, OpinionQA, TruthfulQA — stratified to a balanced N=900 (300 items per dataset, seed=42) to prevent MMLU's larger size from dominating aggregate metrics.

**Pipeline:**
1. **First-token probability** — prompt with the MCQ and inspect log-probabilities of the very first generated token (A/B/C/D/Refusal).
2. **Unconstrained generation** — same prompt, open-ended ("take a clear stance"), higher `max_tokens` to allow full reasoning.
3. **Classification** — map unconstrained text back to an MCQ option (LLM-judge or regex) and flag mismatches against the first-token answer.
4. **Linguistic feature extraction** — run all open-ended outputs through elfen (readability, dependency tree depth, sentiment, hedge frequency, entropy, etc.).
5. **Statistical analysis** — Mantel tests to compare correlation structures across constraint levels; Local Outlier Factor (LOF) to detect whether specific fingerprints (e.g., high hedge ratio) align with mismatches/stance-flips.

**Metrics:** Mismatch Rate, Mantel Correlation Coefficient, Hedge Ratio, Logistic Regression Feature Importance.

## Repository Structure

```
.
├── FinalScript.ipynb              # End-to-end inference + mismatch pipeline
├── NLP-Elfen-Visualization.ipynb  # elfen feature extraction & visualization
├── results/                       # Output data, figures, and analysis artifacts
└── README.md
```

## Poster Visuals

- **Heatmap comparison** — elfen correlation matrices for Forced-Choice vs. Open-Ended outputs across 11 feature areas.
- **t-SNE projection** — clustering of refusals and stance-flips in linguistic feature space.
- **Feature importance bar chart** — which linguistic signals (e.g., hedge frequency, dependency depth) best predict a first-token/text mismatch.

## Key Takeaway

Mismatches between forced and unconstrained model answers aren't random — they leave a measurable linguistic signature. Identifying that signature is a step toward diagnosing *why* evaluation results shift with prompt format, not just *that* they do.