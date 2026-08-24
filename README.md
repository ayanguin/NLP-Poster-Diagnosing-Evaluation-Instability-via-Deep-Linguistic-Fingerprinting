# Diagnosing Evaluation Instability via Deep Linguistic Fingerprinting

**NLP Poster Project — University of Trier**

## Overview

LLM evaluation often forces a model into a rigid format — picking a multiple-choice letter, or reading off the highest first-token probability. Prior work shows this constrained setup can disagree sharply with what the model says when allowed to answer in open-ended text, with mismatch rates reported as high as 60%. This project asks whether that disagreement leaves a measurable **linguistic signature**: does the way a model *writes* its open-ended answer carry information about whether it's about to contradict its own first-token choice?

We combine [elfen](https://github.com/mmilbig/elfen) (a large linguistic feature-extraction library) with three evaluation benchmarks (MMLU, OpinionQA, TruthfulQA) across five instruction-tuned models to test this directly.

## Research Questions

1. **How do syntactic complexity, hedging, readability, and sentiment metrics of open-ended LLM generations correlate with decision shifts** (e.g., agreeing vs. disagreeing, or choosing an option vs. refusing)?
2. **Can a classifier trained on elfen features predict when a model is about to experience a first-token vs. text mismatch?**

## Method

**Models:** Qwen (`QWEN`), SmolLM (`HuggingFaceTB`), Mistral (`mistralai`), Gemma (`google`), Llama (`meta-llama`) — run locally via vLLM.

**Datasets:** MMLU, OpinionQA, TruthfulQA, stratified to a balanced N=900 (300 items per dataset, seed=42) so no single benchmark dominates the aggregate metrics.

**Pipeline:**
1. **First-token probability** — prompt each model with the MCQ and record the highest-probability first token (A/B/C/D/Refusal).
2. **Unconstrained generation** — same question, open-ended instruction ("take a clear stance"), higher `max_tokens` to capture full reasoning.
3. **Answer parsing** — map the unconstrained text back to an MCQ option (or flag as refusal), and label each row `is_mismatch` against the first-token answer.
4. **Linguistic feature extraction** — run every open-ended response through elfen to extract linguistic features (readability, syntactic dependency structure, sentiment, hedging, entropy, psycholinguistic norms, etc.).
5. **RQ1 — correlation analysis** — correlate each linguistic feature with `is_mismatch` to identify which features distinguish matched from mismatched responses.
6. **RQ2 — classification** — train an L1-regularized logistic regression on the elfen feature set to predict `is_mismatch`, evaluated with 5-fold stratified cross-validation (ROC-AUC) and validated against a permutation-test null distribution (200 permutations) to confirm the signal is statistically above chance.

## Repository Structure

```
.
├── FinalScript.ipynb              # Inference pipeline: first-token + open-ended generation, mismatch labeling
├── NLP-Elfen-Visualization.ipynb  # Feature extraction, RQ1 correlation analysis, RQ2 classifier + permutation test
├── results/                       # Output CSVs (elfen features per model), figures, and analysis artifacts
└── README.md
```

## How to Reproduce

1. Install dependencies: `pip install polars elfen spacy scikit-learn matplotlib seaborn` and `python -m spacy download en_core_web_sm`.
2. Run `FinalScript.ipynb` to generate the per-model `*_dataset_experiment_results.csv` files (first-token vs. unconstrained answers, mismatch labels).
3. Run `NLP-Elfen-Visualization.ipynb` top to bottom:
   - Feature extraction produces `results/{model}_sample_features.csv` per model.
   - RQ1 analysis produces correlation rankings and feature-importance plots.
   - RQ2 analysis (`mismatch_classifier`) produces per-model AUC, permutation p-value, and top classifier coefficients, saved to `results/`.

## Results Summary

| Model | Mismatch Rate | Classifier AUC (5-fold CV) | Permutation p-value |
|---|---|---|---|
| QWEN | 46.3% | 0.572 | 0.010 |
| HuggingFaceTB | 49.3% | 0.630 | 0.005 |
| mistralai | 29.3% | 0.660 | 0.005 |
| google | 25.1% | 0.608 | 0.005 |
| meta-llama | 33.9% | 0.661 | 0.005 |

**RQ1:** Several linguistic features (readability indices, word length, lexical density, age-of-acquisition, sentiment, entity/dependency counts) show statistically detectable but modest correlations with mismatch (|r| ≈ 0.10–0.21). The specific features that matter most differ across models, suggesting no single universal linguistic fingerprint.

**RQ2:** An L1-regularized logistic regression trained on elfen features predicts mismatch significantly above chance for all five models (permutation test, all p < 0.01). Discriminative power is modest in absolute terms (AUC 0.57–0.66), indicating a real but partial linguistic signal — most of the variance in mismatch is not explained by surface linguistic form alone.

## Limitations

- Model families differ in architecture, alignment training, and parameter scale simultaneously; AUC differences across models cannot be attributed to any single factor (e.g., scale) without a controlled same-family, multi-scale comparison.
- Supplementary analyses (Local Outlier Factor outlier detection, Mantel correlation-structure comparison) did not show meaningful alignment with mismatch and are not treated as supporting evidence for either RQ.
- Correlation and classifier results reflect this specific balanced sample (N=900 per model) and prompt design; results may not generalize to other datasets or prompting strategies.

## Poster

This repository accompanies a poster submitted for the NLP module examination. Poster title: *Beyond "My Answer is C": Unpacking the Linguistic Mechanics of LLM Evaluation Instability.*