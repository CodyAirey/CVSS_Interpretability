# CVSS_Interpretability

This repository contains the full codebase, data processing pipeline, trained model configurations, interpretability tooling, and survey materials used in:

> **When the Model Is Right for the Wrong Reason: Interpretability and Specification Failures in CVSS Prediction**  
> and the associated MSc thesis on CVSS metric prediction and explainability.

This repository is organised to allow:

- Reproduction of CVSS metric prediction experiments  
- Reproduction of interpretability analyses (IG, SHAP, LIME, LRP)  
- Inspection of cross-model and cross-method agreement  
- Full transparency of the practitioner user study  

---

# Repository Overview

## 1. Prediction Experiments (Thesis Chapter: Prediction)

All training and evaluation code for transformer-based CVSS metric prediction is located in:

```
training_scripts/
```

Supporting utilities:

```
utils/
```

Datasets:

```
cve_data/
```

This includes:

- Data loading and preprocessing
- Model configuration
- Seed control and reproducibility logic
- Evaluation metrics
- Cross-seed aggregation
- CSV exports for further analysis

To reproduce prediction experiments, begin with the scripts in `training_scripts/`.

---

## 2. Interpretability Experiments (Thesis Chapter: Interpretability)

All attribution and explanation tooling is located in:

```
interpretability/
```

This includes implementations and orchestration for:

- Integrated Gradients
- SHAP
- LIME
- Layer-wise Relevance Propagation (LRP)
- Cross-model comparison scripts
- Token-level aggregation
- Agreement metrics
- SVG visualisation rendering

Key orchestration and analysis scripts are found under:

```
interpretability/
utils/
```

These components allow reproduction of:

- Per-CVE attribution visualisations
- Cross-method disagreement analysis
- Global token aggregation experiments
- Jaccard overlap analysis
- Cross-model anecdotal analysis

If you are reviewing the interpretability paper, this directory contains all attribution logic and analysis tooling.

---

# User Study Materials (For Reviewers)

If you are reviewing the interpretability paper and looking for the **practitioner validation study**, all materials are here:

```
survey/
```

This directory contains:

- The full participant-facing survey instrument (`README.md`)
- Question wording
- Scale definitions
- Sampling design
- Justification Value scale (8-point)
- Follow-up Likert items
- Final reflection items

The survey instrument reproduced in the paper is fully documented in:

```
survey/README.md
```

This includes:

- Consent flow
- Demographic questions
- CVE justification task layout
- Full wording of all response options
- Description of the 60-CVE candidate pool
- Description of the 10-CVE per-participant sampling design

The released CSV accompanying the thesis artefacts contains:

- Per-participant responses
- Per-CVE assignment
- Likert follow-up ratings
- Free-text responses

If you are specifically assessing the claim that practitioner judgements align with model disagreement patterns, the survey materials above document the exact instrument used.

---

# Thesis Structure Mapping

For clarity, the repository corresponds to the thesis structure as follows:

### Prediction (Reproducibility and Model Behaviour)

- `training_scripts/`
- `cve_data/`
- `utils/`

### Interpretability (Attribution and Agreement)

- `interpretability/`
- Supporting utilities in `utils/`

### Human Validation (Practitioner Study)

- `survey/`
- Associated CSV artefacts (linked in thesis appendix)

---

# Reproducibility Notes

- All experiments were run with fixed seeds.
- Model configurations are explicitly defined in scripts.
- Aggregation scripts compute mean and standard deviation across seeds.
- Output CSVs are designed to allow independent re-analysis.
- No closed-source tooling is required.

Large model checkpoints are archived separately due to size constraints and are linked in the thesis appendix.

---

# What This Repository Enables

This repository allows independent verification of:

- CVSS metric prediction performance
- Cross-seed variability
- Attribution method divergence
- Cross-model agreement patterns
- Token-level explanation fragmentation
- Human-practitioner justification behaviour
- Alignment between human judgement and model disagreement

---

# Contact

For questions regarding the thesis, interpretability experiments, or practitioner study, please open an issue or contact the author directly.

