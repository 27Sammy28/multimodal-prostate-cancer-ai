# Prostate Cancer Multimodal AI Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Research](https://img.shields.io/badge/Project-Prostate%20Cancer%20AI-purple)
![Status](https://img.shields.io/badge/Status-Reproducible%20Smoke%20Tests-green)
![Modalities](https://img.shields.io/badge/Modalities-Pathology%20%7C%20MRI%20%7C%20Clinical%20%7C%20Survival-orange)

A GitHub-ready research codebase for prostate cancer risk stratification, starting from the uploaded exploratory notebook and reorganized into reusable modules, reproducible scripts, generated results, and a manuscript-style article.

The project supports pathology preprocessing, Macenko stain normalization, Haralick and color-histogram feature engineering, deep tile embeddings, CNN-style classification outputs, MRI feature integration, CLAM-style multiple-instance learning, foundation-model interfaces, explainable AI, multimodal transformer fusion, and survival analysis.

> **Important:** Current metrics are smoke-test results on synthetic/demo data. They validate that the software pipeline runs end-to-end; they are not clinical performance claims.

---

## Highlights

- **Reproducible structure:** notebooks for exploration, `src/` for reusable code, `scripts/` for experiments, `results/` for outputs.
- **Pathology pipeline:** tiling utilities, Macenko normalization, Haralick texture features, RGB color histograms, and tile embeddings.
- **CNN-style outputs:** metrics table, confusion matrix, ROC curve, PR curve, and training curve.
- **MRI integration:** PI-RADS, ADC, lesion-size, lesion-volume, T2, and DWI feature support.
- **Advanced modules:** CLAM, foundation-model embeddings, explainable AI, transformer fusion, and survival analysis.
- **Article included:** the LaTeX article is in `main.tex`, with compiled output in `main.pdf`.

---

## Repository Layout

```text
notebooks/              exploration and demonstrations
src/pathology/          reusable preprocessing, features, MRI, XAI, survival, metrics
models/                 CNN, risk model, CLAM, Virchow, fusion transformer definitions
scripts/                reproducible experiment entry points
results/                generated outputs: figures, tables, reports, embeddings, checkpoints
docs/                   advanced module documentation
uploads/                original uploaded notebook
main.tex                manuscript-style article
main.pdf                compiled article with results and figures
```

---

## Workflow Rule

| Area | Role |
|---|---|
| `notebooks/` | Exploration, visual checks, demonstrations |
| `src/pathology/` | Reusable library code |
| `models/` | Model definitions and reusable architectures |
| `scripts/` | Reproducible experiment entry points |
| `results/` | Generated outputs only |

The guiding rule is:

```text
notebooks = exploration + demonstrations
src/      = reusable code
scripts/  = reproducible experiments
results/  = generated outputs
```

---

## Result Gallery

### Macenko Stain Normalization

![Macenko stain normalization](results/figures/stain_normalization_macenko.png)

**Output:** `results/figures/stain_normalization_macenko.png`

---

### Tile Embedding Space

![Tile embedding visualization](results/figures/umap_tile_embeddings.png)

**Output:** `results/figures/umap_tile_embeddings.png`  
**Note:** UMAP is used when available; otherwise the pipeline falls back to PCA.

---

### CNN-Style Classification Outputs

| Confusion Matrix | ROC Curve | Precision-Recall Curve |
|---|---|---|
| ![Confusion matrix](results/figures/confusion_matrix.png) | ![ROC curve](results/figures/roc_curve.png) | ![PR curve](results/figures/pr_curve.png) |

![Training curve](results/figures/training_curve.png)

**Outputs:**

- `results/figures/confusion_matrix.png`
- `results/figures/roc_curve.png`
- `results/figures/pr_curve.png`
- `results/figures/training_curve.png`

---

### Explainable AI

| Feature Importance | Saliency Overlay |
|---|---|
| ![XAI feature importance](results/figures/xai_feature_importance.png) | ![XAI saliency overlay](results/figures/xai_saliency_overlay.png) |

**Outputs:**

- `results/tables/xai_permutation_importance.csv`
- `results/figures/xai_feature_importance.png`
- `results/figures/xai_saliency_overlay.png`

---

### Survival Analysis

![Kaplan-Meier curve](results/figures/kaplan_meier_curve.png)

**Outputs:**

- `results/tables/synthetic_survival_data.csv`
- `results/tables/kaplan_meier_curve.csv`
- `results/figures/kaplan_meier_curve.png`

---

## Current Smoke-Test Results

### CNN Deep-Feature Baseline

| Model | ROC-AUC | PR-AUC | F1 | ECE | Brier Score |
|---|---:|---:|---:|---:|---:|
| CNN deep-feature baseline | 1.000 | 1.000 | 1.000 | 0.0017 | 0.000011 |

Source: `results/tables/cnn_metrics.csv`

### Combined Tile Pipeline

| Tiles | Handcrafted Features | Deep Features | Dimensionality Reduction |
|---:|---:|---:|---|
| 32 | 68 | 128 | PCA fallback |

Source: `results/reports/combined_pipeline_summary.csv`

### Explainable AI Top Features

| Feature | Baseline ROC-AUC | Importance |
|---|---:|---:|
| `clinical_mri_feature_2` | 0.987 | 0.328 |
| `clinical_mri_feature_0` | 0.987 | 0.156 |
| `clinical_mri_feature_5` | 0.987 | 0.043 |

Source: `results/tables/xai_permutation_importance.csv`

### Advanced Modules Summary

| Module | Output |
|---|---|
| MRI integration | 80 synthetic patients, 7 MRI feature families |
| Foundation embeddings | 16 tiles, 64-dimensional Virchow-style embeddings |
| Explainable AI | top feature: `clinical_mri_feature_2` |
| Survival analysis | 120 synthetic patients, C-index ≈ 0.621 |

Source: `results/reports/advanced_modules_report.json`

---

## Quick Start

Run all reproducible smoke workflows from the repository root.

### 1. Tile-Level Stain, Feature, and Embedding Pipeline

```bash
python scripts/run_tile_pipeline.py --synthetic --output-dir results
```

This generates:

- `results/figures/stain_normalization_macenko.png`
- `results/figures/umap_tile_embeddings.png`
- `results/tables/haralick_color_features.csv`
- `results/embeddings/deep_tile_embeddings.csv`
- `results/embeddings/tile_embedding_2d.csv`
- `results/reports/combined_pipeline_summary.csv`

### 2. CNN-Style Result Artifacts

```bash
python scripts/generate_cnn_results.py --synthetic --output-dir results
```

This generates:

- `results/checkpoints/efficientnet_best.pkl`
- `results/tables/cnn_metrics.csv`
- `results/figures/confusion_matrix.png`
- `results/figures/roc_curve.png`
- `results/figures/pr_curve.png`
- `results/figures/training_curve.png`
- `results/embeddings/cnn_deep_features.csv`
- `results/reports/cnn_results_report.json`

### 3. MRI-Informed Risk Stratification

Use a real CSV when available:

```bash
python scripts/train_risk_model.py \
  --csv Prostate_Cancer.csv \
  --target diagnosis_result \
  --positive-label M \
  --model logistic \
  --output-dir results
```

Smoke test without private data:

```bash
python scripts/train_risk_model.py --synthetic --model logistic --output-dir results
```

Reported metrics include ROC-AUC, PR-AUC, F1, expected calibration error, and Brier score.

### 4. Foundation Models, MRI, XAI, and Survival

```bash
python scripts/run_foundation_mri_xai_survival.py --output-dir results
```

This generates MRI feature tables, Virchow-style embeddings, XAI plots, Kaplan-Meier curves, and `results/reports/advanced_modules_report.json`.

### 5. Compile the Article

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

The compiled article is written to `main.pdf`.

---

## Notebook Plan

The original uploaded notebook remains at `uploads/prostate_cancer_MRI.ipynb`. The planned notebook sequence is:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_wsi_visualization.ipynb`
3. `notebooks/03_mask_analysis.ipynb`
4. `notebooks/04_stain_normalization.ipynb`
5. `notebooks/05_feature_engineering.ipynb`
6. `notebooks/06_umap_analysis.ipynb`
7. `notebooks/07_tile_dataset.ipynb`
8. `notebooks/08_efficientnet_baseline.ipynb`
9. `notebooks/09_virchow_embeddings.ipynb`
10. `notebooks/10_clam_training.ipynb`

---

## Key Modules

| Module | Purpose |
|---|---|
| `src/pathology/stain_normalization.py` | Reinhard and Macenko stain normalization |
| `src/pathology/features.py` | Haralick texture and RGB histogram features |
| `src/pathology/deep_features.py` | deterministic deep tile embeddings for smoke tests |
| `src/pathology/reduction.py` | UMAP with PCA fallback |
| `src/pathology/mri.py` | MRI/radiomics feature validation and aggregation |
| `src/pathology/explainability.py` | permutation importance and saliency overlays |
| `src/pathology/survival.py` | Kaplan-Meier, Cox-style risk, C-index |
| `src/pathology/foundation_models.py` | foundation-model embedding interface |
| `models/clam.py` | attention-based multiple-instance learning |
| `models/fusion_transformer.py` | multimodal transformer fusion |

---

## Research Roadmap

- Replace synthetic smoke-test data with curated patient cohorts.
- Add real whole-slide image tiling and slide-level CLAM attention heatmaps.
- Connect true pathology foundation models such as Virchow, UNI, or CONCH.
- Integrate MRI DICOM-derived features, radiomics, and structured report variables.
- Evaluate multimodal transformer fusion against unimodal baselines.
- Extend survival modeling to time-dependent AUC, competing risks, and external validation.
- Add calibration plots and decision-curve analysis for clinical utility assessment.

---

## Clinical and Scientific Caution

This repository is a research scaffold for method development and reproducibility. It is not a medical device, diagnostic tool, or clinically validated decision-support system. Real-world use would require institutional review, data governance, independent validation, calibration checks, bias analysis, and prospective clinical evaluation.
