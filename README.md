# Prostate Cancer Pathology Project

This repository organizes the uploaded exploration notebook into a GitHub-ready research codebase.

## Repository Layout

```text
notebooks/              exploration and demonstrations
src/pathology/          reusable data, tiling, transforms, and stain utilities
models/                 CNN, Virchow, and CLAM model definitions
scripts/                reproducible experiment entry points
results/                generated outputs, checkpoints, tables, reports, figures
uploads/                original uploaded notebook
```

## Workflow Rule

- Use `notebooks/` for exploration, visual checks, and demonstrations.
- Move reusable functions into `src/pathology/`.
- Run reproducible experiments from `scripts/`.
- Save generated outputs into `results/` instead of committing them into notebooks.

## Expected Results

### EDA Figures

- `results/figures/brightness_vs_isup.png`
- `results/figures/contrast_distribution.png`
- `results/figures/image_quality_space.png`

### Pathology Visualizations

- `results/figures/wsi_examples.png`
- `results/figures/mask_examples.png`
- `results/figures/overlay_examples.png`

### Stain Normalization

- `results/figures/stain_normalization_example.png`

### UMAP Outputs

- `results/figures/umap_haralick.png`
- `results/figures/umap_virchow.png`
- `results/figures/umap_clam.png`

### CNN Outputs

- `results/checkpoints/efficientnet_best.pt`
- `results/tables/cnn_metrics.csv`
- `results/figures/confusion_matrix.png`
- `results/figures/roc_curve.png`
- `results/figures/training_curve.png`

## Notebook Plan

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

The original uploaded notebook remains at `uploads/prostate_cancer_MRI.ipynb`.
## MRI-Based Risk Stratification

Run a reproducible prostate cancer risk-stratification baseline from clinical, biomarker, MRI, or radiomics CSV features:

```bash
python scripts/train_risk_model.py --csv Prostate_Cancer.csv --target diagnosis_result --positive-label M --model logistic
```

For a smoke test without private data:

```bash
python scripts/train_risk_model.py --synthetic --model logistic
```

The script writes:

- `results/checkpoints/risk_model.pkl`
- `results/tables/risk_model_metrics.csv`
- `results/figures/risk_model_roc_curve.png`
- `results/figures/risk_model_pr_curve.png`
- `results/figures/risk_model_calibration_curve.png`
- `results/reports/risk_model_report.json`

Reported evaluation metrics include ROC-AUC, PR-AUC, F1, expected calibration error, and Brier score.

## Multimodal Fusion Roadmap

`models/fusion_transformer.py` contains a transformer-based fusion module for combining MRI, pathology, biomarker, and clinical embeddings. This is an extension point for diagnosis, prognosis, and treatment-planning experiments once aligned multimodal patient-level features are available.
## Tile-Level CNN, Stain, Feature, and Embedding Pipeline

Run the complete tile analysis smoke test:

```bash
python scripts/run_tile_pipeline.py --synthetic --output-dir results
```

This combines:

- Macenko stain normalization
- Haralick texture features
- RGB color histograms
- deterministic deep tile embeddings
- UMAP when available, otherwise PCA fallback

It writes:

- `results/figures/stain_normalization_macenko.png`
- `results/figures/umap_tile_embeddings.png`
- `results/tables/haralick_color_features.csv`
- `results/embeddings/deep_tile_embeddings.csv`
- `results/embeddings/tile_embedding_2d.csv`
- `results/reports/combined_pipeline_summary.csv`

Generate CNN-style result artifacts:

```bash
python scripts/generate_cnn_results.py --synthetic --output-dir results
```

It writes:

- `results/checkpoints/efficientnet_best.pkl`
- `results/tables/cnn_metrics.csv`
- `results/figures/confusion_matrix.png`
- `results/figures/roc_curve.png`
- `results/figures/pr_curve.png`
- `results/figures/training_curve.png`
- `results/embeddings/cnn_deep_features.csv`
- `results/reports/cnn_results_report.json`
