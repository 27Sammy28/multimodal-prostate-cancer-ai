import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pathology.metrics import binary_classification_metrics, curve_data


DEFAULT_TARGET_CANDIDATES = ["diagnosis_result", "clinically_significant_cancer", "target", "label"]


from models.risk import NumpyLogisticRiskModel, sigmoid

def make_synthetic_prostate_data(n_samples=500, random_state=42):
    rng = np.random.default_rng(random_state)
    age = rng.normal(66, 7, n_samples)
    psa = rng.lognormal(2.0, 0.6, n_samples)
    prostate_volume = rng.normal(45, 15, n_samples).clip(15, 120)
    pi_rads = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.12, 0.18, 0.25, 0.25, 0.20])
    adc_mean = rng.normal(900, 180, n_samples)
    lesion_size = rng.gamma(2.0, 4.0, n_samples)
    biomarker_score = rng.normal(0, 1, n_samples)
    logit = -7.5 + 0.04 * age + 0.12 * psa + 0.75 * pi_rads - 0.002 * adc_mean + 0.08 * lesion_size + 0.7 * biomarker_score
    probability = sigmoid(logit)
    labels = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "age": age,
            "psa": psa,
            "prostate_volume": prostate_volume,
            "pi_rads": pi_rads,
            "adc_mean": adc_mean,
            "lesion_size": lesion_size,
            "biomarker_score": biomarker_score,
            "family_history": np.where(rng.random(n_samples) > 0.75, "yes", "no"),
            "diagnosis_result": np.where(labels == 1, "M", "B"),
        }
    )


def find_target_column(df, requested_target=None):
    if requested_target:
        return requested_target
    for candidate in DEFAULT_TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not infer target column. Pass --target. Available columns: {list(df.columns)}")


def encode_binary_target(series, positive_label=None):
    if series.dtype.kind in "biufc":
        values = series.astype(int)
        unique = sorted(values.dropna().unique())
        if len(unique) != 2:
            raise ValueError("Numeric target must contain exactly two classes for binary risk modeling.")
        return (values == unique[-1]).astype(int), str(unique[-1])
    normalized = series.astype(str).str.strip()
    if positive_label is None:
        for candidate in ["M", "malignant", "cancer", "yes", "1", "true", "csPCa"]:
            matches = normalized.str.lower() == candidate.lower()
            if matches.any():
                positive_label = normalized[matches].iloc[0]
                break
    if positive_label is None:
        positive_label = sorted(normalized.unique())[-1]
    return (normalized == positive_label).astype(int), positive_label


def make_design_matrix(df, target_column):
    feature_df = df.drop(columns=[target_column])
    numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in feature_df.columns if col not in numeric_features]
    design = pd.get_dummies(feature_df, columns=categorical_features, dummy_na=True, drop_first=False)
    design = design.apply(pd.to_numeric, errors="coerce")
    design = design.fillna(design.median(numeric_only=True)).fillna(0)
    return design, numeric_features, categorical_features


def stratified_split_indices(y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []
    y = np.asarray(y)
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def save_curves(y_true, y_prob, output_dir):
    curves = curve_data(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(curves["roc"]["fpr"], curves["roc"]["tpr"], label="Risk model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "risk_model_roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(curves["pr"]["recall"], curves["pr"]["precision"], label="Risk model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "risk_model_pr_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(curves["calibration"]["mean_predicted"], curves["calibration"]["fraction_positive"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed event rate")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "risk_model_calibration_curve.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train a prostate cancer clinical/MRI risk-stratification model.")
    parser.add_argument("--csv", help="Input CSV with clinical, biomarker, MRI, or radiomics variables.")
    parser.add_argument("--target", help="Binary target column, e.g. diagnosis_result or clinically_significant_cancer.")
    parser.add_argument("--positive-label", help="Positive/cancer label. Defaults to M/malignant-like labels if found.")
    parser.add_argument("--model", choices=["logistic"], default="logistic")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for smoke testing.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for folder in ["checkpoints", "figures", "tables", "reports"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    if args.synthetic or not args.csv:
        df = make_synthetic_prostate_data(random_state=args.random_state)
        data_source = "synthetic"
    else:
        df = pd.read_csv(args.csv)
        data_source = args.csv

    target_column = find_target_column(df, args.target)
    y, positive_label = encode_binary_target(df[target_column], args.positive_label)
    design, numeric_features, categorical_features = make_design_matrix(df, target_column)
    train_idx, test_idx = stratified_split_indices(y, test_size=args.test_size, random_state=args.random_state)
    x_train = design.iloc[train_idx].to_numpy(dtype=float)
    x_test = design.iloc[test_idx].to_numpy(dtype=float)
    y_train = y.iloc[train_idx].to_numpy(dtype=int)
    y_test = y.iloc[test_idx].to_numpy(dtype=int)

    model = NumpyLogisticRiskModel().fit(x_train, y_train)
    y_prob = model.predict_proba(x_test)[:, 1]
    metrics = binary_classification_metrics(y_test, y_prob)

    metrics_row = {
        "model": args.model,
        "data_source": data_source,
        "target": target_column,
        "positive_label": positive_label,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "numeric_features": ";".join(numeric_features),
        "categorical_features": ";".join(categorical_features),
        **metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(output_dir / "tables" / "risk_model_metrics.csv", index=False)
    checkpoint = {"model": model, "feature_columns": design.columns.tolist(), "metrics": metrics_row}
    with (output_dir / "checkpoints" / "risk_model.pkl").open("wb") as handle:
        pickle.dump(checkpoint, handle)
    save_curves(y_test, y_prob, output_dir)

    report = {
        "purpose": "MRI-informed prostate cancer risk stratification for clinical decision support",
        "important_note": "This pipeline is inspired by published clinical decision-support workflows, but it is not an exact reproduction of any Karolinska Institutet model.",
        "modalities_supported": ["clinical", "biomarker", "MRI/radiomics CSV features"],
        "future_fusion_modalities": ["MRI CNN embeddings", "pathology embeddings", "biomarkers", "clinical variables"],
        "metrics": metrics_row,
    }
    (output_dir / "reports" / "risk_model_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(metrics_row, indent=2))


if __name__ == "__main__":
    main()