from pathlib import Path

import numpy as np
import pandas as pd


MRI_FEATURE_COLUMNS = [
    "pi_rads",
    "adc_mean",
    "adc_min",
    "lesion_size_mm",
    "lesion_volume_mm3",
    "t2_intensity_mean",
    "dwi_intensity_mean",
]


def load_mri_features(csv_path):
    return pd.read_csv(csv_path)


def validate_mri_feature_table(df, patient_id_col="patient_id"):
    missing = [col for col in [patient_id_col] if col not in df.columns]
    available_features = [col for col in MRI_FEATURE_COLUMNS if col in df.columns]
    if missing:
        raise ValueError(f"Missing required MRI identifier columns: {missing}")
    if not available_features:
        raise ValueError(f"No recognized MRI feature columns found. Expected any of: {MRI_FEATURE_COLUMNS}")
    return available_features


def aggregate_lesion_features(df, patient_id_col="patient_id"):
    feature_cols = validate_mri_feature_table(df, patient_id_col=patient_id_col)
    aggregations = {col: ["mean", "max", "min"] for col in feature_cols}
    aggregated = df.groupby(patient_id_col).agg(aggregations)
    aggregated.columns = [f"mri_{feature}_{stat}" for feature, stat in aggregated.columns]
    return aggregated.reset_index()


def make_synthetic_mri_features(n_patients=80, random_state=42):
    rng = np.random.default_rng(random_state)
    return pd.DataFrame(
        {
            "patient_id": [f"P{idx:04d}" for idx in range(n_patients)],
            "pi_rads": rng.choice([1, 2, 3, 4, 5], n_patients, p=[0.12, 0.18, 0.25, 0.25, 0.20]),
            "adc_mean": rng.normal(900, 170, n_patients),
            "adc_min": rng.normal(650, 130, n_patients),
            "lesion_size_mm": rng.gamma(2.2, 4.5, n_patients),
            "lesion_volume_mm3": rng.gamma(2.4, 900, n_patients),
            "t2_intensity_mean": rng.normal(0.55, 0.12, n_patients),
            "dwi_intensity_mean": rng.normal(0.62, 0.15, n_patients),
        }
    )


def save_mri_feature_dictionary(output_path):
    dictionary = pd.DataFrame(
        [
            {"feature": "pi_rads", "description": "PI-RADS assessment score, ordinal 1-5"},
            {"feature": "adc_mean", "description": "Mean ADC value in lesion/ROI"},
            {"feature": "adc_min", "description": "Minimum ADC value in lesion/ROI"},
            {"feature": "lesion_size_mm", "description": "Maximum lesion diameter in millimeters"},
            {"feature": "lesion_volume_mm3", "description": "Estimated lesion volume in cubic millimeters"},
            {"feature": "t2_intensity_mean", "description": "Mean normalized T2 signal intensity"},
            {"feature": "dwi_intensity_mean", "description": "Mean normalized DWI signal intensity"},
        ]
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(output_path, index=False)
    return dictionary