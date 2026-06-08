from pathlib import Path

import pandas as pd
import numpy as np


def load_labels(data_dir="data", csv_name="train.csv"):
    return pd.read_csv(Path(data_dir) / csv_name)


def load_clinical_csv(csv_path):
    return pd.read_csv(csv_path)


def split_dataframe(df, target, test_size=0.2, random_state=42, stratify=True):
    rng = np.random.default_rng(random_state)
    if stratify and target in df:
        train_indices = []
        test_indices = []
        for _, group in df.groupby(target):
            indices = group.index.to_numpy()
            rng.shuffle(indices)
            n_test = max(1, int(round(len(indices) * test_size)))
            test_indices.extend(indices[:n_test])
            train_indices.extend(indices[n_test:])
    else:
        indices = df.index.to_numpy()
        rng.shuffle(indices)
        n_test = int(round(len(indices) * test_size))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    return df.loc[train_indices].reset_index(drop=True), df.loc[test_indices].reset_index(drop=True)


def slide_path(data_dir, image_id, folder="train_images", suffix=".tiff"):
    return Path(data_dir) / folder / f"{image_id}{suffix}"


def mask_path(data_dir, image_id, folder="train_label_masks", suffix="_mask.tiff"):
    return Path(data_dir) / folder / f"{image_id}{suffix}"
