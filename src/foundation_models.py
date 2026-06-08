import hashlib

import numpy as np
import pandas as pd

from src.pathology.deep_features import extract_deep_features


SUPPORTED_FOUNDATION_MODELS = {
    "virchow": "Pathology foundation model placeholder for WSI/tile embeddings",
    "uni": "Universal pathology encoder placeholder",
    "conch": "Vision-language pathology encoder placeholder",
    "radimagenet": "Radiology/MRI transfer-learning encoder placeholder",
}


def stable_model_seed(model_name):
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def extract_foundation_embeddings(images, model_name="virchow", output_dim=256):
    if model_name not in SUPPORTED_FOUNDATION_MODELS:
        raise ValueError(f"Unsupported model_name={model_name}. Choose from {sorted(SUPPORTED_FOUNDATION_MODELS)}")
    seed = stable_model_seed(model_name)
    embeddings = [extract_deep_features(image, output_dim=output_dim, seed=seed) for image in images]
    return np.vstack(embeddings).astype(np.float32)


def foundation_embeddings_dataframe(images, image_ids=None, model_name="virchow", output_dim=256):
    image_ids = image_ids or [f"tile_{idx:04d}" for idx in range(len(images))]
    embeddings = extract_foundation_embeddings(images, model_name=model_name, output_dim=output_dim)
    columns = [f"{model_name}_feature_{idx:03d}" for idx in range(output_dim)]
    df = pd.DataFrame(embeddings, columns=columns)
    df.insert(0, "image_id", image_ids)
    df.insert(1, "foundation_model", model_name)
    return df, embeddings