import numpy as np
import pandas as pd

from src.pathology.transforms import center_crop


def random_projection_matrix(input_dim, output_dim=128, seed=42):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 1 / np.sqrt(output_dim), size=(input_dim, output_dim))
    return matrix.astype(np.float32)


def patch_statistics(image, grid_size=8):
    image = center_crop(image.convert("RGB"), min(image.size))
    image = image.resize((grid_size * 8, grid_size * 8))
    array = np.asarray(image, dtype=np.float32) / 255.0
    patch_h = array.shape[0] // grid_size
    patch_w = array.shape[1] // grid_size
    features = []
    for row in range(grid_size):
        for col in range(grid_size):
            patch = array[row * patch_h : (row + 1) * patch_h, col * patch_w : (col + 1) * patch_w]
            features.extend(patch.mean(axis=(0, 1)))
            features.extend(patch.std(axis=(0, 1)))
    return np.asarray(features, dtype=np.float32)


def extract_deep_features(image, output_dim=128, grid_size=8, seed=42):
    base = patch_statistics(image, grid_size=grid_size)
    projection = random_projection_matrix(len(base), output_dim=output_dim, seed=seed)
    embedding = np.tanh(base @ projection)
    return embedding.astype(np.float32)


def deep_features_dataframe(images, image_ids=None, output_dim=128, grid_size=8, seed=42):
    image_ids = image_ids or [f"tile_{idx:04d}" for idx in range(len(images))]
    embeddings = np.vstack(
        [extract_deep_features(image, output_dim=output_dim, grid_size=grid_size, seed=seed) for image in images]
    )
    columns = [f"deep_feature_{idx:03d}" for idx in range(output_dim)]
    df = pd.DataFrame(embeddings, columns=columns)
    df.insert(0, "image_id", image_ids)
    return df, embeddings