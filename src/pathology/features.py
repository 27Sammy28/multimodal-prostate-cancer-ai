import numpy as np
import pandas as pd
from PIL import Image


def rgb_array(image):
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def grayscale_array(image):
    rgb = rgb_array(image).astype(np.float32)
    return np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0, 255).astype(np.uint8)


def color_histogram_features(image, bins=16):
    rgb = rgb_array(image)
    features = {}
    for channel_index, channel_name in enumerate(["red", "green", "blue"]):
        hist, _ = np.histogram(rgb[..., channel_index], bins=bins, range=(0, 256), density=True)
        for bin_index, value in enumerate(hist):
            features[f"color_hist_{channel_name}_{bin_index:02d}"] = float(value)
        channel = rgb[..., channel_index].astype(np.float32)
        features[f"{channel_name}_mean"] = float(channel.mean())
        features[f"{channel_name}_std"] = float(channel.std())
    return features


def gray_level_cooccurrence(gray, levels=32, distance=1, angle=(0, 1)):
    gray = (gray.astype(np.float32) / 256.0 * levels).astype(np.int32).clip(0, levels - 1)
    dy, dx = angle
    if dy >= 0:
        rows_a = slice(0, gray.shape[0] - dy)
        rows_b = slice(dy, gray.shape[0])
    else:
        rows_a = slice(-dy, gray.shape[0])
        rows_b = slice(0, gray.shape[0] + dy)
    if dx >= 0:
        cols_a = slice(0, gray.shape[1] - dx)
        cols_b = slice(dx, gray.shape[1])
    else:
        cols_a = slice(-dx, gray.shape[1])
        cols_b = slice(0, gray.shape[1] + dx)
    first = gray[rows_a, cols_a].ravel()
    second = gray[rows_b, cols_b].ravel()
    matrix = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(matrix, (first, second), 1)
    matrix += matrix.T
    total = matrix.sum()
    return matrix / total if total > 0 else matrix


def haralick_features(image, levels=32):
    gray = grayscale_array(image)
    angles = [(0, 1), (1, 0), (1, 1), (1, -1)]
    values = []
    i, j = np.indices((levels, levels))
    for angle in angles:
        p = gray_level_cooccurrence(gray, levels=levels, angle=angle)
        contrast = np.sum(((i - j) ** 2) * p)
        dissimilarity = np.sum(np.abs(i - j) * p)
        homogeneity = np.sum(p / (1.0 + (i - j) ** 2))
        energy = np.sqrt(np.sum(p**2))
        asm = np.sum(p**2)
        mean_i = np.sum(i * p)
        mean_j = np.sum(j * p)
        std_i = np.sqrt(np.sum(((i - mean_i) ** 2) * p))
        std_j = np.sqrt(np.sum(((j - mean_j) ** 2) * p))
        correlation = np.sum((i - mean_i) * (j - mean_j) * p) / max(std_i * std_j, 1e-8)
        entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
        values.append([contrast, dissimilarity, homogeneity, energy, asm, correlation, entropy])
    names = ["contrast", "dissimilarity", "homogeneity", "energy", "asm", "correlation", "entropy"]
    array = np.asarray(values)
    features = {}
    for index, name in enumerate(names):
        features[f"haralick_{name}_mean"] = float(array[:, index].mean())
        features[f"haralick_{name}_std"] = float(array[:, index].std())
    return features


def extract_handcrafted_features(image, bins=16, levels=32):
    features = {}
    features.update(color_histogram_features(image, bins=bins))
    features.update(haralick_features(image, levels=levels))
    return features


def features_dataframe(images, image_ids=None, bins=16, levels=32):
    rows = []
    image_ids = image_ids or [f"tile_{idx:04d}" for idx in range(len(images))]
    for image_id, image in zip(image_ids, images):
        row = {"image_id": image_id}
        row.update(extract_handcrafted_features(image, bins=bins, levels=levels))
        rows.append(row)
    return pd.DataFrame(rows)