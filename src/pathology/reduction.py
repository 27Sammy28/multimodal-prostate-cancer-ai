import numpy as np
import pandas as pd


def standardize(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0)
    std = np.where(matrix.std(axis=0) == 0, 1.0, matrix.std(axis=0))
    return (matrix - mean) / std


def pca_reduce(matrix, n_components=2):
    x = standardize(matrix)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[:n_components]
    embedding = x @ components.T
    return embedding.astype(np.float32)


def umap_or_pca(matrix, n_components=2, random_state=42):
    try:
        import umap

        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        method = "umap"
        embedding = reducer.fit_transform(matrix)
    except Exception:
        method = "pca_fallback"
        embedding = pca_reduce(matrix, n_components=n_components)
    return embedding, method


def embedding_dataframe(embedding, ids=None, prefix="umap"):
    ids = ids or [f"item_{idx:04d}" for idx in range(len(embedding))]
    df = pd.DataFrame(embedding, columns=[f"{prefix}_{idx + 1}" for idx in range(embedding.shape[1])])
    df.insert(0, "image_id", ids)
    return df