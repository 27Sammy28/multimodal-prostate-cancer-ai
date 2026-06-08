from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.pathology.deep_features import deep_features_dataframe
from src.pathology.features import features_dataframe
from src.pathology.reduction import embedding_dataframe, umap_or_pca
from src.pathology.stain_normalization import macenko_normalize


def load_images(image_paths, limit=None):
    selected = list(image_paths)[:limit]
    images = [Image.open(path).convert("RGB") for path in selected]
    ids = [Path(path).stem for path in selected]
    return images, ids


def save_stain_example(original, normalized, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(normalized)
    axes[1].set_title("Macenko Normalized")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_embedding_plot(embedding, output_path, labels=None, title="Tile Embedding Space"):
    plt.figure(figsize=(6, 5))
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=28, alpha=0.85)
    else:
        labels = np.asarray(labels)
        for label in sorted(set(labels)):
            mask = labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1], s=28, alpha=0.85, label=str(label))
        plt.legend(title="label")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_tile_analysis_pipeline(images, image_ids=None, output_dir="results", labels=None):
    output_dir = Path(output_dir)
    for folder in ["figures", "tables", "embeddings", "reports"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    image_ids = image_ids or [f"tile_{idx:04d}" for idx in range(len(images))]
    normalized_images = [macenko_normalize(image) for image in images]
    if normalized_images:
        save_stain_example(images[0], normalized_images[0], output_dir / "figures" / "stain_normalization_macenko.png")

    handcrafted = features_dataframe(normalized_images, image_ids=image_ids)
    handcrafted.to_csv(output_dir / "tables" / "haralick_color_features.csv", index=False)

    deep_df, deep_embeddings = deep_features_dataframe(normalized_images, image_ids=image_ids)
    deep_df.to_csv(output_dir / "embeddings" / "deep_tile_embeddings.csv", index=False)

    reduced, method = umap_or_pca(deep_embeddings, n_components=2)
    reduced_df = embedding_dataframe(reduced, ids=image_ids, prefix="umap" if method == "umap" else "pca")
    reduced_df.to_csv(output_dir / "embeddings" / "tile_embedding_2d.csv", index=False)
    save_embedding_plot(reduced, output_dir / "figures" / "umap_tile_embeddings.png", labels=labels)

    summary = pd.DataFrame(
        [
            {
                "n_tiles": len(images),
                "n_handcrafted_features": handcrafted.shape[1] - 1,
                "n_deep_features": deep_embeddings.shape[1] if len(images) else 0,
                "dimensionality_reduction": method,
            }
        ]
    )
    summary.to_csv(output_dir / "reports" / "combined_pipeline_summary.csv", index=False)
    return {
        "normalized_images": normalized_images,
        "handcrafted_features": handcrafted,
        "deep_embeddings": deep_df,
        "embedding_2d": reduced_df,
        "summary": summary,
    }