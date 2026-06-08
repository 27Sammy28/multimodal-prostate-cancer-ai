from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def permutation_importance(model, x, y, metric_fn, feature_names=None, n_repeats=5, random_state=42):
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    baseline = metric_fn(y, model.predict_proba(x)[:, 1])
    names = feature_names or [f"feature_{idx}" for idx in range(x.shape[1])]
    rows = []
    for col_idx, name in enumerate(names):
        scores = []
        for _ in range(n_repeats):
            shuffled = x.copy()
            shuffled[:, col_idx] = rng.permutation(shuffled[:, col_idx])
            scores.append(metric_fn(y, model.predict_proba(shuffled)[:, 1]))
        rows.append(
            {
                "feature": name,
                "baseline_score": float(baseline),
                "permuted_score": float(np.mean(scores)),
                "importance": float(baseline - np.mean(scores)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def saliency_map_from_image(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    grad_y, grad_x = np.gradient(gray)
    saliency = np.sqrt(grad_x**2 + grad_y**2)
    saliency = saliency / max(float(saliency.max()), 1e-8)
    return saliency


def save_saliency_overlay(image, output_path, alpha=0.45):
    saliency = saliency_map_from_image(image)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(image.convert("RGB"))
    plt.imshow(saliency, cmap="magma", alpha=alpha)
    plt.axis("off")
    plt.title("Explainable AI Saliency Overlay")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return saliency


def save_feature_importance_plot(importance_df, output_path, top_k=20):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = importance_df.head(top_k).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.35 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Permutation importance")
    plt.title("Explainable AI Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()