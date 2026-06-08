import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from src.pathology.pipeline import load_images, run_tile_analysis_pipeline


def make_synthetic_tiles(n_tiles=24, size=128, seed=42):
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for idx in range(n_tiles):
        base = np.full((size, size, 3), [232, 205, 220], dtype=np.uint8)
        label = int(idx >= n_tiles // 2)
        n_blobs = rng.integers(12, 28) + label * 8
        image = Image.fromarray(base)
        draw = ImageDraw.Draw(image, "RGBA")
        for _ in range(n_blobs):
            x = int(rng.integers(0, size - 12))
            y = int(rng.integers(0, size - 12))
            radius = int(rng.integers(4, 18))
            color = (95 + label * 35, 45, 130 + label * 20, int(rng.integers(70, 150)))
            draw.ellipse((x, y, x + radius, y + radius), fill=color)
        images.append(image.convert("RGB"))
        labels.append(label)
    ids = [f"synthetic_tile_{idx:03d}" for idx in range(n_tiles)]
    return images, ids, labels


def main():
    parser = argparse.ArgumentParser(description="Run Macenko + Haralick/color + deep features + UMAP/PCA pipeline.")
    parser.add_argument("--image-dir", help="Optional directory of tile images.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.image_dir and not args.synthetic:
        paths = []
        for pattern in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
            paths.extend(Path(args.image_dir).glob(pattern))
        images, image_ids = load_images(sorted(paths), limit=args.limit)
        labels = None
    else:
        images, image_ids, labels = make_synthetic_tiles(n_tiles=min(args.limit, 32))

    outputs = run_tile_analysis_pipeline(images, image_ids=image_ids, labels=labels, output_dir=args.output_dir)
    print(outputs["summary"].to_string(index=False))


if __name__ == "__main__":
    main()