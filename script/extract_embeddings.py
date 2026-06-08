import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Extract tile or WSI embeddings.")
    parser.add_argument("--output-dir", default="results/embeddings")
    parser.add_argument("--name", default="virchow_embeddings.npy")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    placeholder = np.empty((0, 0), dtype=np.float32)
    np.save(output_dir / args.name, placeholder)
    print("Wrote placeholder embeddings to", output_dir / args.name)


if __name__ == "__main__":
    main()
