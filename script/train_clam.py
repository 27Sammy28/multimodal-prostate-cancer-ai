import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Train a CLAM/attention MIL model.")
    parser.add_argument("--embeddings-dir", default="results/embeddings")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([{"model": "clam", "status": "template", "balanced_accuracy": None}])
    metrics.to_csv(output_dir / "tables" / "clam_metrics.csv", index=False)
    print("Wrote template metrics to", output_dir / "tables" / "clam_metrics.csv")


if __name__ == "__main__":
    main()
