import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Train an EfficientNet/CNN baseline.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame([{"model": "cnn_baseline", "status": "template", "accuracy": None}])
    metrics.to_csv(output_dir / "tables" / "cnn_metrics.csv", index=False)
    print("Wrote template metrics to", output_dir / "tables" / "cnn_metrics.csv")


if __name__ == "__main__":
    main()
