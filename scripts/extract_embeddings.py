from pathlib import Path


def main():

    print(
        "Extracting Virchow embeddings..."
    )

    output_dir = Path(
        "results/embeddings"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


if __name__ == "__main__":
    main()
