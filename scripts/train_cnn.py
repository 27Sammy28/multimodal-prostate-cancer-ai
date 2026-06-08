import torch

from models.cnn import EfficientNetBaseline


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = EfficientNetBaseline()

    model.to(device)

    print(model)

    print(
        f"Training on {device}"
    )


if __name__ == "__main__":
    main()
