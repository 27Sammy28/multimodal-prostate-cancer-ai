import numpy as np


def normalize_stain(image):

    image = image.astype(np.float32)

    for channel in range(3):

        p2, p98 = np.percentile(
            image[:, :, channel],
            (2, 98)
        )

        image[:, :, channel] = (
            np.clip(
                (image[:, :, channel] - p2)
                / (p98 - p2),
                0,
                1,
            )
            * 255
        )

    return image.astype(np.uint8)
