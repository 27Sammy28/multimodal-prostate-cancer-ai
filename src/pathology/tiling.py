slide.read_region(...)

import openslide
import numpy as np


def extract_tiles(
    slide_path,
    tile_size=256,
    n_tiles=32,
    downsample=32
):
    slide = openslide.OpenSlide(slide_path)

    level = slide.get_best_level_for_downsample(
        downsample
    )

    tiles = []

    width, height = slide.level_dimensions[level]

    for _ in range(n_tiles):

        x = np.random.randint(
            0,
            width - tile_size
        )

        y = np.random.randint(
            0,
            height - tile_size
        )

        tile = np.array(
            slide.read_region(
                (x * downsample, y * downsample),
                level,
                (tile_size, tile_size)
            )
        )[:, :, :3]

        tiles.append(tile)

    slide.close()

    return tiles
