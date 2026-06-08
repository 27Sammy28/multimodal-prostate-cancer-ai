import numpy as np
from PIL import Image


def tissue_mask(rgb_image, background_threshold=220):
    image = np.asarray(rgb_image.convert("RGB"))
    return np.mean(image, axis=2) < background_threshold


def extract_tiles(rgb_image, tile_size=256, stride=None, tissue_threshold=0.25):
    stride = stride or tile_size
    image = rgb_image.convert("RGB")
    width, height = image.size
    tiles = []
    for top in range(0, max(height - tile_size + 1, 1), stride):
        for left in range(0, max(width - tile_size + 1, 1), stride):
            tile = image.crop((left, top, left + tile_size, top + tile_size))
            if tile.size != (tile_size, tile_size):
                padded = Image.new("RGB", (tile_size, tile_size), "white")
                padded.paste(tile, (0, 0))
                tile = padded
            if tissue_mask(tile).mean() >= tissue_threshold:
                tiles.append({"tile": tile, "x": left, "y": top})
    return tiles
