from pathlib import Path
import pandas as pd
import openslide
import numpy as np
import torch
from torch.utils.data import Dataset


class PandaWSIDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None,
        tile_size=256,
        n_tiles=16
    ):
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)

        self.transform = transform
        self.tile_size = tile_size
        self.n_tiles = n_tiles

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = row["image_id"]
        label = int(row["isup_grade"])

        slide_path = self.image_dir / f"{image_id}.tiff"

        slide = openslide.OpenSlide(str(slide_path))

        level = slide.get_best_level_for_downsample(32)

        tiles = []

        width, height = slide.level_dimensions[level]

        for _ in range(self.n_tiles):

            x = np.random.randint(
                0,
                max(1, width - self.tile_size)
            )

            y = np.random.randint(
                0,
                max(1, height - self.tile_size)
            )

            tile = np.array(
                slide.read_region(
                    (x * 32, y * 32),
                    level,
                    (self.tile_size, self.tile_size)
                )
            )[:, :, :3]

            if self.transform:
                tile = self.transform(tile)

            tiles.append(tile)

        slide.close()

        tiles = torch.stack(tiles)

        return tiles, torch.tensor(label)
