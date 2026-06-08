from pathlib import Path
import pandas as pd


class PandaDataset:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return self.df["isup_grade"]

    def get_image_ids(self):
        return self.df["image_id"].tolist()

    def get_row(self, idx):
        return self.df.iloc[idx]
