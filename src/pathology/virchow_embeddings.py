import torch


class VirchowEncoder:

    def __init__(self, model):

        self.model = model
        self.model.eval()

    def encode(self, tile):

        with torch.no_grad():

            embedding = self.model(tile)

        return embedding
