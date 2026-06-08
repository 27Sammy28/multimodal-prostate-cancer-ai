try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


class AttentionMIL(nn.Module if nn else object):
    def __init__(self, embedding_dim, hidden_dim=256, num_classes=6):
        if nn is None:
            raise ImportError("Install PyTorch to use AttentionMIL.")
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings):
        weights = torch.softmax(self.attention(embeddings), dim=0)
        bag_embedding = torch.sum(weights * embeddings, dim=0)
        return self.classifier(bag_embedding), weights
