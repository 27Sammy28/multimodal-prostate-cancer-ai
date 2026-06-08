try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


class MultimodalFusionTransformer(nn.Module if nn else object):
    def __init__(self, modality_dims, hidden_dim=256, num_heads=4, num_layers=2, num_classes=2):
        if nn is None:
            raise ImportError("Install PyTorch to use MultimodalFusionTransformer.")
        super().__init__()
        self.modalities = list(modality_dims)
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, hidden_dim) for name, dim in modality_dims.items()}
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, features_by_modality):
        tokens = []
        for name in self.modalities:
            tokens.append(self.projections[name](features_by_modality[name]).unsqueeze(1))
        encoded = self.encoder(torch.cat(tokens, dim=1))
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)