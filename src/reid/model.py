from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class EmbeddingNet(nn.Module):
    """ResNet18 backbone for metric-learning embeddings."""

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        emb = self.head(feats)
        return nn.functional.normalize(emb, p=2, dim=1)
