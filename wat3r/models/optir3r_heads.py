"""Small condition heads for the Opti3R second-stage experiments."""
from __future__ import annotations

import torch
from torch import nn


class OpticsConditionHead(nn.Module):
    """Predict a bounded optical proxy and a near-identity FiLM condition.

    The proxy is intentionally not interpreted as a calibrated physical
    measurement. It is supervised only on synthetic rendering statistics and
    is used as a condition variable for controlled intervention/swap tests.
    """

    def __init__(self, token_dim: int, embedding_dim: int = 64, params_dim: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.embedding = nn.Sequential(
            nn.Linear(token_dim, 256), nn.GELU(), nn.Linear(256, embedding_dim),
        )
        self.params = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, params_dim), nn.Sigmoid())
        self.quality = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.film = nn.Linear(embedding_dim, 2 * token_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, tokens: torch.Tensor, embedding_override: torch.Tensor | None = None):
        embedding = self.embedding(self.norm(tokens[:, :, 0])) if embedding_override is None else embedding_override
        return {
            "optics_embedding": embedding,
            "optics_params": self.params(embedding),
            "optics_quality": self.quality(embedding),
            "optics_film": self.film(embedding),
        }
