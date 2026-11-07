"""Small configurable PyTorch networks for waveform and spectrogram inputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

# Frequency position carries the class cue for resonant transients, so the 2-D heads pool over time only.
_FREQ_BINS = 4


class CNN1D(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_len: int,
        channels: Sequence[int] = (16, 32, 64),
        kernel: int = 9,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for c in channels:
            layers += [
                nn.Conv1d(c_in, c, kernel, padding=kernel // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.MaxPool1d(4, ceil_mode=True),
            ]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T)
        return self.head(self.features(x.unsqueeze(1)))


class CNN2D(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_shape: tuple[int, int],
        channels: Sequence[int] = (16, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for c in channels:
            layers += [
                nn.Conv2d(c_in, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(),
                nn.MaxPool2d(2, ceil_mode=True),
            ]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, F, T)
        return self.head(self.features(x.unsqueeze(1)))


class _ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c_out)
        self.skip = (
            nn.Identity()
            if (c_in == c_out and stride == 1)
            else nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + self.skip(x))


class ResCNN2D(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_shape: tuple[int, int],
        width: int = 16,
        blocks: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU()
        )
        layers: list[nn.Module] = []
        c = width
        for i in range(blocks):
            nxt = width * 2 ** min(i, 2)
            layers.append(_ResBlock(c, nxt, 1 if i == 0 else 2))
            c = nxt
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x.unsqueeze(1))))


class AudioTransformer(nn.Module):
    """Research baseline: spectrogram frames as tokens -> small Transformer encoder -> mean pool."""

    def __init__(
        self,
        n_classes: int,
        in_shape: tuple[int, int],
        d_model: int = 48,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        f, t = in_shape
        self.proj = nn.Linear(f, d_model)
        self.pos = nn.Parameter(torch.zeros(1, t, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F, T)
        z = self.proj(x.transpose(1, 2)) + self.pos
        return self.head(self.norm(self.enc(z).mean(dim=1)))


class Normalized(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    """Input standardisation with statistics registered as buffers (fit on training data only)."""

    def __init__(self, net: nn.Module, stat_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.zeros(stat_shape))
        self.register_buffer("std", torch.ones(stat_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)
