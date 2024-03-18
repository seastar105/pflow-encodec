from typing import Optional

import torch
import torch.nn as nn

from pflow_encodec.modules.transformer import Wav2Vec2PositionEncoderLayer
from pflow_encodec.utils.helper import exists


class DurationPredictor(nn.Module):
    def __init__(self, dim_input: int, dim: int, depth: int, kernel_size: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(dim_input, dim)

        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for _ in range(depth):
            self.convs.append(Wav2Vec2PositionEncoderLayer(dim, kernel_size, groups=1))
        self.output_proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        if exists(mask):
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = self.dropout(x)
        x = x.transpose(1, 2)
        if exists(mask):
            x = x.masked_fill(~mask, 0.0)
        return self.output_proj(x).squeeze(-1)
