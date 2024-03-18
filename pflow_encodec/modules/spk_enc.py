from typing import Literal, Optional

import torch
import torch.nn as nn

from pflow_encodec.modules.transformer import (
    MultiHeadAttention,
    Transformer,
    Wav2Vec2StackedPositionEncoder,
)


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        dim_input: int,
        conv_pos_kernel_size: int,
        conv_pos_depth: int,
        conv_pos_groups: int,
        depth: int,
        dim: int,
        dim_head: int,
        heads: int,
        ff_mult: float,
        attn_dropout: float,
        ff_dropout: float,
        attn_processor: Literal["naive", "sdpa"] = "naive",
        norm_type: Literal["layer", "ada_proj", "ada_embed"] = "layer",
        ff_type: Literal["conv", "linear"] = "linear",
        ff_kernel_size: Optional[int] = None,
        ff_groups: Optional[int] = None,
        layer_norm_eps: float = 1e-6,
        scale_type: Literal["none", "ada_proj", "ada_embed"] = "none",
        pool_query_range: float = 0.02,
    ):
        super().__init__()
        self.proj = nn.Linear(dim_input, dim)

        self.conv_pos = Wav2Vec2StackedPositionEncoder(
            depth=conv_pos_depth,
            dim=dim,
            kernel_size=conv_pos_kernel_size,
            groups=conv_pos_groups,
        )
        self.encoder = Transformer(
            depth=depth,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            attn_processor=attn_processor,
            norm_type=norm_type,
            ff_type=ff_type,
            ff_kernel_size=ff_kernel_size,
            ff_groups=ff_groups,
            layer_norm_eps=layer_norm_eps,
            scale_type=scale_type,
        )
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.trunc_normal_(self.query, mean=0, std=pool_query_range)

        self.pool = MultiHeadAttention(
            dim=dim,
            dim_head=dim,
            heads=1,
            processor=attn_processor,
        )

    def reset_parameters(self):
        self.conv_pos.reset_parameters()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.proj(x)
        x = x + self.conv_pos(x, mask)
        x = self.encoder(x, mask)
        emb = self.pool(self.query, context=x, mask=mask)
        return emb
