import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from einops import rearrange

from pflow_encodec.modules.transformer import (
    AlibiPositionalBias,
    Transformer,
    Wav2Vec2StackedPositionEncoder,
)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, dim_time: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)
        self.net = nn.Sequential(
            nn.Linear(dim, dim_time),
            nn.SiLU(),
            nn.Linear(dim_time, dim_time),
        )

    def forward(self, t: torch.Tensor):
        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.net(embedding)


class FlowMatchingTransformer(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_ctx: int,
        dim_output: int,
        dim_time: int,
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
    ):
        super().__init__()

        self.input_proj = nn.Linear(dim_input + dim_ctx, dim)

        self.time_embed = TimestepEmbedder(dim, dim_time)
        self.conv_pos = Wav2Vec2StackedPositionEncoder(
            depth=conv_pos_depth,
            dim=dim,
            kernel_size=conv_pos_kernel_size,
            groups=conv_pos_groups,
        )

        self.norm_type = norm_type
        self.scale_type = scale_type

        if norm_type == "ada_embed":
            self.adaln_linear = nn.Linear(dim_time, dim * 4)
        if scale_type == "ada_embed":
            self.ada_scale_linear = nn.Linear(dim_time, dim * 2)

        self.transformer = Transformer(
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
            dim_cond=dim if norm_type == "ada_embed" else dim_time,
            dim_final_norm_cond=dim_time if norm_type == "ada_embed" else None,
        )

        self.output_proj = nn.Linear(dim, dim_output)

        self.alibi = AlibiPositionalBias(heads)

    def reset_parameters(self):
        self.conv_pos.reset_parameters()
        nn.init.trunc_normal_(self.time_embed.net[0].weight, std=0.02)
        nn.init.zeros_(self.time_embed.net[0].bias)
        nn.init.trunc_normal_(self.time_embed.net[2].weight, std=0.02)
        nn.init.zeros_(self.time_embed.net[2].bias)

        # init adaln
        if self.norm_type == "ada_embed":
            nn.init.zeros_(self.adaln_linear.weight)
            nn.init.zeros_(self.adaln_linear.bias)

        if self.scale_type == "ada_embed":
            nn.init.zeros_(self.ada_scale_linear.weight)
            nn.init.zeros_(self.ada_scale_linear.bias)

        self.transformer.reset_adaln_parameters()

        # zero init output proj
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        x_ctx: torch.Tensor,
        times: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        drop_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # apply dropout to context
        x_ctx = torch.where(rearrange(drop_ctx, "b -> b 1 1"), 0, x_ctx)

        x = torch.cat([x, x_ctx], dim=-1)
        x = self.input_proj(x)

        x = x + self.conv_pos(x, mask=padding_mask)
        cond = self.time_embed(times).unsqueeze(1)

        cond_input = dict()
        if self.norm_type == "ada_proj":
            cond_input["attn_norm_cond"] = cond
            cond_input["ff_norm_cond"] = cond
            cond_input["final_norm_cond"] = cond
        elif self.norm_type == "ada_embed":
            attn_norm_scale, attn_norm_bias, ff_norm_scale, ff_norm_bias = self.adaln_linear(cond).chunk(4, dim=-1)
            cond_input["attn_norm_cond"] = torch.cat([attn_norm_scale, attn_norm_bias], dim=-1)
            cond_input["ff_norm_cond"] = torch.cat([ff_norm_scale, ff_norm_bias], dim=-1)
            cond_input["final_norm_cond"] = cond

        if self.scale_type == "ada_proj":
            cond_input["attn_scale_cond"] = cond
            cond_input["ff_scale_cond"] = cond
        elif self.scale_type == "ada_embed":
            attn_scale, ff_scale = self.ada_scale_linear(cond).chunk(2, dim=-1)
            cond_input["attn_scale_cond"] = attn_scale
            cond_input["ff_scale_cond"] = ff_scale

        seq_len = x.size(1)
        bias = self.alibi(seq_len)
        x = self.transformer(x, mask=padding_mask, cond_input=cond_input, bias=bias)

        return self.output_proj(x)
