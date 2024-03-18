from typing import Literal, Optional

import torch
import torch.nn as nn

from pflow_encodec.modules.transformer import (
    AlibiPositionalBias,
    Transformer,
    Wav2Vec2StackedPositionEncoder,
)


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim_text: int,
        dim_spk: int,
        dim_output: int,
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

        self.text_emb = nn.Embedding(vocab_size, dim_text)
        self.input_proj = nn.Linear(dim_text, dim)

        self.conv_pos = Wav2Vec2StackedPositionEncoder(
            depth=conv_pos_depth,
            dim=dim,
            kernel_size=conv_pos_kernel_size,
            groups=conv_pos_groups,
        )

        self.norm_type = norm_type
        self.scale_type = scale_type

        if norm_type == "ada_embed":
            self.adaln_linear = nn.Linear(dim_spk, dim * 4)
        if scale_type == "ada_embed":
            self.ada_scale_linear = nn.Linear(dim_spk, dim * 2)

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
            dim_cond=dim if norm_type == "ada_embed" else dim_spk,
        )

        self.output_proj = nn.Linear(dim, dim_output)

        self.alibi = AlibiPositionalBias(heads)

    def reset_parameters(self):
        self.conv_pos.reset_parameters()

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
        text_tokens: torch.Tensor,
        spk_emb: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        x = self.input_proj(self.text_emb(text_tokens))
        x = x + self.conv_pos(x, padding_mask)

        cond = spk_emb
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
        return self.output_proj(x), x
