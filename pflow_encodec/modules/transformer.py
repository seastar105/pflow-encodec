import math
from functools import partial
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from torch.nn.utils import remove_weight_norm, weight_norm

from pflow_encodec.utils.helper import exists


class AdaptiveLayerNormProj(nn.Module):
    def __init__(self, dim: int, dim_cond: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.scale = nn.Linear(dim_cond, dim)
        self.bias = nn.Linear(dim_cond, dim)

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.bias.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond)
        bias = self.bias(cond)
        return self.norm(x) * (1 + scale) + bias


class AdaptiveLayerNormEmbed(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.scale = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, bias = cond.chunk(2, dim=-1)
        scale = self.scale + scale
        bias = self.bias + bias
        return self.norm(x) * (1 + scale) + bias


class AdaptiveScaleProj(nn.Module):
    def __init__(self, dim: int, dim_cond: int):
        super().__init__()
        self.scale = nn.Linear(dim_cond, dim)

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond)
        return x * scale


class AdaptiveScaleEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.scale + cond
        return x * scale


class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        x, gate = x.chunk(2, dim=dim)
        return F.gelu(gate) * x


class ConvFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float, kernel_size: int, groups: int = 1, dropout: float = 0.0):
        super().__init__()
        intermediate_dim = int(dim * mult * 3 / 4)
        self.conv1 = nn.Conv1d(dim, 2 * intermediate_dim, kernel_size, padding="same", groups=groups)
        self.act = GEGLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(intermediate_dim, dim, kernel_size, padding="same", groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b t d -> b d t")
        x = self.conv1(x)
        x = self.act(x, dim=1)
        x = self.dropout(x)
        x = self.conv2(x)
        x = rearrange(x, "b d t -> b t d")
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float, dropout: float = 0.0):
        super().__init__()
        intermediate_dim = int(dim * mult * 2 / 3)
        self.proj1 = nn.Linear(dim, 2 * intermediate_dim)
        self.act = GEGLU()
        self.dropout = nn.Dropout(dropout)
        self.proj2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.proj2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        dim_context: Optional[int] = None,
        scale: Optional[float] = None,
        dropout: float = 0.0,
        processor: Literal["naive", "sdpa", "flash"] = "naive",
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.dim_context = dim_context if exists(dim_context) else dim
        self.scale = scale if exists(scale) else dim_head ** -0.5
        self.processor = processor
        self.heads = heads

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(self.dim_context, inner_dim, bias=False)
        self.to_v = nn.Linear(self.dim_context, inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)  # apply to attn score

        self.to_out = nn.Linear(inner_dim, dim)

        self.attn_processor_dict = {
            "naive": self.naive_attention,
            "sdpa": self.sdpa_attention,
        }

        if self.processor not in self.attn_processor_dict:
            raise NotImplementedError(f"processor {self.processor} is not implemented yet")

    def process_attn_mask_bias(self, mask, bias):
        if not exists(bias):
            return mask, False

        if exists(mask):
            bias = bias.masked_fill(~mask, -torch.finfo(bias.dtype).max)
        return bias, True

    def naive_attention(self, q, k, v, mask, bias, **attn_kwargs):
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        attn_mask, is_bias = self.process_attn_mask_bias(mask, bias)
        dots = einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale

        if exists(attn_mask):
            if is_bias:
                dots = dots + attn_mask
            else:
                dots.masked_fill_(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum(attn, v, "b h i j, b h j d -> b h i d")
        out = rearrange(out, "b h n d -> b n (h d)")

        return out

    def sdpa_attention(self, q, k, v, mask, bias, **attn_kwargs):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError(
                "torch.nn.functional.scaled_dot_product_attention is not available. Please upgrade to PyTorch 2.0.0 or later."
            )
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        attn_mask, _ = self.process_attn_mask_bias(mask, bias)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out

    def get_attn_processor(self, processor):
        assert processor in self.attn_processor_dict, f"processor {processor} is not implemented yet"
        return self.attn_processor_dict[processor]

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        **attn_kwargs,
    ):
        if not exists(context):
            context = x

        b, t, d = x.shape
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        attn_output = self.get_attn_processor(self.processor)(q, k, v, mask, bias, **attn_kwargs)

        return self.to_out(attn_output)


# code from https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/models/wav2vec2/position_encoder.py
class Wav2Vec2PositionEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        groups: int,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            padding="same",
            groups=groups,
        )

        self.layer_norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        encodings = self.conv(encodings)

        encodings = encodings.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        encodings = self.layer_norm(encodings)
        encodings = encodings.transpose(1, 2)  # (B, T, D) -> (B, D, T)

        encodings = self.activation(encodings)
        return encodings


class Wav2Vec2StackedPositionEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        kernel_size: int,
        groups: int,
    ) -> None:
        super().__init__()

        k = max(3, kernel_size // depth)

        self.layers = nn.Sequential()

        for _ in range(depth):
            layer = Wav2Vec2PositionEncoderLayer(
                dim,
                k,
                groups,
            )

            self.layers.append(layer)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if exists(mask):
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)

        if exists(mask):
            x = x.masked_fill(~mask, 0.0)

        return x

    def reset_parameters(self):
        def init_(m):
            if isinstance(m, nn.Conv1d):
                model_dim, kernel_size = m.in_channels, m.kernel_size[0]
                try:
                    remove_weight_norm(m)
                except ValueError:
                    # Raised during the `__init__` call since we don't have the weight
                    # norm hook registered yet. Safe to ignore.
                    pass

                nn.init.normal_(m.weight, mean=0.0, std=(4.0 / (kernel_size * model_dim)) ** 0.5)

                weight_norm(m, dim=2)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.apply(init_)


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads

        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> 1 h 1 1")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def get_bias(self, seq_len: int):
        i_arange = torch.arange(seq_len, device=self.device)
        j_arange = torch.arange(seq_len, device=self.device)
        bias = -torch.abs(rearrange(j_arange, "j -> 1 1 j") - rearrange(i_arange, "i -> 1 i 1"))
        return bias.unsqueeze(0)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: heads - closest_power_of_2]
        )

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len: int):
        if exists(self.bias) and self.bias.shape[-1] >= seq_len:
            return self.bias[..., -seq_len:, -seq_len:]

        bias = self.get_bias(seq_len)
        bias = bias * self.slopes

        self.register_buffer("bias", bias, persistent=False)

        return self.bias


class Transformer(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        dim_head: int,
        heads: int,
        ff_mult: float,
        attn_dropout: float,
        ff_dropout: float,
        dim_cond: Optional[int] = None,
        attn_processor: Literal["naive", "sdpa"] = "naive",
        norm_type: Literal["layer", "ada_proj", "ada_embed"] = "layer",
        ff_type: Literal["conv", "linear"] = "linear",
        ff_kernel_size: Optional[int] = None,
        ff_groups: Optional[int] = None,
        layer_norm_eps: float = 1e-6,
        scale_type: Literal["none", "ada_proj", "ada_embed"] = "none",
        use_skip_connection: bool = False,
        dim_final_norm_cond: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.norm_type = norm_type
        norm_class = self.get_norm_class(norm_type, dim_cond)

        self.ff_type = ff_type
        ff_class = self.get_ff_class(ff_type, ff_kernel_size, ff_groups)

        self.scale_type = scale_type
        if self.scale_type != "none":
            assert (
                self.norm_type == self.scale_type
            ), f"norm type {self.norm_type} and scale type {self.scale_type} must be the same"
        scale_class = self.get_scale_class(scale_type, dim, dim_cond)

        self.layers = nn.ModuleList([])
        for ind in range(depth):
            layer = ind + 1
            has_skip = use_skip_connection and layer > (depth // 2)
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Linear(dim * 2, dim) if has_skip else None,
                        norm_class(dim, eps=layer_norm_eps),
                        MultiHeadAttention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            scale=None,
                            dropout=attn_dropout,
                            processor=attn_processor,
                        ),
                        scale_class(),
                        norm_class(dim, eps=layer_norm_eps),
                        ff_class(dim=dim, mult=ff_mult, dropout=ff_dropout),
                        scale_class(),
                    ]
                )
            )

        if self.norm_type == "ada_embed":
            assert exists(dim_final_norm_cond), "dim_final_norm_cond must be provided when using ada_embed"

        self.final_norm = (
            nn.LayerNorm(dim, eps=layer_norm_eps)
            if self.norm_type == "layer"
            else AdaptiveLayerNormProj(
                dim, dim_cond=dim_cond if self.norm_type == "ada_proj" else dim_final_norm_cond, eps=layer_norm_eps
            )
        )

    def reset_adaln_parameters(self):
        def init_(m):
            if isinstance(m, AdaptiveLayerNormProj):
                m.reset_parameters()

        self.apply(init_)

    @staticmethod
    def expand_mask(mask: Optional[torch.Tensor] = None):
        if exists(mask):
            if mask.ndim == 2:  # B L
                mask = rearrange(mask, "b j -> b 1 1 j")
            elif mask.ndim == 3:  # B q_len k_len
                mask = rearrange(mask, "b i j -> b 1 i j")
        return mask

    @staticmethod
    def get_norm_class(norm_type, dim_cond):
        if norm_type == "layer":
            return nn.LayerNorm
        elif norm_type == "ada_proj":
            return partial(AdaptiveLayerNormProj, dim_cond=dim_cond)
        elif norm_type == "ada_embed":
            return AdaptiveLayerNormEmbed
        else:
            raise NotImplementedError(f"norm type {norm_type} is not implemented yet")

    @staticmethod
    def get_scale_class(scale_type, dim, dim_cond):
        if scale_type == "none":
            return nn.Identity
        elif scale_type == "ada_proj":
            return partial(AdaptiveScaleProj, dim=dim, dim_cond=dim_cond)
        elif scale_type == "ada_embed":
            return partial(AdaptiveScaleEmbed, dim=dim)
        else:
            raise NotImplementedError(f"scale type {scale_type} is not implemented yet")

    @staticmethod
    def get_ff_class(ff_type, kernel_size, groups):
        if ff_type == "conv":
            return partial(ConvFeedForward, kernel_size=kernel_size, groups=groups)
        elif ff_type == "linear":
            return FeedForward
        else:
            raise NotImplementedError(f"ff type {ff_type} is not implemented yet")

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        cond_input: Dict[str, torch.Tensor] = dict(),
    ):
        mask = self.expand_mask(mask)
        if exists(bias):
            assert bias.ndim == 4, f"bias must have 4 dimensions in Transformer, got {bias.ndim}"

        skip_connects = []
        for skip_combiner, attn_norm, attn, attn_scale, ff_norm, ff, ff_scale in self.layers:
            if not exists(skip_combiner):
                skip_connects.append(x)
            else:
                skip_connect = skip_connects.pop()
                x = torch.cat([x, skip_connect], dim=-1)
                x = skip_combiner(x)
            residual = x
            if self.norm_type == "layer":
                x = attn_norm(x)
            else:
                x = attn_norm(x, cond=cond_input.get("attn_norm_cond", None))
            x = attn(x, context=context, mask=mask, bias=bias)
            if self.scale_type != "none":
                x = attn_scale(x, cond=cond_input.get("attn_scale_cond", None))
            x = x + residual

            residual = x
            if self.norm_type == "layer":
                x = ff_norm(x)
            else:
                x = ff_norm(x, cond=cond_input.get("ff_norm_cond", None))
            x = ff(x)
            if self.scale_type != "none":
                x = ff_scale(x, cond=cond_input.get("ff_scale_cond", None))
            x = x + residual

        final_output = (
            self.final_norm(x)
            if self.norm_type == "layer"
            else self.final_norm(x, cond=cond_input.get("final_norm_cond", None))
        )
        return final_output
