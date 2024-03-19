from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
from einops import rearrange

from pflow_encodec.modules import (
    DurationPredictor,
    FlowMatchingTransformer,
    SpeakerEncoder,
    TextEncoder,
)


class PFlow(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        text_encoder_vocab_size: int = 10094,
        text_encoder_embed_dim: int = 192,
        text_encoder_conv_pos_depth: int = 2,
        text_encoder_conv_pos_kernel_size: int = 15,
        text_encoder_conv_pos_groups: int = 16,
        text_encoder_depth: int = 6,
        text_encoder_dim: int = 192,
        text_encoder_dim_head: int = 96,
        text_encoder_heads: int = 2,
        text_encoder_ff_mult: float = 4.0,
        text_encoder_attn_dropout: float = 0.1,
        text_encoder_ff_dropout: float = 0.0,
        text_encoder_attn_processor: str = "naive",
        text_encoder_norm_type: str = "ada_proj",
        text_encoder_ff_type: str = "conv",
        text_encoder_ff_kernel_size: int = 3,
        text_encoder_ff_groups: int = 1,
        text_encoder_scale_type: str = "ada_proj",
        speaker_encoder_dim_input: int = 128,
        speaker_encoder_conv_pos_depth: int = 2,
        speaker_encoder_conv_pos_kernel_size: int = 15,
        speaker_encoder_conv_pos_groups: int = 16,
        speaker_encoder_depth: int = 2,
        speaker_encoder_dim: int = 192,
        speaker_encoder_dim_head: int = 96,
        speaker_encoder_heads: int = 2,
        speaker_encoder_ff_mult: float = 4.0,
        speaker_encoder_attn_dropout: float = 0.1,
        speaker_encoder_ff_dropout: float = 0.0,
        speaker_encoder_attn_processor: str = "naive",
        speaker_encoder_norm_type: str = "layer",
        speaker_encoder_ff_type: str = "conv",
        speaker_encoder_ff_kernel_size: int = 3,
        speaker_encoder_ff_groups: int = 1,
        speaker_encoder_scale_type: str = "none",
        flow_matching_dim_time: int = 2048,
        flow_matching_conv_pos_kernel_size: int = 31,
        flow_matching_conv_pos_depth: int = 2,
        flow_matching_conv_pos_groups: int = 16,
        flow_matching_depth: int = 6,
        flow_matching_dim: int = 512,
        flow_matching_dim_head: int = 128,
        flow_matching_heads: int = 4,
        flow_matching_ff_mult: float = 4.0,
        flow_matching_attn_dropout: float = 0.1,
        flow_matching_ff_dropout: float = 0.0,
        flow_matching_attn_processor: str = "naive",
        flow_matching_norm_type: str = "ada_embed",
        flow_matching_ff_type: str = "conv",
        flow_matching_ff_kernel_size: int = 3,
        flow_matching_ff_groups: int = 2,
        flow_matching_scale_type: str = "ada_embed",
        duration_predictor_dim: int = 256,
        duration_predictor_depth: int = 2,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout: float = 0.1,
        p_uncond: float = 0.1,
        interpolate_mode: str = "linear",
        sigma: float = 0.01,  # from pflow paper
    ):
        super().__init__()

        self.text_encoder = TextEncoder(
            vocab_size=text_encoder_vocab_size,
            dim_text=text_encoder_embed_dim,
            dim_spk=speaker_encoder_dim,
            dim_output=feature_dim,
            conv_pos_kernel_size=text_encoder_conv_pos_kernel_size,
            conv_pos_depth=text_encoder_conv_pos_depth,
            conv_pos_groups=text_encoder_conv_pos_groups,
            depth=text_encoder_depth,
            dim=text_encoder_dim,
            dim_head=text_encoder_dim_head,
            heads=text_encoder_heads,
            ff_mult=text_encoder_ff_mult,
            attn_dropout=text_encoder_attn_dropout,
            ff_dropout=text_encoder_ff_dropout,
            attn_processor=text_encoder_attn_processor,
            norm_type=text_encoder_norm_type,
            ff_type=text_encoder_ff_type,
            ff_kernel_size=text_encoder_ff_kernel_size,
            ff_groups=text_encoder_ff_groups,
            scale_type=text_encoder_scale_type,
        )

        self.spk_encoder = SpeakerEncoder(
            dim_input=speaker_encoder_dim_input,
            conv_pos_kernel_size=speaker_encoder_conv_pos_kernel_size,
            conv_pos_depth=speaker_encoder_conv_pos_depth,
            conv_pos_groups=speaker_encoder_conv_pos_groups,
            depth=speaker_encoder_depth,
            dim=speaker_encoder_dim,
            dim_head=speaker_encoder_dim_head,
            heads=speaker_encoder_heads,
            ff_mult=speaker_encoder_ff_mult,
            attn_dropout=speaker_encoder_attn_dropout,
            ff_dropout=speaker_encoder_ff_dropout,
            attn_processor=speaker_encoder_attn_processor,
            norm_type=speaker_encoder_norm_type,
            ff_type=speaker_encoder_ff_type,
            ff_kernel_size=speaker_encoder_ff_kernel_size,
            ff_groups=speaker_encoder_ff_groups,
            scale_type=speaker_encoder_scale_type,
        )

        self.flow_matching_decoder = FlowMatchingTransformer(
            dim_input=feature_dim,
            dim_ctx=feature_dim,
            dim_output=feature_dim,
            dim_time=flow_matching_dim_time,
            conv_pos_kernel_size=flow_matching_conv_pos_kernel_size,
            conv_pos_depth=flow_matching_conv_pos_depth,
            conv_pos_groups=flow_matching_conv_pos_groups,
            depth=flow_matching_depth,
            dim=flow_matching_dim,
            dim_head=flow_matching_dim_head,
            heads=flow_matching_heads,
            ff_mult=flow_matching_ff_mult,
            attn_dropout=flow_matching_attn_dropout,
            ff_dropout=flow_matching_ff_dropout,
            attn_processor=flow_matching_attn_processor,
            norm_type=flow_matching_norm_type,
            ff_type=flow_matching_ff_type,
            ff_kernel_size=flow_matching_ff_kernel_size,
            ff_groups=flow_matching_ff_groups,
            scale_type=flow_matching_scale_type,
        )

        self.duration_predictor = DurationPredictor(
            dim_input=text_encoder_embed_dim,
            dim=duration_predictor_dim,
            depth=duration_predictor_depth,
            kernel_size=duration_predictor_kernel_size,
            dropout=duration_predictor_dropout,
        )

        self.reset_parameters()

        self.p_uncond = p_uncond
        self.interpolate_mode = interpolate_mode
        self.sigma = sigma

    def reset_parameters(self):
        def default_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

        self.apply(default_init)

        # init conv pos
        self.text_encoder.reset_parameters()
        self.spk_encoder.reset_parameters()
        self.flow_matching_decoder.reset_parameters()

    def length_to_attn_mask(self, lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lens is None:
            return None
        attn_mask = torch.arange(lens.max()).to(lens.device) < lens.unsqueeze(-1)
        return attn_mask

    def length_regulator(self, embs, emb_lens, durations, duration_lens):
        # can we do it faster? unpad then expand then pad? https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
        max_len = 0
        expanded = []
        for emb, token_len, dur, dur_len in zip(embs, emb_lens, durations, duration_lens):
            emb = emb[:token_len, :]
            dur = dur[:dur_len]
            expanded.append(torch.repeat_interleave(emb, dur, dim=0).unsqueeze(0))
            max_len = max(max_len, expanded[-1].shape[-2])
        return torch.cat([F.pad(e, (0, 0, 0, max_len - e.shape[-2]), value=0) for e in expanded], dim=0)

    def duration_loss(self, duration_pred, durations, duration_lens):
        mask = self.length_to_attn_mask(duration_lens)
        pred = duration_pred[mask]
        target = durations[mask]
        log_target = torch.log1p(target)
        loss = F.mse_loss(pred, log_target)
        return loss

    def enc_loss(self, h, latents, prompt_masks):
        assert h.shape == latents.shape, f"Shape mismatch: {h.shape} != {latents.shape}"
        pred = h[prompt_masks]
        target = latents[prompt_masks]
        loss = F.mse_loss(pred, target)
        return loss

    @staticmethod
    def interpolate(h: torch.Tensor, latent: torch.Tensor, mode: str = "linear") -> torch.Tensor:
        assert mode in ["linear", "nearest"], f"Interpolation mode {mode} is not supported"
        latent_len = latent.shape[-2]
        h_len = h.shape[-2]
        if latent_len == h_len:
            return h
        h = rearrange(h, "b t c -> b c t")
        h = F.interpolate(h, size=latent_len, mode=mode)
        h = rearrange(h, "b c t -> b t c")
        return h

    def forward(
        self, text_tokens, text_token_lens, durations, duration_lens, latents, latent_lens, prompts, prompt_masks
    ):
        # text encoder, speaker encoder
        spk_emb = self.spk_encoder(prompts)
        text_padding_mask = self.length_to_attn_mask(text_token_lens)
        h, text_emb = self.text_encoder(text_tokens=text_tokens, spk_emb=spk_emb, padding_mask=text_padding_mask)

        # duration predictor
        duration_pred = self.duration_predictor(text_emb.detach(), text_padding_mask)
        duration_loss = self.duration_loss(duration_pred, durations, duration_lens)

        # encoder loss
        h = self.length_regulator(h, text_token_lens, durations, duration_lens)
        h = self.interpolate(h, latents, mode=self.interpolate_mode)
        enc_loss = self.enc_loss(h, latents, prompt_masks)

        # flow matching
        times = torch.rand((h.shape[0],)).to(h.device)
        times = rearrange(times, "b -> b 1 1")
        x0 = torch.randn_like(latents)
        xt = (1 - (1 - self.sigma) * times) * x0 + times * latents
        flow = latents - (1 - self.sigma) * x0
        times = rearrange(times, "b 1 1 -> b")
        drop_cond = torch.rand((h.shape[0],)).to(h.device) < self.p_uncond
        x_ctx = h
        latent_padding_mask = self.length_to_attn_mask(latent_lens)

        vt = self.flow_matching_decoder(
            x=xt, x_ctx=x_ctx, times=times, padding_mask=latent_padding_mask, drop_ctx=drop_cond
        )
        flow_matching_loss = F.mse_loss(vt[prompt_masks], flow[prompt_masks])

        return duration_loss, enc_loss, flow_matching_loss

    @torch.no_grad()
    def generate(
        self, text_tokens, prompts, durations=None, nfe: int = 16, ode_method: str = "midpoint", cfg_scale: float = 0.0
    ):
        assert text_tokens.shape[0] == 1, "generation with batch size > 1 is not supported yet"
        spk_emb = self.spk_encoder(prompts)
        h, text_emb = self.text_encoder(text_tokens=text_tokens, spk_emb=spk_emb)

        if durations is None:
            duration_pred = self.duration_predictor(text_emb.detach())
            durations = torch.expm1(duration_pred).clamp(min=1).ceil().long()

        h = torch.repeat_interleave(h, durations.squeeze(), dim=1)

        def sample_fn(t, x_t):
            batch_size = x_t.shape[0]
            t = t.expand(batch_size)
            drop_cond = torch.zeros_like(t).bool()
            v = self.flow_matching_decoder(x_t, h, t, drop_ctx=drop_cond)
            if cfg_scale != 0:
                v_null = self.transformer(x_t, h, t, drop_ctx=~drop_cond)
                v = (1 + cfg_scale) * v - v_null

            return v

        times = torch.linspace(0, 1, nfe).to(h.device)
        x0 = torch.randn_like(h)
        traj = torchdiffeq.odeint(sample_fn, x0, times, atol=1e-4, rtol=1e-4, method=ode_method)
        return traj[-1]
