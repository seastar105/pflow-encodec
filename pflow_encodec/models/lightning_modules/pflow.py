from typing import List, Optional

import hydra
import lightning as L
import torch
from audiotools import AudioSignal
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig

from pflow_encodec.data.tokenizer import EncodecTokenizer
from pflow_encodec.models.pflow import PFlow
from pflow_encodec.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class PFlowLightningModule(L.LightningModule):
    def __init__(
        self,
        net: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        prompt_length: int = 225,
        sample_freq: int = 10000,
        sample_idx: List[int] = [],
        mean: float = 0.0,
        std: float = 1.0,
        text2latent_ratio: float = 1.5,
        net_ckpt_path: Optional[str] = None,
        languages: Optional[List[str]] = None,
        lang_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.save_hyperparameters()

        self.net: PFlow = hydra.utils.instantiate(net, _recursive_=False)

        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

        self.prompt_length = prompt_length

        self.sample_freq = sample_freq
        self.sample_idx = sample_idx

        self.first_sample = True
        self.codec = [EncodecTokenizer(device="cpu")]  # avoid move codec to gpu for memory reduction

        self.mean = mean
        self.std = std
        self.text2latent_ratio = text2latent_ratio

        if languages is not None:
            self.languages = languages
            self.lang2idx = {lang: idx for idx, lang in enumerate(languages)}
            self.lang_loss_weight = lang_loss_weight

        if net_ckpt_path is not None:
            logger.info(f"Loading model from {net_ckpt_path}")
            missing, unexpected = self.net.load_state_dict(
                torch.load(net_ckpt_path, map_location="cpu")["state_dict"], strict=False
            )
            if missing:
                logger.warning(f"Missing keys: {missing}")

            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected}")

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.net.parameters())
        if self.scheduler_cfg is None:
            return [optimizer], []
        scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def on_before_optimizer_step(self, optimizer):
        self.log(
            "other/grad_norm",
            grad_norm(self.net, norm_type=2)["grad_2.0_norm_total"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

    def get_prompt(self, latents: torch.Tensor, latent_lens: torch.Tensor):
        b, t, d = latents.shape
        max_start = latent_lens - self.prompt_length
        start_idx = (torch.rand((b,), device=latents.device) * max_start).long().clamp(min=0)
        prompts = torch.zeros((b, self.prompt_length, d), device=latents.device, dtype=latents.dtype)
        for i in range(latents.shape[0]):
            prompts[i] = latents[i, start_idx[i] : start_idx[i] + self.prompt_length]

        max_len = latent_lens.max()
        prompt_mask = torch.arange(max_len, device=latent_lens.device).expand(latent_lens.shape[0], -1) < (
            start_idx.unsqueeze(1) + self.prompt_length
        )
        prompt_mask &= torch.arange(max_len, device=latent_lens.device).expand(
            latent_lens.shape[0], -1
        ) >= start_idx.unsqueeze(1)
        return prompts, prompt_mask

    def get_input(self, batch):
        text_tokens = batch["text_tokens"]
        text_token_lens = batch["text_token_lens"]
        durations = batch["durations"]
        duration_lens = batch["duration_lens"]
        latents = batch["latents"]
        latent_lens = batch["latent_lens"]
        prompts, prompt_masks = self.get_prompt(latents, latent_lens)
        lang_ids = None
        if "lang_ids" in batch:
            lang_ids = batch["lang_ids"]
        return (
            text_tokens,
            text_token_lens,
            durations,
            duration_lens,
            latents,
            latent_lens,
            prompts,
            prompt_masks,
            lang_ids,
        )

    def training_step(self, batch, batch_idx):
        (
            text_tokens,
            text_token_lens,
            durations,
            duration_lens,
            latents,
            latent_lens,
            prompts,
            prompt_masks,
            lang_ids,
        ) = self.get_input(batch)
        duration_loss, enc_loss, flow_matching_loss, lang_loss = self.net(
            text_tokens,
            text_token_lens,
            durations,
            duration_lens,
            latents,
            latent_lens,
            prompts,
            prompt_masks,
            lang_ids=lang_ids,
        )

        self.log("train/enc_loss", enc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/duration_loss", duration_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "train/flow_matching_loss", flow_matching_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train/latent_loss", enc_loss + flow_matching_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        loss = enc_loss + duration_loss + flow_matching_loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if lang_loss is not None:
            self.log("train/lang_loss", lang_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss = loss + lang_loss * self.lang_loss_weight

        if self.global_step % self.sample_freq == 0:
            self.log_audio()

        return loss

    def validation_step(self, batch, batch_idx):
        (
            text_tokens,
            text_token_lens,
            durations,
            duration_lens,
            latents,
            latent_lens,
            prompts,
            prompt_masks,
            lang_ids,
        ) = self.get_input(batch)
        duration_loss, enc_loss, flow_matching_loss, lang_loss = self.net(
            text_tokens,
            text_token_lens,
            durations,
            duration_lens,
            latents,
            latent_lens,
            prompts,
            prompt_masks,
            lang_ids=lang_ids,
        )

        self.log("val/enc_loss", enc_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/duration_loss", duration_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/flow_matching_loss", flow_matching_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val/latent_loss", enc_loss + flow_matching_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        loss = enc_loss + duration_loss + flow_matching_loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if lang_loss is not None:
            self.log("val/lang_loss", lang_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            loss = loss + lang_loss * self.lang_loss_weight

        return loss

    @torch.inference_mode()
    def log_audio(self):
        if self.global_rank != 0:
            return

        self.net.eval()
        codec = self.codec[0]
        writer = self.logger.experiment

        def write_to_tb(codec_latent: torch.Tensor, name: str):
            # denormalize
            codec_latent = codec_latent * self.std + self.mean
            with torch.amp.autocast(device_type="cuda", enabled=False):
                recon = codec.decode_latents(codec_latent.to(device=codec.device, dtype=codec.dtype))
            signal = AudioSignal(recon, sample_rate=codec.sample_rate).float().ensure_max_of_audio()
            signal.write_audio_to_tb(name, writer, self.global_step)

        if self.first_sample:
            self.first_sample = False
            for idx, sample_idx in enumerate(self.sample_idx):
                sample = self.trainer.datamodule.val_ds[sample_idx]
                if len(sample) == 3:
                    _, _, latent = sample
                else:
                    _, _, latent, _ = sample
                write_to_tb(latent, f"recon/sample_{idx}.wav")

        # sample with gt duration
        for idx, sample_idx in enumerate(self.sample_idx):
            sample = self.trainer.datamodule.val_ds[sample_idx]
            if len(sample) == 3:
                text_token, duration, latent = sample
                lang_id = None
            else:
                text_token, duration, latent, lang = sample
                lang_id = torch.tensor([self.lang2idx[lang]], device=self.device)
            start_idx = torch.randint(0, latent.shape[-2] - self.prompt_length, (1,))
            prompt = latent[:, start_idx : start_idx + self.prompt_length]
            sampled = self.net.generate(
                text_token.to(self.device),
                prompt.to(self.device),
                duration.to(self.device),
                upscale_ratio=self.text2latent_ratio,
                lang_ids=lang_id,
            )
            write_to_tb(sampled, f"sampled/gt_dur_{idx}.wav")

        # sample with pred duration
        for idx, sample_idx in enumerate(self.sample_idx):
            sample = self.trainer.datamodule.val_ds[sample_idx]
            if len(sample) == 3:
                text_token, duration, latent = sample
                lang_id = None
            else:
                text_token, duration, latent, lang = sample
                lang_id = torch.tensor([self.lang2idx[lang]], device=self.device)
            start_idx = torch.randint(0, latent.shape[-2] - self.prompt_length, (1,))
            prompt = latent[:, start_idx : start_idx + self.prompt_length]
            sampled = self.net.generate(
                text_token.to(self.device),
                prompt.to(self.device),
                upscale_ratio=self.text2latent_ratio,
                lang_ids=lang_id,
            )
            write_to_tb(sampled, f"sampled/pred_dur_{idx}.wav")

        self.net.train()
