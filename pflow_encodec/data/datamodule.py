import os
from typing import List

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from pflow_encodec.data.sampler import DistributedBucketSampler
from pflow_encodec.data.text_latent_dur_dataset import TextLatentDataset


class TextLatentLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_tsv_path: str,
        val_tsv_path: str,
        add_trailing_silence: bool = True,
        batch_durations: float = 50.0,
        boundaries: List[float] = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 60.0],
        num_workers: int = 4,
        return_upsampled: bool = False,
        max_frame: int = 1500,
        text2latent_rate: float = 1.5,
        mean: float = -0.5444963574409485,
        std: float = 5.242217063903809,
    ):
        super().__init__()
        self.train_tsv_path = train_tsv_path
        self.val_tsv_path = val_tsv_path

        self.train_ds: TextLatentDataset = None
        self.val_ds: TextLatentDataset = None

        self.add_trailing_silence = add_trailing_silence
        self.pad_idx = None

        self.batch_durations = batch_durations
        self.boundaries = boundaries
        self.num_workers = num_workers
        self.return_upsampled = return_upsampled

        self.max_frame = max_frame
        self.text2latent_rate = text2latent_rate

        self.mean = mean
        self.std = std

    def setup(self, stage: str):
        if stage != "fit":
            raise ValueError(f"Stage {stage} is not supported")
        self.train_ds = TextLatentDataset(
            self.train_tsv_path, add_trailing_silence=self.add_trailing_silence, mean=self.mean, std=self.std
        )
        self.val_ds = TextLatentDataset(
            self.val_tsv_path, add_trailing_silence=self.add_trailing_silence, mean=self.mean, std=self.std
        )

        self.pad_idx = self.train_ds.tokenizer.pad_idx

    def prepare_data(self):
        if not os.path.exists(self.train_tsv_path):
            raise FileNotFoundError(f"File {self.train_tsv_path} does not exist")
        if not os.path.exists(self.val_tsv_path):
            raise FileNotFoundError(f"File {self.val_tsv_path} does not exist")

    def _collate(self, batch):
        text_tokens, durations, latents = map(list, zip(*batch))
        if self.return_upsampled:
            # used for training AudioModel
            for t, d in zip(text_tokens, durations):
                if t.shape[-1] != d.shape[-1]:
                    raise ValueError(f"Text token and duration shape mismatch: {t.shape} != {d.shape}")
            text_tokens = [torch.repeat_interleave(t, d.squeeze(), dim=1) for t, d in zip(text_tokens, durations)]

            # truncate if there's sample over max_frame
            for i in range(t.shape[0]):
                seq_len = latents[i].shape[-2]
                if seq_len <= self.max_frame:
                    continue
                start_idx = np.random.randint(0, seq_len - self.max_frame)
                latent = latents[i][:, start_idx : start_idx + self.max_frame, :]

                text_start_idx = max(0, int(start_idx / self.text2latent_rate))
                text_frame_len = int(self.max_frame / self.text2latent_rate)
                text_token = text_tokens[i][:, text_start_idx : text_start_idx + text_frame_len]

                latents[i] = latent
                text_tokens[i] = text_token

            max_text_len = max(t.shape[-1] for t in text_tokens)
            text_token_lens = torch.tensor([t.shape[-1] for t in text_tokens])
            text_tokens = torch.cat(
                [torch.nn.functional.pad(t, (0, max_text_len - t.shape[-1]), value=self.pad_idx) for t in text_tokens],
                dim=0,
            )

            max_latent_len = max(latent.shape[-2] for latent in latents)
            latent_lens = torch.tensor([latent.shape[-2] for latent in latents])
            latents = torch.cat(
                [
                    torch.nn.functional.pad(latent, (0, 0, 0, max_latent_len - latent.shape[-2]), value=0)
                    for latent in latents
                ],
                dim=0,
            )
            return text_tokens, text_token_lens, latents, latent_lens
        else:
            # used for training AudioModel
            for t, d in zip(text_tokens, durations):
                if t.shape[-1] != d.shape[-1]:
                    raise ValueError(f"Text token and duration shape mismatch: {t.shape} != {d.shape}")
            max_text_len = max(t.shape[-1] for t in text_tokens)
            text_token_lens = torch.tensor([t.shape[-1] for t in text_tokens])
            text_tokens = torch.cat(
                [torch.nn.functional.pad(t, (0, max_text_len - t.shape[-1]), value=self.pad_idx) for t in text_tokens],
                dim=0,
            )

            max_duration_len = max(d.shape[-1] for d in durations)
            duration_lens = torch.tensor([d.shape[-1] for d in durations])
            durations = torch.cat(
                [torch.nn.functional.pad(d, (0, max_duration_len - d.shape[-1]), value=0) for d in durations],
                dim=0,
            )

            max_latent_len = max(latent.shape[-2] for latent in latents)
            latent_lens = torch.tensor([latent.shape[-2] for latent in latents])
            latents = torch.cat(
                [
                    torch.nn.functional.pad(latent, (0, 0, 0, max_latent_len - latent.shape[-2]), value=0)
                    for latent in latents
                ],
                dim=0,
            )
            return text_tokens, text_token_lens, durations, duration_lens, latents, latent_lens

    def train_dataloader(self):
        world_size = 1 if not torch.distributed.is_initialized() else None
        rank = 0 if not torch.distributed.is_initialized() else None
        sampler = DistributedBucketSampler(
            self.train_ds,
            batch_durations=self.batch_durations,
            boundaries=self.boundaries,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
        )
        return DataLoader(
            self.train_ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        world_size = 1 if not torch.distributed.is_initialized() else None
        rank = 0 if not torch.distributed.is_initialized() else None
        sampler = DistributedBucketSampler(
            self.val_ds,
            batch_durations=self.batch_durations,
            boundaries=self.boundaries,
            num_replicas=world_size,
            rank=rank,
            drop_last=False,
            shuffle=True,
        )
        return DataLoader(
            self.val_ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            shuffle=False,
        )
