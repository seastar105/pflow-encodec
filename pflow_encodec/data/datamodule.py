import os
from typing import List, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader

from pflow_encodec.data.sampler import DistributedBucketSampler
from pflow_encodec.data.text_latent_dur_dataset import (
    TextLatentDataset,
    TextLatentLangDataset,
)


class TextLatentLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_tsv_path: str,
        val_tsv_path: str,
        add_trailing_silence: bool = True,
        batch_durations: float = 50.0,
        min_duration: float = 3.0,
        max_duration: float = 15.0,
        boundaries: List[float] = [3.0, 5.0, 7.0, 10.0, 15.0],
        num_workers: int = 4,
        return_upsampled: bool = False,
        max_frame: int = 1500,
        text2latent_rate: float = 1.5,
        mean: float = -0.5444963574409485,
        std: float = 5.242217063903809,
        use_lang_id: bool = False,
        languages: Optional[List[str]] = None,
    ):
        super().__init__()
        self.train_tsv_path = train_tsv_path
        self.val_tsv_path = val_tsv_path

        self.train_ds: TextLatentDataset = None
        self.val_ds: TextLatentDataset = None

        self.add_trailing_silence = add_trailing_silence
        self.pad_idx = None

        self.batch_durations = batch_durations
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.boundaries = boundaries
        self.num_workers = num_workers
        self.return_upsampled = return_upsampled
        # do not use return_upsampled
        assert not self.return_upsampled, "return_upsampled is not supported"

        self.max_frame = max_frame
        self.text2latent_rate = text2latent_rate

        self.mean = mean
        self.std = std
        self.use_lang_id = use_lang_id
        if languages is not None:
            self.languages = languages
            self.lang2idx = {lang: idx for idx, lang in enumerate(languages)}

    def setup(self, stage: str):
        if stage != "fit":
            raise ValueError(f"Stage {stage} is not supported")
        dataset_cls = TextLatentLangDataset if self.use_lang_id else TextLatentDataset
        self.train_ds = dataset_cls(
            self.train_tsv_path,
            add_trailing_silence=self.add_trailing_silence,
            mean=self.mean,
            std=self.std,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
        )
        self.val_ds = dataset_cls(
            self.val_tsv_path,
            add_trailing_silence=self.add_trailing_silence,
            mean=self.mean,
            std=self.std,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
        )

        self.pad_idx = self.train_ds.tokenizer.pad_idx

    def prepare_data(self):
        if not os.path.exists(self.train_tsv_path):
            raise FileNotFoundError(f"File {self.train_tsv_path} does not exist")
        if not os.path.exists(self.val_tsv_path):
            raise FileNotFoundError(f"File {self.val_tsv_path} does not exist")

    def _collate(self, batch):
        result = {}
        if self.use_lang_id:
            text_tokens, durations, latents, languages = map(list, zip(*batch))
            lang_ids = torch.stack([torch.tensor(self.lang2idx[lang]) for lang in languages])
            result["lang_ids"] = lang_ids
        else:
            text_tokens, durations, latents = map(list, zip(*batch))
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
        result["text_tokens"] = text_tokens
        result["text_token_lens"] = text_token_lens
        result["durations"] = durations
        result["duration_lens"] = duration_lens
        result["latents"] = latents
        result["latent_lens"] = latent_lens
        return result

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
