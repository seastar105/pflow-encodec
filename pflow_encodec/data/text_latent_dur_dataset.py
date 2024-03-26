import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pflow_encodec.data.tokenizer import TextTokenizer


class TextLatentDataset(Dataset):
    """Dataset for Voicebox Training, returns text tokens, duration, and pre-quantize encodec latent.

    text_tokens: torch.Tensor, shape (1, T_text)
    duration: torch.Tensor, shape (1, T_text)
    latent: torch.Tensor, shape (1, T_latent, D_latent)
    """

    def __init__(
        self,
        tsv_path: str,
        add_trailing_silence: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        min_duration: float = 3.0,
        max_duration: float = 15.0,
    ):
        df = pd.read_csv(tsv_path, sep="\t", engine="pyarrow")
        df = df[df["duration"] >= min_duration]
        df = df[df["duration"] <= max_duration]

        self.paths = df["audio_path"].tolist()
        self.texts = df["text"].tolist()

        self.audio_durations = df["duration"].tolist()

        self.tokenizer = TextTokenizer(add_trailing_silence=add_trailing_silence)

        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        text = self.texts[idx]
        text_tokens = self.tokenizer.encode_text(text).squeeze()

        latent_npy_path = Path(path).with_suffix(".latent.npy")
        duration_npy_path = Path(path).with_suffix(".duration.npy")

        latent = torch.from_numpy(np.load(latent_npy_path)).squeeze().unsqueeze(0)
        duration = torch.from_numpy(np.load(duration_npy_path)).squeeze().unsqueeze(0)

        # text_tokens = torch.repeat_interleave(text_tokens, duration, dim=0)
        if text_tokens.ndim == 1:
            text_tokens = text_tokens.unsqueeze(0)

        if text_tokens.shape[-1] != duration.shape[-1]:
            raise ValueError(
                f"Text token and duration shape mismatch: {text_tokens.shape} != {duration.shape}, path={path}, text={text}"
            )

        latent = (latent - self.mean) / self.std

        return text_tokens, duration, latent
