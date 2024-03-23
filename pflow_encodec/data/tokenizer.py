import torch
from audiotools import AudioSignal
from seamless_communication.models.unity.char_tokenizer import load_unity_char_tokenizer
from transformers import EncodecModel


class EncodecTokenizer:
    def __init__(
        self, model_name: str = "facebook/encodec_24khz", device: str = "cuda", dtype: torch.dtype = torch.float32
    ):
        model = EncodecModel.from_pretrained(model_name)

        self.device = torch.device(device)
        self.dtype = dtype
        self.codec: EncodecModel = model.to(self.device, dtype=self.dtype).eval()

        self.sample_rate = self.codec.config.sampling_rate

    def load_audio(self, path: str) -> torch.Tensor:
        """Load audio file and transform it to the correct format for the model.

        Args:
            path (str): audio file path
        Returns:
            audio (torch.Tensor): audio tensor of shape (1, 1, T)
        """
        signal = AudioSignal(path)
        if signal.sample_rate != self.sample_rate:
            signal = signal.resample(self.sample_rate)
        if signal.num_channels > 1:
            signal = signal.to_mono()
        return signal.audio_data.to(device=self.device, dtype=self.dtype)

    def encode_audio(self, audio: torch.Tensor, return_code: bool = False) -> torch.Tensor:
        """Encode audio to latent space, return discrete tokens if return_latent is False.

        Args:
            audio (torch.Tensor): audio tensor of shape (1, 1, T)
            return_latent (bool, optional): return discrete tokens if False, return continuous latent before quantization if True.

        Returns:
            torch.Tensor: encoded tokens or latent
        """
        latents = self.codec.encoder(audio).transpose(-2, -1)
        if return_code:
            return self.codec.quantizer.encode(latents.transpose(-2, -1)).transpose(0, 1)
        return latents

    def encode_file(self, path: str, return_code: bool = False) -> torch.Tensor:
        audio = self.load_audio(path)
        return self.encode_audio(audio, return_code)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens to audio.

        Args:
            codes (torch.Tensor): discrete tokens of shape (1, Q, T)

        Returns:
            torch.Tensor: audio tensor of shape (1, 1, T)
        """
        return self.codec.decode(codes[None], [None])[0]

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode continuous latent to audio.

        Args:
            latents (torch.Tensor): continuous latent of shape (1, T, D)

        Returns:
            torch.Tensor: audio tensor of shape (1, 1, T)
        """
        codes = self.codec.quantizer.encode(latents.transpose(-2, -1)).transpose(0, 1)
        return self.decode_codes(codes)

    def quantize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Quantize continuous latent to discrete tokens.

        Args:
            latents (torch.Tensor): continuous latent of shape (1, T, D)

        Returns:
            torch.Tensor: discrete tokens of shape (1, Q, T)
        """
        return self.codec.quantizer.encode(latents.transpose(-2, -1)).transpose(0, 1)


class TextTokenizer:
    def __init__(self, add_trailing_silence: bool = True) -> None:
        text_tokenizer = load_unity_char_tokenizer("nar_t2u_aligner")
        self.tokenizer = text_tokenizer.create_raw_encoder()
        self.vocab_info = text_tokenizer.vocab_info

        self.bos_idx = self.vocab_info.bos_idx
        self.eos_idx = self.vocab_info.eos_idx
        self.pad_idx = self.vocab_info.pad_idx

        self.add_trailing_silence = add_trailing_silence

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to discrete tokens.

        Args:
            text (str): input text

        Returns:
            torch.Tensor: discrete tokens
        """
        tokens = self.tokenizer(text)
        if self.add_trailing_silence:
            tokens = torch.cat([tokens, tokens[0:1]])
        return tokens
