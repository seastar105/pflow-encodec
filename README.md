# PFlow Encodec

Implementation of TTS based on paper [P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting](https://openreview.net/pdf?id=zNA7u7wtIN). You can check main differences between implementation and paper in [Differences](#difference-from-paper) section.

# Main Goal

I have two goals to achieve in this project.

- First, I want to test character-based input with [SeamlessM4T](https://arxiv.org/abs/2308.11596)'s [Aligner](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/unity2_aligner_README.md) for English, Korean, Japanese and other languages. but, mainly for three languages mentioned above.
- Second, zero-shot multilingual TTS model. since this model will be trained with sentencepiece tokenizer input, it does not need phonemizer. so, it would be easily adapted to other languages.

# TODO

- [x] Implement baseline model.
- [ ] Train model on libritts-r.
- [ ] Simple gradio demo.
- [ ] Dataset preparation documentation.
- [ ] Train model on another language, i'm planning to train on Korean and Japanese.
- [ ] Multilingual model.

# Difference from paper

- Use [Encodec](https://github.com/facebookresearch/audiocraft/blob/main/docs/ENCODEC.md) latent instead of MelSpectrogram.
- Use character-base input instead of phoneme, and GT duration as a target of duration predictor instead of MAS.
- Use AdaLN-Single from [PixArt-α](https://arxiv.org/abs/2310.00426) for speaker-conditioned text encoder instead of concat and self-attention.
- Use transformer as Flow Matching decoder instead of Wavenet blocks.

# Credits

- I borrowed some code from [VITS repo](https://github.com/jaywalnut310/vits), lucidrain's [voicebox](https://github.com/lucidrains/voicebox-pytorch) implementation, and [fairseq2](https://github.com/facebookresearch/fairseq2).
