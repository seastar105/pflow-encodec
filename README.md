# PFlow Encodec

Implementation of TTS based on paper [P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting](https://openreview.net/pdf?id=zNA7u7wtIN). You can check main differences between implementation and paper in [Differences](#difference-from-paper) section.

# Main Goal

I have two goals to achieve in this project.

- First, I want to test character-based input with [SeamlessM4T](https://arxiv.org/abs/2308.11596)'s [Aligner](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/unity2_aligner_README.md) for English, Korean, Japanese and other languages. but, mainly for three languages mentioned above.
- Second, zero-shot multilingual TTS model. since this model will be trained with sentencepiece tokenizer input, it does not need phonemizer. so, it would be easily adapted to other languages.

# TODO

- [x] Implement baseline model.
- [ ] Train model on libritts-r.
- [ ] Add some optimization(flash-attention, fused kernels) for faster training.
- [ ] Simple gradio demo.
- [ ] Dataset preparation documentation.
- [ ] Train model on another language, i'm planning to train on Korean and Japanese.
- [ ] Multilingual model.

# Difference from paper

I did not conduct ablations for each changes due to lack of resources.

- Use [Encodec](https://github.com/facebookresearch/audiocraft/blob/main/docs/ENCODEC.md) latent instead of MelSpectrogram.
- Use character-base input instead of phoneme, and GT duration as a target of duration predictor instead of MAS.
- Use AdaLN-Zero from [DiT](https://arxiv.org/abs/2212.09748) for speaker-conditioned text encoder instead of concat and self-attention.
- Use transformer as Flow Matching decoder instead of Wavenet blocks with AdaLN-Single timestep conditioning from [PixArt-α](https://arxiv.org/abs/2310.00426)
- Use attention pooling instead of mean pooling to get fixed-size speaker embedding.
- Use conv-feedforward and GeGLU
- Use Alibi + Convolution positional encoding in transformer
- Use null cond for CFG sampling instead of mean-pooled hidden vectors.
- Upscale hidden vectors from text encoder to encodec latent after expanding text encoder's output

# Credits

- I borrowed some code from [VITS repo](https://github.com/jaywalnut310/vits), [voicebox-pytorch](https://github.com/lucidrains/voicebox-pytorch), and [fairseq2](https://github.com/facebookresearch/fairseq2).
