# PFlow Encodec

Implementation of TTS based on paper [P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting](https://openreview.net/pdf?id=zNA7u7wtIN). You can check main differences between implementation and paper in [Differences](#difference-from-paper) section.

# Main Goal

I have two goals to achieve in this project.

- First, I want to test character-based input with [SeamlessM4T](https://arxiv.org/abs/2308.11596)'s [Aligner](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/unity2_aligner_README.md) for English, Korean, Japanese and other languages. but, mainly for three languages mentioned above.
- Second, zero-shot multilingual TTS model. since this model will be trained with sentencepiece tokenizer input, it does not need phonemizer. so, it would be easily adapted to other languages.

# Samples

Generated Samples from model trained on LibriTTS-R are at [samples](https://github.com/seastar105/pflow-encodec/tree/main/samples) folder. All samples are decoded with MultiBand-Diffusion model from [AudioCraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MBD.md)Pretrained checkpoint is available [here](https://huggingface.co/seastar105/pflow-encodec-libritts/tree/main). you can check how to use it in [sample notebook](https://github.com/seastar105/pflow-encodec/blob/main/notebooks/generate.ipynb).

Text: P-Flow encodec is Text-to-Speech model trained on Encodec latent space, using Flow Matching.

Prompt Audio

https://github.com/seastar105/pflow-encodec/assets/30820469/7ef45589-4f08-478e-a8be-6bd6f30a7f1d

Generated Audio

https://github.com/seastar105/pflow-encodec/assets/30820469/8289bffe-f967-4af5-91c7-3fcb754822fb

# Pre-trained models

- English model trained for LibriTTS-R, about 265K steps. [model link](https://huggingface.co/seastar105/pflow-encodec-libritts/tree/main), [config link](https://github.com/seastar105/pflow-encodec/blob/main/configs/experiment/libritts_base.yaml), [tensorboard screenshot](https://github.com/seastar105/pflow-encodec/blob/main/screenshots/pflow_libri_tb.png)

# TODO

- [x] Implement baseline model.
- [x] Train model on libritts-r.
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
