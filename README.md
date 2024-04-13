# PFlow Encodec

Implementation of TTS based on paper [P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting](https://openreview.net/pdf?id=zNA7u7wtIN). You can check main differences between implementation and paper in [Differences](#difference-from-paper) section.

# Main goal of this project

I have two goals to achieve in this project. It seems work but, really poor at Japanese and numbers. 

- First, I want to test character-based input with [SeamlessM4T](https://arxiv.org/abs/2308.11596)'s [Aligner](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/unity2_aligner_README.md) for English, Korean, Japanese and other languages. but, mainly for three languages mentioned above.
- Second, zero-shot multilingual TTS model. since this model will be trained with sentencepiece tokenizer input, it does not need phonemizer. so, it would be easily adapted to other languages tokenizer supports. check out supported languages of tokenizer [here](https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/cards/nar_t2u_aligner.yaml)

# Samples

Generated Samples from model trained on LibriTTS-R, korean and japanese corpus of AIHub 131 datasets. All samples are decoded with MultiBand-Diffusion model from [AudioCraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MBD.md).
Pretrained checkpoint used here is available on [huggingface](https://huggingface.co/seastar105/pflow-encodec-ejk).

you can check how to use it in [sample notebook](https://github.com/seastar105/pflow-encodec/blob/main/notebooks/generate.ipynb).

Currently, speaker embedding of multi-lingual model seems to be highly entangled with language info. it shows worse zero-shot capability. I'm planning to train new model with language ID to reduce language bias in speaker embedding.

Code-switch Text: There's famous japanese sentence, つきがきれいですね, which means 나는 당신을 사랑합니다.

English Prompt Generation

https://github.com/seastar105/pflow-encodec/assets/30820469/57a0450b-e1b2-48b6-b0ec-9433806edb10

Japanese Prompt Generation

https://github.com/seastar105/pflow-encodec/assets/30820469/bf5e4c29-2545-411a-adbc-b461a5c2cefa

Korean Prompt Generation

https://github.com/seastar105/pflow-encodec/assets/30820469/74f2ff7a-554d-4797-9841-a8b7b74d9fbf

English Text: P-Flow encodec is Text-to-Speech model trained on Encodec latent space, using Flow Matching.

Prompt Audio (from [LibriTTS-R](https://www.openslr.org/141/))

https://github.com/seastar105/pflow-encodec/assets/30820469/a3c1b3d8-ea94-4cb7-bd21-7226e3fd54b1

Generated Audio

https://github.com/seastar105/pflow-encodec/assets/30820469/1de00f81-4c87-402e-a4bc-66deb29c194d

Japanese Text: こんにちは、初めまして。あなたの名前はなんですか？これは音声合成モデルから作られた音声です。

Prompt Audio (from [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut))

https://github.com/seastar105/pflow-encodec/assets/30820469/fb4f1a10-fb8b-413e-8bec-d1d0f58d8423

Generated Audio

https://github.com/seastar105/pflow-encodec/assets/30820469/137d4e34-f674-4681-a652-93c4a44f4554

Korean Text: 백남준은 미디어 아트의 개척자로서 다양한 테크놀로지를 이용하여 실험적이고 창의적으로 작업했다.

Prompt Audio (from [KSS](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset))

https://github.com/seastar105/pflow-encodec/assets/30820469/db3435d0-8e8f-45ef-b3b3-a164ad316d71

Generated Audio

https://github.com/seastar105/pflow-encodec/assets/30820469/8dff38ec-a2d7-49a6-80fb-de6012b33a1b

# Environment Setup

I've developed in WSL, Windows 11. I have not tested on other platforms and torch version. I recommend using conda environment.

```bash
conda create -n pflow-encodec -y python=3.10
conda activate pflow-encodec
conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y -c conda-forge libsndfile==1.0.31
pip install -r requirements.txt
pip install -r infer-requirements.txt
```

# Dataset Preparation

## meta tsv file

First of all, you need to prepare tsv file, which contains three columns: `audio_path`, `text`, `duration`. each column is separated by tab.

`audio_path` is path to audio file, `text` is transcript of audio file, and `duration` is duration of audio file in seconds.

### Example

```tsv
audio_path	text	duration
/path/to/audio1.wav	Hello, World!	1.5
/path/to/audio2.wav 안녕하세요, 세계!	2.0
/path/to/audio3.wav こんにちは、世界！	2.5
```

## Dump encodec latent and sentencepiece token durations

Here, use encodec latent as output, and duration per token as target of duration predictor.

you can dump encodec latent and sentencepiece token durations with following command.

```bash
python scripts/dump_durations.py --input_tsv <meta_tsv_file>
python scripts/dump_latents.py --input_tsv <meta_tsv_file>
```

this command requires GPU and `scripts/dump_durations.py` may require more than 8GB of GPU memory.

`scripts/dump_durations.py` takes about 6 hours for 1000 hours of audio files. `scripts/dump_latents.py` takes about 4 hours for 1000 hours of audio files. both time was measured on RTX 4090.

each script will make two files per audio file:
`<audio_path stem>.latent.npy` and `<audio_path stem>.duration.npy`.

**NOTE: `scripts/dump_latents.py` will print out global mean and std of dataset's latent. You should keep it since this value is used for training model.**

Now, you can start training.

# Train Model

Repository's code is based on [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

After preparing dataset, you can start training after setting dataset config and experiment config. Let your dataset name be `new_dataset`. first you need to set dataset config in `configs/data/new_dataset.yaml`.

```yaml
_target_: pflow_encodec.data.datamodule.TextLatentLightningDataModule

train_tsv_path: <train_tsv_path>
val_tsv_path: <val_tsv_path>
add_trailing_silence: True
batch_durations: 50.0   # mini-batch duration in seconds
min_duration: 3.5    # minimum duration of files, this value MUST be bigger than 3.0
max_duration: 15.0
boundaries: [3.0, 5.0, 7.0, 10.0, 15.0]
num_workers: 8
return_upsampled: False
max_frame: 1500 # 20s
text2latent_rate: 1.5 # 50Hz:75Hz
mean: <mean>
std: <std>
```

fill `<train_tsv_path>`, `<val_tsv_path>`, `<mean>`, and `<std>` with your dataset's meta path and mean/std values.

then, create config in `configs/experiment/new_dataset.yaml`.

```yaml
# @package _global_

defaults:
  - override /data: new_dataset.yaml # your dataset config name here!!!
  - override /model: pflow_base.yaml
  - override /callbacks: default.yaml
  - override /trainer: gpu.yaml
  - override /logger: tensorboard.yaml

task_name: pflow
tags: ["pflow"]
seed: 998244353
test: False

callbacks:
  val_checkpoint:
    filename: "val_latent_loss_{val/latent_loss:.4f}-{step:06d}"
    monitor: val/latent_loss
    mode: "min"
model:
  scheduler:
    total_steps: ${trainer.max_steps}
    pct_start: 0.02
  sample_freq: 5000
  mean: ${data.mean}
  std: ${data.std}
trainer:
  max_steps: 500000
  max_epochs: 10000 # arbitrary large number
  precision: bf16-mixed # you should check if your GPU supports bf16
  accumulate_grad_batches: 4    # effective batch size
  gradient_clip_val: 0.2
  num_nodes: 1
  devices: 1
hydra:
  run:
    dir: <fill experiment result path>
```

now you can run training with following command.

```bash
python pflow_encodec/train.py experiment=new_dataset
```

**NOTE: If you want to train model with multiple GPUs, you should adjust trainer.num_nodes and trainer.devices in experiment config. Also you should set trainer.use_distributed_sampler to be False. For more detailed information, check out Pytorch Lightning's documents.**

Example of single node 4 gpus

```
trainer:
  num_nodes: 1
  devices: 4
  use_distributed_sampler: False
```

# Pre-trained models

| Language          | Weights                                                                       | Model Card                                                                  |
| ----------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| MultiLingual(EJK) | [🤗 Hub](https://huggingface.co/seastar105/pflow-encodec-ejk)                  | [Link](https://github.com/seastar105/pflow-encodec/blob/main/MODEL_CARD.md) |
| English           | [🤗 Hub](https://huggingface.co/seastar105/pflow-encodec-libritts)             |                                                                             |
| Japanese          | [🤗 Hub](https://huggingface.co/seastar105/pflow-encodec-aihub-libri-japanese) |                                                                             |
| Korean            | [🤗 Hub](https://huggingface.co/seastar105/pflow-encodec-aihub-libri-korean)   |                                                                             |

# TODO

- [x] Implement baseline model.
- [x] Train model on libritts-r.
- [ ] Simple gradio demo.
- [x] Dataset preparation documentation.
- [x] Train model on another language, i'm planning to train on Korean and Japanese.
- [x] Multilingual model.
- [ ] Test Language ID embedding in Text Encoder for Multilingual Model
- [ ] Train small bert with SeamlessM4T's tokenizer then apply it to Text Encoder.

# Difference from paper

I did not conduct ablation studies for each changes due to lack of resources.

- Use [Encodec](https://github.com/facebookresearch/audiocraft/blob/main/docs/ENCODEC.md) instead of MelSpectrogram.
- Use character-base input instead of phoneme, and GT duration as a target of duration predictor instead of MAS.
- Use AdaLN-Zero from [DiT](https://arxiv.org/abs/2212.09748) for speaker-conditioned text encoder instead of concat and self-attention.
- Use transformer as Flow Matching decoder instead of Wavenet blocks with AdaLN-Single timestep conditioning from [PixArt-α](https://arxiv.org/abs/2310.00426)
- Use attention pooling instead of mean pooling to get fixed-size speaker embedding as P-Flow used in their ablation study.
- Use conv-feedforward(FFT Block from Fastspeech) and GeGLU
- Use Alibi + Convolution positional encoding in transformer, from data2vec 2.0 and voicebox
- Use null cond for CFG sampling instead of mean-pooled hidden vectors.

# Credits

- I borrowed some code from [VITS repo](https://github.com/jaywalnut310/vits), [voicebox-pytorch](https://github.com/lucidrains/voicebox-pytorch), and [fairseq2](https://github.com/facebookresearch/fairseq2).
- This research used datasets from 'The Open AI Dataset Project (AI-Hub, S. Korea)'. All data information can be accessed through 'AI-Hub (www.aihub.or.kr)
