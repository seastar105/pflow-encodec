# Model Card for P-Flow Encodec TTS (English, Korean, Japanese)

## Model Details

### Model Description

P-Flow Encodec is Text-to-Speech model based on paper [P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting](https://openreview.net/pdf?id=zNA7u7wtIN), with some modification. You can check differences [here](https://github.com/seastar105/pflow-encodec?tab=readme-ov-file#difference-from-paper). Model consists of Encodec model from Meta, and Multiband Diffusion decoder, which is also from Meta.

- **Developed by**: [seastar105](https://github.com/seastar105)
- **Model type**: Text-to-Speech
- **Language**: English, Korean, Japanese
- **License**: MIT for codes, also it's free to use model weights, but you should indicate that model weight is trained with data from AI Hub, (e.g. This research (paper) used datasets from 'The Open AI Dataset Project (AI-Hub, S. Korea)'. All data information can be accessed through 'AI-Hub (www.aihub.or.kr))
- **Model version**: 1.0
- **Code Repository**: https://github.com/seastar105/pflow-encodec
- **Model Repository**: https://huggingface.co/seastar105/pflow-encodec-ejk

## Intended Use

### Primary intended use

This model is trained for zero-shot Multilingual TTS. It can be used for generating speech from text in English, Korean, Japanese. Primary intended use is for research purpose, as a baseline for multilingual, code-switch TTS.

### Out of scope use cases

This model should not be used for generating or editing someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities.

## Training Details

- **Training dataset**: LibriTTS-R, Korean and Japanese corpus from [AIHub 131](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71524) dataset (Multi-lingual Read Speech corpus for Translation). Only samples with duration less than 15 seconds and over 3.5 seconds are used, 380 hours for english, 637 hours for japanese, 705 hours for korean.
- **Finetuned from**: Multilingual model is initialized using merged pretrained model for each languages mentioned above. Monolingual model for each language is trained with ~250K steps, and then merge their weights with average. Then, it is finetuned with ~250K steps with all languages.
- **Compute Resource**: All model is trained with one RTX 4090 GPU. It takes about 1 day for 100K steps using 4 gradient accumulation steps with batch_durations of 100.
