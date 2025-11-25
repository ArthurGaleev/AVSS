# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

Implementation of the AVSS models based on the articles:

[RTFS-NET: RECURRENT TIME-FREQUENCY MODELLING FOR EFFICIENT AUDIO-VISUAL SPEECH SEPARATION](https://arxiv.org/pdf/2309.17189).

[DUAL-PATH RNN: EFFICIENT LONG SEQUENCE MODELING FOR
TIME-DOMAIN SINGLE-CHANNEL SPEECH SEPARATION](https://arxiv.org/pdf/1910.06379).
## Installation

1. Install uv
   ```bash
   pip install uv
   ```
2. Install all required packages

   ```bash
   uv init
   uv sync
   ```
2. Download all required models and dataset:

   ```bash
   uv run scripts/download_gdrive.py
   ```

## How To Use
Before using model should make the env YANDEX_DISK_URL global depending on the dataset you want to download from yandex disk(example with our dataset):
```bash
   export YANDEX_DISK_URL=https://disk.360.yandex.ru/d/5pz96ysIZi33IQ
```
To train our best rtfs model, run the following command:

```bash
   uv run train.py model=rtfs-4-reuse -cn=rtfs-net
```
To train our best dprnn model, run the following command:

```bash
   uv run train.py -cn=dprnn
```
(more details in report)
To run inference on our best rtfs checkpoint(rtfs-3-reuse):
Using Yandex disk dataset(need to export, see above):

```bash
   uv run inference.py inferencer.save_path="pred_small_av" inferencer.from_pretrained="data/models/rtfs-3-reuse.pth" download_name=dla_dataset_small_av model=rtfs-3-reuse -cn=inference
```
Inference our best rtfs checkpoint on downloaded dataset(YOUR_FOLDER should be in data/datasets folder):
```bash
   uv run inference.py inferencer.save_path=PRED_FOLDER_NAME inferencer.from_pretrained="data/models/rtfs-3-reuse.pth" download_name=YOUR_FOLDER model=rtfs-3-reuse -cn=inference
```
## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
