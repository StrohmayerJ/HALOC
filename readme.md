# PyTorch Dataloader for the HALOC Dataset

### Paper
**Strohmayer, J., and Kampel, M.** (2024). “WiFi CSI-based Long-Range Person Localization Using Directional Antennas”, The Second Tiny Papers Track at ICLR 2024, May 2024, Vienna, Austria. https://openreview.net/forum?id=AOJFcEh5Eb

BibTeX:
```BibTeX
@inproceedings{strohmayer2024wifi,
title={WiFi CSI-based Long-Range Person Localization Using Directional Antennas},
author={Julian Strohmayer and Martin Kampel},
booktitle={The Second Tiny Papers Track at ICLR 2024},
year={2024},
url={https://openreview.net/forum?id=AOJFcEh5Eb}
}
```

### WiFi System
<img src="https://github.com/user-attachments/assets/79caebc8-6d96-4726-a88f-dfee70093980" alt="System" width="300"/>

### Prerequisites
```
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset
Get the HALOC dataset from https://zenodo.org/records/10715595 and put it in the `/data` directory.

### Training & Testing 
Example command for training and testing a dummy ResNet18 model on CSI amplitude features with a window size of 351 WiFi packets (~3.51 seconds):

```
python3 train.py --data /data/HALOC --bs 128 --ws 351 
```
In this configuration, the samples will have a shape of [128, 1, 52, 351] = [batch size, channels, subcarriers, window size].

Note: When you run `train.py` for the first time, the program will create CSI cache files (.npy) for each sequence in `/data/HALOC` to improve data loading times on subsequent runs. 

