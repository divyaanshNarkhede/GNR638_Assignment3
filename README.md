<div align="center">

# PSPNet
### GNR 638 · Assignment 3

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PASCAL%20VOC%202012-green)](http://host.robots.ox.ac.uk/pascal/VOC/)
[![Paper](https://img.shields.io/badge/Paper-CVPR%202017-purple)](https://arxiv.org/abs/1612.01105)

> **Student:** Divyaansh Narkhede  
> **Reference:** Zhao et al., *"Pyramid Scene Parsing Network"*, CVPR 2017

</div>

---

## What's This?

A **PyTorch implementation of PSPNet built from scratch**, benchmarked against the [official hszhao/semseg](https://github.com/hszhao/semseg) implementation. Both models are trained on a toy subset of **PASCAL VOC 2012** and evaluated on loss, pixel accuracy, and mean IoU.

The official repo is auto-cloned on first run — no manual setup needed.

---

## Project Layout

```
GNR638_Assignment3/
│
├── my_pspnet.py          ← From-scratch PSPNet (Dilated ResNet-50 + PPM + Aux loss)
├── dataset_loader.py     ← PASCAL VOC 2012 loader with augmentations
├── train.py              ← Training loop, evaluation, plots, and comparison
│
├── requirements.txt      ← Python dependencies
├── README.md             ← You are here
│
├── semseg/               ← Auto-cloned: hszhao/semseg official implementation
├── data/                 ← Auto-downloaded: PASCAL VOC 2012
└── checkpoints/          ← Saved model weights after training
```

---

## Architecture at a Glance

<details>
<summary><b>Click to expand full architecture table</b></summary>

| Component | Detail |
|:---|:---|
| **Backbone** | Dilated ResNet-50 · deep stem (3× Conv 3×3) · dilated layer3 & layer4 · output stride = 8 |
| **Pyramid Pooling Module** | Avg pool at bins (1, 2, 3, 6) → each 512-ch → upsample → concat → **4096 channels** |
| **Classifier Head** | Conv 3×3 (4096→512) → BN → ReLU → Dropout(0.1) → Conv 1×1 (512→C) |
| **Auxiliary Loss** | On layer3 output · weight = 0.4 · head: 1024→256→C |
| **Optimizer** | SGD · momentum=0.9 · weight_decay=1e-4 |
| **LR Schedule** | Poly decay: `lr = base_lr × (1 − iter/max_iter)^0.9` |
| **Differential LR** | Backbone @ base_lr · new heads @ **10× base_lr** |
| **Crop Size** | 473×473 (paper setting for VOC) |
| **Augmentations** | Random scale [0.5, 2.0] · H-flip · brightness jitter · Gaussian blur · random crop |
| **Multi-scale Test** | Scales [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] + horizontal flip |
| **Metrics** | Mean IoU *(primary)* · pixel accuracy · loss |

</details>

---

## Quick Start

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Train & compare both models

```bash
python train.py
```

<details>
<summary>What happens when you run <code>train.py</code>?</summary>

- Downloads PASCAL VOC 2012 into `data/` *(first run only)*
- Clones hszhao/semseg into `semseg/` *(first run only)*
- Trains scratch PSPNet for **15 epochs** on 257×257 crops
- Trains official hszhao PSPNet for **15 epochs** on the same data
- Runs multi-scale inference (scales 0.75–1.25 + flip) on both
- Saves comparison plots → `comparison_plots.png`
- Saves qualitative predictions → `qualitative_results.png`
- Prints final metrics table (single-scale + multi-scale)
- Saves checkpoints → `checkpoints/`

</details>

### Step 3 — Smoke-test the model

```bash
python my_pspnet.py
```

### Step 4 — Test dataset loading

```bash
python dataset_loader.py
```

---

## Outputs

| File | Description |
|:---|:---|
| `comparison_plots.png` | 2×2 grid — train loss · val loss · pixel accuracy · mIoU over epochs |
| `qualitative_results.png` | Side-by-side: input · ground truth · scratch pred · official pred |
| `checkpoints/` | Saved `.pth` weights for both models |

---

## My PSPNet vs. Official — Head-to-Head

| Aspect | My PSPNet (Ours) | Official (hszhao) |
|:---|:---:|:---:|
| Backbone | Dilated ResNet-50 + deep stem | ResNet-50 (standard) |
| Stem design | Three 3×3 convs *(paper's mod)* | Single 7×7 conv |
| PPM | Custom `PyramidPoolingModule` | hszhao's built-in PSP head |
| Auxiliary loss | (weight = 0.4) | (native) |
| Crop size | 257×257 *(6 GB VRAM friendly)* | Same |
| Multi-scale test | (0.75–1.25 + flip) | Same |
| Optimizer | SGD + Poly LR (10× heads) | SGD + Poly LR |
| Dataset | VOC 2012 subset (500 train / 100 val) | Same |

---

## Notes

- We use a subset of VOC (500 train / 100 val) and 15 epochs to keep training practical on a 6 GB GPU while still producing meaningful learning curves. The paper trains on the full dataset for 50 epochs.
- The deep stem (three 3×3 convs) matches the paper's modified ResNet. Since torchvision doesn't provide this variant, we build it from scratch and bridge it to the pretrained residual stages by patching the first Bottleneck of layer1.
- The official **hszhao/semseg** repo is cloned automatically into `semseg/` inside the project directory on first run — no manual setup required.
- The PASCAL VOC 2012 dataset is downloaded automatically into `data/` inside the project directory on first run.
- Multi-scale inference at test time follows the paper's evaluation protocol (horizontal flipping included).
- All comparisons use the exact same data splits and hyperparameters for fairness.
