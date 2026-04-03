# GNR 638 — Assignment 3: PSPNet from Scratch

**Paper:** Zhao et al., *"Pyramid Scene Parsing Network"*, CVPR 2017  
**Student:** Satwik Bhole

---

## Overview

This repository contains a **from-scratch PyTorch implementation** of PSPNet and a comparison against the [official hszhao/semseg PSPNet](https://github.com/hszhao/semseg). Both models are trained on a **toy subset of PASCAL VOC 2012** and evaluated on loss, pixel accuracy, and mean IoU. The hszhao repository is automatically cloned into the project directory on first run.

## Repository Structure

```
├── my_pspnet.py        # From-scratch PSPNet (Dilated ResNet-50 + PPM + Aux loss)
├── dataset_loader.py   # PASCAL VOC 2012 toy dataset with augmentations
├── train.py            # Training loop, evaluation, plots, and comparison
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── semseg/             # Auto-cloned: hszhao/semseg official implementation
├── data/               # Auto-downloaded: PASCAL VOC 2012 dataset
└── checkpoints/        # Saved model weights after training
```

## Key Architecture Details (from the paper)

| Component | Detail |
|---|---|
| **Backbone** | Dilated ResNet-50 with deep stem (three 3×3 convs) + dilated layer3 & layer4 (output stride = 8) |
| **Pyramid Pooling Module** | Adaptive average pooling at bin sizes (1, 2, 3, 6), each reduced to 512 channels, then upsampled and concatenated → 4096 channels |
| **Classifier** | Conv 3×3 (4096→512) → BN → ReLU → Dropout(0.1) → Conv 1×1 (512→C) |
| **Auxiliary Loss** | Attached to layer3 output; weight = 0.4; same head structure (1024→256→C) |
| **Optimizer** | SGD, momentum=0.9, weight_decay=1e-4 |
| **LR Schedule** | Poly decay: `lr = base_lr × (1 − iter/max_iter)^0.9` |
| **Differential LR** | Backbone at base_lr, new heads at 10× base_lr |
| **Crop Size** | 473×473 (paper's setting for VOC) |
| **Augmentations** | Random scale [0.5, 2.0], horizontal flip, brightness jitter, Gaussian blur, random crop |
| **Multi-scale Test** | Scales [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] with horizontal flip |
| **Metrics** | Mean IoU (primary), pixel accuracy, loss |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train both models and generate comparison

```bash
python train.py
```

This will:
- Download PASCAL VOC 2012 into `data/` (on first run)
- Clone the official hszhao/semseg repo into `semseg/` (on first run)
- Train the scratch PSPNet for 15 epochs on 257×257 crops
- Train the official hszhao PSPNet for 15 epochs on the same data
- Run multi-scale inference (scales 0.75–1.25 + flip) on both models
- Save comparison plots to `comparison_plots.png`
- Save qualitative predictions to `qualitative_results.png`
- Print a final metrics comparison table (single-scale + multi-scale)
- Save model checkpoints to `checkpoints/`

### 3. Run the model smoke test

```bash
python my_pspnet.py
```

### 4. Test the dataset loading

```bash
python dataset_loader.py
```

## Outputs

| File | Description |
|---|---|
| `comparison_plots.png` | 2×2 grid: train loss, val loss, pixel accuracy, mIoU over epochs |
| `qualitative_results.png` | Side-by-side: input, ground truth, scratch prediction, official prediction |
| `checkpoints/` | Saved model weights for both models |

## Scratch vs Official — What's Compared

| Aspect | Scratch | Official (hszhao) |
|---|---|---|
| Backbone | Dilated ResNet-50 with deep stem (ImageNet pretrained stages) | ResNet-50 (ImageNet pretrained) |
| Stem | Three 3×3 convs (paper's modified ResNet) | Standard 7×7 conv |
| PPM | Custom PyramidPoolingModule | hszhao's built-in PSP head |
| Auxiliary loss | Yes (0.4 weight) | Yes (hszhao includes it natively) |
| Crop size | 257×257 (fits 6 GB VRAM; satisfies (x−1)%8=0) | Same |
| Multi-scale test | Yes (0.75–1.25 + flip) | Same |
| Optimizer | SGD + Poly LR (10× for heads) | SGD + Poly LR (same for fair comparison) |
| Dataset | VOC 2012 subset (500 train / 100 val) | Same |

## Notes

- We use a subset of VOC (500 train / 100 val) and 15 epochs to keep training practical on a 6 GB GPU while still producing meaningful learning curves. The paper trains on the full dataset for 50 epochs.
- The deep stem (three 3×3 convs) matches the paper's modified ResNet. Since torchvision doesn't provide this variant, we build it from scratch and bridge it to the pretrained residual stages by patching the first Bottleneck of layer1.
- The official **hszhao/semseg** repo is cloned automatically into `semseg/` inside the project directory on first run — no manual setup required.
- The PASCAL VOC 2012 dataset is downloaded automatically into `data/` inside the project directory on first run.
- Multi-scale inference at test time follows the paper's evaluation protocol (horizontal flipping included).
- All comparisons use the exact same data splits and hyperparameters for fairness.
