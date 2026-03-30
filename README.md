# PSPNet — My Implementation (Assignment 3)

Paper: Zhao et al., "Pyramid Scene Parsing Network", CVPR 2017

## What's in here

I implemented PSPNet from scratch by reading the paper and coding each component myself. Then I compared it against the `segmentation-models-pytorch` (SMP) library's PSPNet to see how well my version holds up.

Everything runs on a subset of PASCAL VOC 2012 (500 train / 50 val images). The paper trains on much bigger datasets for way more epochs, but this is enough to verify things work.

## Files

| File | What it does |
|---|---|
| `model.py` | The PSPNet architecture — backbone, PPM, classifier heads |
| `data_loader.py` | Loads VOC2012 with augmentations (scaling, flip, crop, blur) |
| `run_training.py` | Trains both models, evaluates, plots, saves weights |
| `requirements.txt` | pip dependencies |

## How the network works (my understanding)

The problem with regular FCN-style networks is they only look at local context. So they make silly mistakes — like calling a boat a car because the local texture is similar. PSPNet fixes this with the **Pyramid Pooling Module (PPM)**:

1. Take the feature map from a dilated ResNet-50 backbone with a **deep stem** (three 3×3 convs instead of one 7×7, as mentioned in the paper) — output stride = 8
2. Pool it at 4 different grid sizes: 1×1, 2×2, 3×3, 6×6
3. Each pooled map goes through a 1×1 conv (reduces channels from 2048 to 512)
4. Upsample all of them back to the original feature map size
5. Concatenate everything → 2048 + 4×512 = 4096 channels
6. Feed that into a classifier head → per-pixel class scores

There's also an auxiliary head on layer3 output that helps training converge faster (deep supervision, weighted at 0.4).

At test time, **multi-scale inference with horizontal flipping** is used (scales 0.5 to 1.75) — predictions are averaged across all scales and their flips for better accuracy.

## Training details

Matching the paper as closely as possible, with some adjustments due to GPU memory constraints (RTX 3050 6GB):
- SGD, momentum=0.9, weight_decay=1e-4
- Poly LR schedule: `lr = base_lr × (1 - iter/max_iter)^0.9`
- New layers learn at 10× the backbone learning rate
- Base LR = 0.01 (same as the paper)
- Batch size 2, crop 256×256, 15 epochs
- Deep stem (three 3×3 convs) with pretrained weight initialization
- Multi-scale inference at validation (scales: 0.75, 1.0, 1.25 + flipping)

**Note on GPU limitations:** The paper uses 473×473 crops with batch size 16 on high-end GPUs. On my 6GB RTX 3050, that causes OOM errors, so I had to reduce the crop size to 256×256, batch size to 2, and use fewer multi-scale levels (3 instead of 6). The architecture and training recipe are still faithful to the paper — only the scale of the experiment is smaller.

## Running it

```bash
pip install -r requirements.txt
python run_training.py
```

This trains both models, makes comparison plots (`comparison.png`), sample predictions (`predictions.png`), and saves weights to `saved_models/`.

## Remaining simplifications vs the paper

- Training on a VOC 2012 subset (500 train / 50 val) instead of full ADE20K/Cityscapes
- Deep stem weights are initialized from the pretrained 7×7 conv (center crop) rather than being fully retrained on ImageNet
- The paper trains for 150+ epochs on full datasets — we do 15 on a small subset
- Crop size reduced from 473×473 to 256×256 and batch size from 16 to 2 due to 6GB GPU memory
- Multi-scale inference uses 3 scales (0.75, 1.0, 1.25) instead of 6 (0.5–1.75) to fit in VRAM
