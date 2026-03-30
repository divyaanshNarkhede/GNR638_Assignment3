# PSPNet — My Implementation (Assignment 3)

Paper: Zhao et al., "Pyramid Scene Parsing Network", CVPR 2017

## What's in here

I implemented PSPNet from scratch by reading the paper and coding each component myself. Then I compared it against the `segmentation-models-pytorch` (SMP) library's PSPNet to see how well my version holds up.

Everything runs on a tiny subset of PASCAL VOC 2012 (200 train / 50 val images). The paper trains on much bigger datasets for way more epochs, but this is enough to verify things work.

## Files

| File | What it does |
|---|---|
| `model.py` | The PSPNet architecture — backbone, PPM, classifier heads |
| `data_loader.py` | Loads VOC2012 with augmentations (scaling, flip, crop, blur) |
| `run_training.py` | Trains both models, evaluates, plots, saves weights |
| `requirements.txt` | pip dependencies |

## How the network works (my understanding)

The problem with regular FCN-style networks is they only look at local context. So they make silly mistakes — like calling a boat a car because the local texture is similar. PSPNet fixes this with the **Pyramid Pooling Module (PPM)**:

1. Take the feature map from a dilated ResNet-50 backbone (output stride = 8, so 1/8 of input resolution)
2. Pool it at 4 different grid sizes: 1×1, 2×2, 3×3, 6×6
3. Each pooled map goes through a 1×1 conv (reduces channels from 2048 to 512)
4. Upsample all of them back to the original feature map size
5. Concatenate everything → 2048 + 4×512 = 4096 channels
6. Feed that into a classifier head → per-pixel class scores

There's also an auxiliary head on layer3 output that helps training converge faster (deep supervision, weighted at 0.4).

## Training details

Tried to follow the paper:
- SGD, momentum=0.9, weight_decay=1e-4
- Poly LR schedule: `lr = base_lr × (1 - iter/max_iter)^0.9`
- New layers learn at 10× the backbone learning rate
- Base LR = 0.005 (lower than paper's 0.01 since dataset is tiny)
- Batch size 4, crop 256×256, 10 epochs

## Running it

```bash
pip install -r requirements.txt
python run_training.py
```

This trains both models, makes comparison plots (`comparison.png`), sample predictions (`predictions.png`), and saves weights to `saved_models/`.

## What I simplified vs the paper

- Standard ResNet-50 stem instead of the deep stem (three 3×3 convs) they mention
- Much smaller dataset and fewer epochs
- 256×256 crops instead of 473×473
- No multi-scale inference at test time
