"""
Data loading pipeline for PSPNet training on a small PASCAL VOC 2012 slice.

PASCAL VOC is used here as a lightweight stand-in for the larger benchmarks
(ADE20K, Cityscapes) that the original paper evaluates on.  Keeping the
subset small lets training finish in a reasonable time during assignment work.
"""

import os
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

# -------------------------------------------------------------------
# Global constants
# -------------------------------------------------------------------
VOC_NUM_CLASSES = 21      # 20 object categories + 1 background
IGNORE_LABEL    = 255     # VOC uses 255 for border / void regions


class VOCSegDataset(Dataset):
    """
    Lightweight wrapper around torchvision VOCSegmentation.

    Responsibilities:
      * restrict the dataset to a small fixed-size subset
      * apply paper-style augmentations during training
        (scale jitter, flip, blur, random crop)
      * keep the ignore label intact through all transforms
    """

    def __init__(
        self,
        root,
        split="train",
        download=True,
        subset_size=50,
        spatial_size=473,
        training_mode=True,
    ):
        self._base = VOCSegmentation(
            root=root, year="2012", image_set=split, download=download
        )
        self.n = min(subset_size, len(self._base))
        self.spatial_size  = spatial_size
        self.training_mode = training_mode

        # ImageNet statistics -- same pre-processing as the paper
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    # ----------------------------------------------------------------
    # Dataset protocol
    # ----------------------------------------------------------------

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        img, seg = self._base[index]

        if self.training_mode:
            img, seg = self._augment(img, seg)
        else:
            # Validation: simple resize to a canonical size
            img = img.resize((self.spatial_size, self.spatial_size), Image.BILINEAR)
            seg = seg.resize((self.spatial_size, self.spatial_size), Image.NEAREST)

        img = self._to_tensor(img)
        seg = torch.from_numpy(np.array(seg)).long()
        return img, seg

    # ----------------------------------------------------------------
    # Augmentation helpers
    # ----------------------------------------------------------------

    def _augment(self, img, seg):
        """
        Training-time augmentation chain (paper-inspired):
          1. Random scale in [0.5, 2.0]
          2. Random horizontal flip (p=0.5)
          3. Random brightness jitter (p=0.5)
          4. Random Gaussian blur (p=0.5)
          5. Pad-then-random-crop to spatial_size x spatial_size
        """
        # --- scale jitter ---
        scale_factor = random.uniform(0.5, 2.0)
        src_w, src_h  = img.size
        tgt_w = int(src_w * scale_factor)
        tgt_h = int(src_h * scale_factor)
        img = img.resize((tgt_w, tgt_h), Image.BILINEAR)
        seg = seg.resize((tgt_w, tgt_h), Image.NEAREST)

        # --- horizontal flip ---
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        # --- brightness jitter ---
        if random.random() > 0.5:
            from PIL import ImageEnhance
            brightness_factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        # --- Gaussian blur ---
        if random.random() > 0.5:
            blur_radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # --- pad + random crop ---
        img, seg = self._pad_then_crop(img, seg)
        return img, seg

    def _pad_then_crop(self, img, seg):
        """Zero-pad image (ignore-pad mask) to at least spatial_size, then crop."""
        w, h = img.size
        extra_w = max(self.spatial_size - w, 0)
        extra_h = max(self.spatial_size - h, 0)

        if extra_w > 0 or extra_h > 0:
            img_arr = np.array(img, dtype=np.uint8)
            img_arr = np.pad(
                img_arr,
                ((0, extra_h), (0, extra_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            img = Image.fromarray(img_arr)

            seg_arr = np.array(seg, dtype=np.uint8)
            seg_arr = np.pad(
                seg_arr,
                ((0, extra_h), (0, extra_w)),
                mode="constant",
                constant_values=IGNORE_LABEL,
            )
            seg = Image.fromarray(seg_arr)

        w, h   = img.size
        crop_x = random.randint(0, w - self.spatial_size)
        crop_y = random.randint(0, h - self.spatial_size)
        box = (crop_x, crop_y,
               crop_x + self.spatial_size,
               crop_y + self.spatial_size)
        img = img.crop(box)
        seg = seg.crop(box)
        return img, seg


# -------------------------------------------------------------------
# Public factory
# -------------------------------------------------------------------

def get_dataloaders(
    root="./data",
    batch_size=4,
    num_train=200,
    num_val=50,
    crop_size=473,
    num_workers=2,
):
    """
    Construct train and validation DataLoaders for the toy VOC subset.

    Parameters
    ----------
    root        : download/cache directory for VOC
    batch_size  : images per mini-batch
    num_train   : number of training images to use
    num_val     : number of validation images to use
    crop_size   : height/width of training crops
    num_workers : background dataloader workers

    Returns
    -------
    (train_loader, val_loader)
    """
    train_set = VOCSegDataset(
        root,
        split="train",
        download=True,
        subset_size=num_train,
        spatial_size=crop_size,
        training_mode=True,
    )
    val_set = VOCSegDataset(
        root,
        split="val",
        download=True,
        subset_size=num_val,
        spatial_size=crop_size,
        training_mode=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# -------------------------------------------------------------------
# Quick smoke-test
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Downloading and preparing toy dataset...")
    train_loader, val_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    for imgs, masks in train_loader:
        print(f"Image batch: {imgs.shape}, Mask batch: {masks.shape}")
        print(f"Mask unique values: {torch.unique(masks).tolist()}")
        break
