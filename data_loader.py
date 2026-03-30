# data_loader.py
# ---------------
# Handles loading PASCAL VOC 2012 data for training and validation.
# Using a subset of VOC since the full thing is too big for a quick experiment.
# The paper trains on ADE20K and Cityscapes but VOC works for demonstrating
# the model works. I matched the paper's 473x473 crop size and augmentations.

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance


# VOC2012 stuff
N_CLASSES = 21        # 20 object classes + background
IGNORE_LBL = 255     # boundary/dont-care pixels in VOC


class VOCSubset(Dataset):
    """
    Wraps torchvision VOCSegmentation but only uses a small chunk of it.
    Also handles all the augmentation and preprocessing.
    
    I followed the augmentation strategy from the paper:
    random scaling, horizontal flips, some blur, and random crops.
    The paper uses 473x473 crops so thats what we do here too.
    """

    def __init__(self, root_dir, split='train', download=True,
                 subset_size=50, crop_sz=473, do_augment=True):
        self.raw_dataset = VOCSegmentation(
            root=root_dir, year='2012',
            image_set=split, download=download
        )
        # only take first N samples
        self.count = min(subset_size, len(self.raw_dataset))
        self.crop_sz = crop_sz
        self.do_augment = do_augment

        # imagenet stats for normalization (since backbone is pretrained on imagenet)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        img, seg = self.raw_dataset[i]

        if self.do_augment:
            img, seg = self._augment(img, seg)
        else:
            # val: just resize, no fancy stuff
            img = img.resize((self.crop_sz, self.crop_sz), Image.BILINEAR)
            seg = seg.resize((self.crop_sz, self.crop_sz), Image.NEAREST)

        img_tensor = self.to_tensor(img)
        seg_tensor = torch.from_numpy(np.array(seg)).long()
        return img_tensor, seg_tensor

    def _augment(self, img, seg):
        """
        Training augmentations. Tried to match what the paper describes
        in section 4.1 — multi-scale input, random mirror, etc.
        I also threw in brightness jitter and gaussian blur because
        ive seen those help in other segmentation papers.
        """

        # 1) random scale between 0.5x and 2.0x
        s = random.uniform(0.5, 2.0)
        w, h = img.size
        nw, nh = int(w * s), int(h * s)
        img = img.resize((nw, nh), Image.BILINEAR)
        seg = seg.resize((nw, nh), Image.NEAREST)

        # 2) horizontal flip — 50% chance
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        # 3) brightness tweak — sometimes helps generalization
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # 4) slight gaussian blur
        if random.random() > 0.5:
            r = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=r))

        # 5) pad if needed + random crop
        img, seg = self._pad_crop(img, seg)
        return img, seg

    def _pad_crop(self, img, seg):
        """If image is smaller than crop size after scaling, pad with zeros
        (and pad segmentation with 255 so those pixels get ignored).
        Then take a random crop."""
        w, h = img.size
        pw = max(self.crop_sz - w, 0)
        ph = max(self.crop_sz - h, 0)

        if pw > 0 or ph > 0:
            # pad image with black pixels
            arr = np.array(img, dtype=np.uint8)
            arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)),
                         mode='constant', constant_values=0)
            img = Image.fromarray(arr)

            # pad seg with ignore label
            sarr = np.array(seg, dtype=np.uint8)
            sarr = np.pad(sarr, ((0, ph), (0, pw)),
                          mode='constant', constant_values=IGNORE_LBL)
            seg = Image.fromarray(sarr)

        # random crop
        w, h = img.size
        x1 = random.randint(0, w - self.crop_sz)
        y1 = random.randint(0, h - self.crop_sz)
        x2, y2 = x1 + self.crop_sz, y1 + self.crop_sz
        img = img.crop((x1, y1, x2, y2))
        seg = seg.crop((x1, y1, x2, y2))
        return img, seg


def build_loaders(data_path='./data', bs=4, n_train=200, n_val=50,
                  crop_sz=473, num_workers=2):
    """Make train/val dataloaders. Nothing special here."""
    train_ds = VOCSubset(data_path, 'train', download=True,
                         subset_size=n_train, crop_sz=crop_sz, do_augment=True)
    val_ds = VOCSubset(data_path, 'val', download=True,
                       subset_size=n_val, crop_sz=crop_sz, do_augment=False)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


if __name__ == '__main__':
    print("testing data loading...")
    tl, vl = build_loaders()
    print(f"train: {len(tl.dataset)} imgs, {len(tl)} batches")
    print(f"val:   {len(vl.dataset)} imgs, {len(vl)} batches")
    batch_imgs, batch_segs = next(iter(tl))
    print(f"img shape: {batch_imgs.shape}, seg shape: {batch_segs.shape}")
    print(f"unique labels in batch: {torch.unique(batch_segs).tolist()}")
