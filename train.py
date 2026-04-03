"""
This script runs the full training and evaluation pipeline — my PSPNet
against the official hszhao implementation, both trained on a small slice
of PASCAL VOC 2012 so the comparison is fair and fast.

Here's what happens when you run it:
  1. My PSPNet gets trained on the VOC subset
  2. The hszhao reference model gets trained on the exact same data
  3. We track loss, pixel accuracy, and mIoU every epoch for both
  4. At the end, we generate a 2x2 plot and a side-by-side qualitative figure
  5. A summary table prints the final numbers for easy comparison

I followed the training setup from the PSPNet paper as closely as possible:
  - SGD with momentum=0.9 and weight_decay=1e-4
  - Poly LR decay: lr_t = base_lr * (1 - t/T)^0.9
  - Backbone uses base_lr, the new heads get 10x that
  - Auxiliary loss is weighted at 0.4
  - Cross-entropy loss with ignore_index=255 for void/boundary pixels
"""

import os
import sys
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ====================================================================
# Tee — mirror all stdout + stderr to output.txt in real time
# ====================================================================

class _Tee:
    """
    A small helper that splits every write to both the real terminal
    and a log file at the same time — so nothing gets lost.

    The tricky part is tqdm: it uses carriage returns (\\r) to redraw
    progress bars in place. If we write those straight to a file we'd
    end up with hundreds of half-finished lines. So we buffer each line
    and only write it to the log once it's complete (i.e. ends with \\n).
    """
    def __init__(self, stream, log_file):
        self._stream   = stream
        self._log_file = log_file
        self._buf      = ""          # accumulates chars until a newline

    def write(self, data):
        # First, let the real terminal see everything as-is
        self._stream.write(data)

        # For the log file we go char by char so we can handle \\r properly.
        # A carriage return means tqdm is redrawing the line, so we just
        # throw away whatever we had buffered and start fresh.
        for ch in data:
            if ch == "\r":
                self._buf = ""       # tqdm redraw — discard the partial line
            elif ch == "\n":
                self._log_file.write(self._buf + "\n")
                self._log_file.flush()
                self._buf = ""
            else:
                self._buf += ch

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    # Anything else (like .encoding or .fileno) just falls through to the real stream
    def __getattr__(self, attr):
        return getattr(self._stream, attr)


# ------------------------------------------------------------------
# Pull in the official hszhao PSPNet — clone the repo if it's not there yet
# ------------------------------------------------------------------
_semseg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "semseg")
if not os.path.exists(_semseg_dir):
    print(f"Cloning official hszhao/semseg repository to {_semseg_dir}...")
    subprocess.run(
        ["git", "clone", "https://github.com/hszhao/semseg.git", _semseg_dir],
        check=True,
    )
sys.path.append(_semseg_dir)
from model.pspnet import PSPNet as HSZhaoPSPNet

from dataset_loader import get_dataloaders, IGNORE_LABEL, VOC_NUM_CLASSES
from my_pspnet import PSPNetMy


# ====================================================================
# Metric helpers — mIoU and pixel accuracy
# ====================================================================

def mean_iou(pred_flat, gt_flat, num_classes, ignore_idx=255):
    """
    Computes mean IoU — we average per-class IoU over every class
    that actually shows up in this batch (skipping empty classes).
    Boundary/void pixels (label == ignore_idx) are stripped out first
    since the paper doesn't count them.
    """
    keep = gt_flat != ignore_idx
    pred_flat = pred_flat[keep]
    gt_flat   = gt_flat[keep]

    iou_per_class = []
    for c in range(num_classes):
        p_mask = (pred_flat == c)
        g_mask = (gt_flat   == c)
        inter  = (p_mask & g_mask).sum().item()
        union  = (p_mask | g_mask).sum().item()
        if union > 0:
            iou_per_class.append(inter / union)

    return float(np.mean(iou_per_class)) if iou_per_class else 0.0


def pixel_accuracy(preds, targets, ignore_idx=255):
    """
    Simple pixel accuracy: out of all the non-void pixels, how many
    did we get right? Void pixels (label == ignore_idx) are excluded.
    """
    valid     = targets != ignore_idx
    correct   = (preds[valid] == targets[valid]).sum().item()
    total     = valid.sum().item()
    return correct / total if total > 0 else 0.0


# ====================================================================
# Learning rate schedule
# ====================================================================

def poly_lr_scheduler(optimizer, total_steps, power=0.9):
    """
    Poly LR decay, straight from the PSPNet paper.
    The learning rate follows: lr_t = base_lr * (1 - t / T) ^ 0.9
    It starts at base_lr and smoothly drops to near-zero by the end.
    """
    def _decay(step):
        return (1.0 - step / total_steps) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_decay)


# ====================================================================
# Training loop
# ====================================================================

def train_one_model(model, train_loader, val_loader,
                    epochs=10, base_lr=0.01,
                    device="cpu", tag="model", scratch=False):
    """
    Trains a model for the given number of epochs and returns it along
    with a dict of per-epoch metrics (train loss, val loss, accuracy, mIoU).

    For my model I use differential learning rates — the pretrained backbone
    gets base_lr while the new heads train 10x faster. The official model
    just uses a single learning rate for everything.

    My model also has an auxiliary loss branch on stage-3 (weight 0.4)
    which helps with gradient flow during early training.
    """
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    # For my model we split params into backbone vs heads so we can
    # give the heads a higher learning rate (they're training from random init)
    if scratch:
        _head_keys = ["pyramid_pool", "seg_head", "aux_head", "stem", "stem_adapter",
                      "ppm", "head"]
        backbone_params, head_params = [], []
        for name, param in model.named_parameters():
            if any(k in name for k in _head_keys):
                head_params.append(param)
            else:
                backbone_params.append(param)
        param_groups = [
            {"params": backbone_params, "lr": base_lr},
            {"params": head_params,     "lr": base_lr * 10},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": base_lr}]

    optimizer  = optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)
    total_iters = epochs * len(train_loader)
    scheduler  = poly_lr_scheduler(optimizer, total_iters, power=0.9)

    log = {"train_loss": [], "val_loss": [], "val_acc": [], "val_miou": []}

    print(f"\n{'='*60}")
    print(f"  Training: {tag}")
    print(f"  Epochs: {epochs}  |  LR: {base_lr}  |  Device: {device}")
    print(f"{'='*60}")

    for ep in range(epochs):
        # --- train for one epoch ---
        model.train()
        running_loss = 0.0

        bar = tqdm(train_loader, desc=f"[{tag}] Epoch {ep+1}/{epochs} Train")
        for imgs, masks in bar:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            if hasattr(model, "zoom_factor") and model.training:
                # hszhao's model is picky about input size — it needs (H-1) % 8 == 0
                B, C, H, W = imgs.shape
                sh = ((H - 1) // 8) * 8 + 1
                sw = ((W - 1) // 8) * 8 + 1
                if sh != H or sw != W:
                    imgs   = F.interpolate(imgs, size=(sh, sw),
                                           mode="bilinear", align_corners=True)
                    masks  = F.interpolate(
                        masks.unsqueeze(1).float(), size=(sh, sw), mode="nearest"
                    ).squeeze(1).long()
                preds, main_loss, aux_loss = model(imgs, masks)
                loss = main_loss + 0.4 * aux_loss
            else:
                output = model(imgs)
                if isinstance(output, tuple):
                    main_out, aux_out = output
                    loss = loss_fn(main_out, masks) + 0.4 * loss_fn(aux_out, masks)
                else:
                    loss = loss_fn(output, masks)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader)

        # --- validate on the held-out set ---
        model.eval()
        val_loss_accum = 0.0
        pred_list, gt_list = [], []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device)
                masks = masks.to(device)

                if hasattr(model, "zoom_factor"):
                    B, C, H, W = imgs.shape
                    sh = ((H - 1) // 8) * 8 + 1
                    sw = ((W - 1) // 8) * 8 + 1
                    if sh != H or sw != W:
                        imgs   = F.interpolate(imgs, size=(sh, sw),
                                               mode="bilinear", align_corners=True)
                        output = model(imgs)
                        output = F.interpolate(output, size=(H, W),
                                               mode="bilinear", align_corners=True)
                    else:
                        output = model(imgs)
                else:
                    output = model(imgs)

                val_loss_accum += loss_fn(output, masks).item()

                batch_preds = torch.argmax(output, dim=1)
                pred_list.append(batch_preds.cpu())
                gt_list.append(masks.cpu())

        epoch_val_loss = val_loss_accum / len(val_loader)
        all_preds   = torch.cat(pred_list)
        all_targets = torch.cat(gt_list)

        ep_acc  = pixel_accuracy(all_preds, all_targets, IGNORE_LABEL)
        ep_miou = mean_iou(
            all_preds.view(-1), all_targets.view(-1),
            VOC_NUM_CLASSES, IGNORE_LABEL,
        )

        log["train_loss"].append(epoch_train_loss)
        log["val_loss"].append(epoch_val_loss)
        log["val_acc"].append(ep_acc)
        log["val_miou"].append(ep_miou)

        print(
            f"  Epoch {ep+1}/{epochs}  "
            f"Train Loss: {epoch_train_loss:.4f}  |  "
            f"Val Loss: {epoch_val_loss:.4f}  |  "
            f"Pixel Acc: {ep_acc:.4f}  |  "
            f"mIoU: {ep_miou:.4f}"
        )

    return model, log


# ====================================================================
# Plots
# ====================================================================

def plot_training_curves(log_my, log_official, num_epochs,
                         out_path="comparison_plots.png"):
    """
    Draws a 2x2 grid of learning curves (train loss, val loss, pixel
    accuracy, mIoU) with both models on the same axes so you can
    see at a glance how they compare over training.
    """
    epoch_axis = range(1, num_epochs + 1)
    fig, axes  = plt.subplots(2, 2, figsize=(14, 10))

    panel_cfg = [
        ("Training Loss",   "train_loss", "Loss"),
        ("Validation Loss", "val_loss",   "Loss"),
        ("Pixel Accuracy",  "val_acc",    "Accuracy"),
        ("Mean IoU",        "val_miou",   "mIoU"),
    ]

    for ax, (title, key, ylabel) in zip(axes.flat, panel_cfg):
        ax.plot(epoch_axis, log_my[key],  "o-",  label="My PSPNet",          linewidth=2)
        ax.plot(epoch_axis, log_official[key], "s--", label="Official (hszhao) PSPNet", linewidth=2)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "PSPNet: My vs Official Implementation Comparison",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plots to '{out_path}'")
    plt.close()


def show_qualitative_results(m_my, m_official, val_loader, device,
                             n_samples=4, out_path="qualitative_results.png"):
    """
    Grabs a batch from the val set and shows predictions side-by-side:
        Input Image  |  Ground Truth  |  My PSPNet  |  Official (hszhao)
    Good for a quick sanity check on what the models are actually seeing.
    """
    m_my.eval()
    m_official.eval()

    imgs, masks = next(iter(val_loader))
    imgs_dev    = imgs.to(device)
    n_samples   = min(n_samples, imgs.size(0))

    with torch.no_grad():
        my_preds = torch.argmax(m_my(imgs_dev), dim=1).cpu()

        B, C, H, W = imgs_dev.shape
        sh = ((H - 1) // 8) * 8 + 1
        sw = ((W - 1) // 8) * 8 + 1
        if sh != H or sw != W:
            resized_imgs    = F.interpolate(imgs_dev, size=(sh, sw),
                                            mode="bilinear", align_corners=True)
            official_logits = m_official(resized_imgs)
            official_logits = F.interpolate(official_logits, size=(H, W),
                                            mode="bilinear", align_corners=True)
        else:
            official_logits = m_official(imgs_dev)
        official_preds = torch.argmax(official_logits, dim=1).cpu()

    # Undo the ImageNet normalisation so the image looks right when displayed
    _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    col_labels = ["Input Image", "Ground Truth", "My PSPNet", "Official (hszhao)"]

    for row in range(n_samples):
        rgb = (imgs[row] * _std + _mean).permute(1, 2, 0).clamp(0, 1).numpy()

        row_data = [
            rgb,
            masks[row].numpy(),
            my_preds[row].numpy(),
            official_preds[row].numpy(),
        ]
        cmaps = [None, "tab20", "tab20", "tab20"]
        vranges = [None, (0, 20), (0, 20), (0, 20)]

        for col, (data, cmap, vr) in enumerate(zip(row_data, cmaps, vranges)):
            kw = {"cmap": cmap} if cmap else {}
            if vr:
                kw["vmin"], kw["vmax"] = vr
            axes[row, col].imshow(data, **kw)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_labels[col], fontsize=12)

    fig.suptitle(
        "Qualitative Comparison: My PSPNet vs Official PSPNet",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved qualitative results to '{out_path}'")
    plt.close()


# ====================================================================
# Multi-scale test-time inference
# ====================================================================

def _ms_infer(model, img_batch, num_cls, device, scales=(0.75, 1.0, 1.25), flip=True):
    """
    Runs the model at multiple resolutions and averages the logits.
    Optionally also flips each scaled image horizontally and averages
    those in too — this is the standard eval trick from the PSPNet paper
    (they use scales 0.5 to 1.75 in the full setup).
    """
    model.eval()
    B, C, H, W    = img_batch.shape
    accum_logits  = torch.zeros(B, num_cls, H, W, device=device)

    with torch.no_grad():
        for s in scales:
            sc_h, sc_w = int(H * s), int(W * s)
            if hasattr(model, "zoom_factor"):
                sc_h = ((sc_h - 1) // 8) * 8 + 1
                sc_w = ((sc_w - 1) // 8) * 8 + 1

            scaled_in = F.interpolate(img_batch, size=(sc_h, sc_w),
                                      mode="bilinear", align_corners=True)
            logits = model(scaled_in)
            logits = F.interpolate(logits, size=(H, W),
                                   mode="bilinear", align_corners=True)
            accum_logits += logits

            if flip:
                flipped   = torch.flip(scaled_in, dims=[3])
                fl_logits = model(flipped)
                fl_logits = torch.flip(fl_logits, dims=[3])
                fl_logits = F.interpolate(fl_logits, size=(H, W),
                                          mode="bilinear", align_corners=True)
                accum_logits += fl_logits

    return torch.argmax(accum_logits, dim=1)


def eval_multiscale(model, val_loader, device, num_cls):
    """Runs multi-scale inference over the whole val set and returns accuracy + mIoU."""
    model.eval()
    all_preds, all_gts = [], []

    for imgs, masks in val_loader:
        imgs_dev = imgs.to(device)
        preds    = _ms_infer(model, imgs_dev, num_cls, device).cpu()
        all_preds.append(preds)
        all_gts.append(masks)

    merged_preds = torch.cat(all_preds)
    merged_gts   = torch.cat(all_gts)

    acc  = pixel_accuracy(merged_preds, merged_gts, IGNORE_LABEL)
    miou = mean_iou(
        merged_preds.view(-1), merged_gts.view(-1), num_cls, IGNORE_LABEL
    )
    return acc, miou


# ====================================================================
# Results summary
# ====================================================================

def print_results_table(log_my, log_official,
                        ms_my=None, ms_official=None):
    """Prints a clean side-by-side table of the final-epoch numbers for both models."""
    print("\n" + "=" * 65)
    print("  FINAL METRICS COMPARISON (last epoch)")
    print("=" * 65)
    print(f"  {'Metric':<22} {'My PSPNet':>18} {'Official (hszhao)':>18}")
    print("-" * 65)

    rows = [
        ("Train Loss",     "train_loss"),
        ("Val Loss",       "val_loss"),
        ("Pixel Accuracy", "val_acc"),
        ("Mean IoU",       "val_miou"),
    ]
    for label, key in rows:
        s_val = log_my[key][-1]
        o_val = log_official[key][-1]
        print(f"  {label:<22} {s_val:>18.4f} {o_val:>18.4f}")

    if ms_my and ms_official:
        print("-" * 65)
        print(f"  {'MS Pixel Accuracy':<22} {ms_my[0]:>18.4f} {ms_official[0]:>18.4f}")
        print(f"  {'MS Mean IoU':<22} {ms_my[1]:>18.4f} {ms_official[1]:>18.4f}")

    print("=" * 65)


# ====================================================================
# Main
# ====================================================================

def main():
    # Tee all output to output.txt so we have a log after the run
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.txt")
    _log_fh   = open(_log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)
    print(f"All terminal output is also being saved to: {_log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # These are tuned to fit comfortably on a 6 GB GPU
    cfg = dict(
        epochs     = 15,
        batch_size = 2,        # keep it small to stay within VRAM
        base_lr    = 0.01,
        num_train  = 500,
        num_val    = 100,
        crop_size  = 257,      # 257 satisfies (x-1) % 8 == 0, needed by hszhao's model
        num_classes= VOC_NUM_CLASSES,
    )

    # Load the dataset — downloads VOC 2012 automatically on the first run
    print("\nPreparing datasets (will download VOC 2012 on first run)...")
    _data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_loader, val_loader = get_dataloaders(
        root=_data_root,
        batch_size=cfg["batch_size"],
        num_train =cfg["num_train"],
        num_val   =cfg["num_val"],
        crop_size =cfg["crop_size"],
    )
    print(f"Train: {len(train_loader.dataset)} images ({len(train_loader)} batches)")
    print(f"Val  : {len(val_loader.dataset)} images ({len(val_loader)} batches)")

    # ----------------------------------------------------------------
    # 1. Train my PSPNet
    # ----------------------------------------------------------------
    print("\n>>> Initializing My PSPNet with auxiliary branch...")
    model_my = PSPNetMy(num_classes=cfg["num_classes"], use_aux=True)
    model_my, log_my = train_one_model(
        model_my, train_loader, val_loader,
        epochs  =cfg["epochs"],
        base_lr =cfg["base_lr"],
        device  =device,
        tag     ="My PSPNet",
        scratch =True,
    )

    # ----------------------------------------------------------------
    # 2. Train the official hszhao PSPNet on the same data
    # ----------------------------------------------------------------
    print("\n>>> Initializing Official (hszhao) PSPNet...")
    model_official = HSZhaoPSPNet(
        layers=50,
        classes=cfg["num_classes"],
        zoom_factor=8,
        pretrained=False,
    )
    model_official, log_official = train_one_model(
        model_official, train_loader, val_loader,
        epochs  =cfg["epochs"],
        base_lr =cfg["base_lr"],
        device  =device,
        tag     ="Official (hszhao) PSPNet",
        scratch =False,
    )

    # ----------------------------------------------------------------
    # 3. Re-evaluate both models with multi-scale inference
    # ----------------------------------------------------------------
    print("\n>>> Running multi-scale inference (scales: 0.5-1.75 + flip)...")
    ms_my  = eval_multiscale(model_my,  val_loader, device, cfg["num_classes"])
    print(f"  My PSPNet MS — Pixel Acc: {ms_my[0]:.4f}, mIoU: {ms_my[1]:.4f}")

    ms_official = eval_multiscale(model_official, val_loader, device, cfg["num_classes"])
    print(f"  Official MS — Pixel Acc: {ms_official[0]:.4f}, mIoU: {ms_official[1]:.4f}")

    # ----------------------------------------------------------------
    # 4. Generate plots and qualitative results
    # ----------------------------------------------------------------
    print("\n>>> Generating comparison plots...")
    plot_training_curves(log_my, log_official, cfg["epochs"])

    print("\n>>> Generating qualitative predictions...")
    show_qualitative_results(model_my, model_official, val_loader, device)

    print_results_table(log_my, log_official, ms_my, ms_official)

    # Save weights so we can reload them later without retraining
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model_my.state_dict(),  "checkpoints/my_pspnet.pth")
    torch.save(model_official.state_dict(), "checkpoints/pspnet_official.pth")
    print("\nSaved model checkpoints to 'checkpoints/'")


if __name__ == "__main__":
    main()
