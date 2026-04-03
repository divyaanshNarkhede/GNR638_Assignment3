"""
Training and evaluation script — scratch PSPNet vs official (hszhao) PSPNet.

What this script does:
  1.  Trains our from-scratch PSPNet on a toy PASCAL VOC 2012 subset
  2.  Trains the hszhao reference PSPNet on the same data
  3.  Records loss, pixel accuracy, and mean IoU for each epoch
  4.  Generates a 2x2 comparison plot and a qualitative side-by-side figure
  5.  Prints a summary table of final-epoch and multi-scale metrics

Training protocol follows the PSPNet paper:
  - SGD, momentum=0.9, weight_decay=1e-4
  - Poly LR schedule: lr_t = base_lr * (1 - t/T)^0.9
  - Backbone LR = base_lr; new head LR = 10 * base_lr
  - Auxiliary loss weight = 0.4
  - Cross-entropy with ignore_index=255
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
    Wraps a stream (stdout or stderr) so every write goes to both the
    original terminal stream and an open log file simultaneously.

    Progress bars (tqdm) use carriage-return (\\r) to overwrite the same
    terminal line.  Writing those raw to a file produces hundreds of
    duplicate lines.  We strip \\r-based rewrites and only persist the
    final completed line (the one that ends with \\n) to the log file.
    """
    def __init__(self, stream, log_file):
        self._stream   = stream
        self._log_file = log_file
        self._buf      = ""          # accumulates chars until a newline

    def write(self, data):
        # Always pass through to the real terminal unchanged
        self._stream.write(data)

        # For the log file: process character by character so we can
        # honour \\r (carriage-return) — discard everything buffered so
        # far on a \\r, keep only what follows the last \\r on each line.
        for ch in data:
            if ch == "\r":
                self._buf = ""       # overwrite: discard current line buffer
            elif ch == "\n":
                self._log_file.write(self._buf + "\n")
                self._log_file.flush()
                self._buf = ""
            else:
                self._buf += ch

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    # Forward any attribute access the underlying stream may need
    def __getattr__(self, attr):
        return getattr(self._stream, attr)


# ------------------------------------------------------------------
# Clone / import hszhao's official PSPNet (cloned into the project folder)
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
from my_pspnet import PSPNetScratch


# ====================================================================
# Evaluation metrics
# ====================================================================

def mean_iou(pred_flat, gt_flat, num_classes, ignore_idx=255):
    """
    Per-class IoU averaged over classes present in the batch.

    The PSPNet paper reports mIoU as the primary benchmark metric.
    Boundary pixels (label==ignore_idx) are excluded before scoring.
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
    Overall pixel accuracy — fraction of correctly labelled pixels,
    ignoring void/boundary pixels.
    """
    valid     = targets != ignore_idx
    correct   = (preds[valid] == targets[valid]).sum().item()
    total     = valid.sum().item()
    return correct / total if total > 0 else 0.0


# ====================================================================
# Learning-rate schedule
# ====================================================================

def poly_lr_scheduler(optimizer, total_steps, power=0.9):
    """
    Polynomial decay schedule used in the PSPNet paper.
        lr_t = base_lr * (1 - t / T) ^ power,  power = 0.9
    Implemented via LambdaLR for simplicity.
    """
    def _decay(step):
        return (1.0 - step / total_steps) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_decay)


# ====================================================================
# Core training loop
# ====================================================================

def train_one_model(model, train_loader, val_loader,
                    epochs=10, base_lr=0.01,
                    device="cpu", tag="model", scratch=False):
    """
    Train a segmentation model for `epochs` epochs and collect metrics.

    For the scratch model:
      - pretrained backbone params  →  lr = base_lr
      - newly added head/stem params →  lr = base_lr * 10

    Both models use SGD with poly decay.  The scratch model also applies
    an auxiliary loss (weight 0.4) from the stage-3 branch.
    """
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    # Build per-group optimizer
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
        # ----------------------------------------------------------
        # Training phase
        # ----------------------------------------------------------
        model.train()
        running_loss = 0.0

        bar = tqdm(train_loader, desc=f"[{tag}] Epoch {ep+1}/{epochs} Train")
        for imgs, masks in bar:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            if hasattr(model, "zoom_factor") and model.training:
                # hszhao PSPNet requires (H-1) % 8 == 0
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

        # ----------------------------------------------------------
        # Validation phase
        # ----------------------------------------------------------
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
# Plotting utilities
# ====================================================================

def plot_training_curves(log_scratch, log_official, num_epochs,
                         out_path="comparison_plots.png"):
    """
    Four-panel figure: train loss / val loss / pixel accuracy / mIoU,
    comparing scratch implementation with the official hszhao model.
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
        ax.plot(epoch_axis, log_scratch[key],  "o-",  label="Scratch PSPNet",          linewidth=2)
        ax.plot(epoch_axis, log_official[key], "s--", label="Official (hszhao) PSPNet", linewidth=2)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "PSPNet: Scratch vs Official Implementation Comparison",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plots to '{out_path}'")
    plt.close()


def show_qualitative_results(m_scratch, m_official, val_loader, device,
                             n_samples=4, out_path="qualitative_results.png"):
    """
    Side-by-side visualisation:
        [Input Image | Ground Truth | Scratch Pred | Official Pred]
    """
    m_scratch.eval()
    m_official.eval()

    imgs, masks = next(iter(val_loader))
    imgs_dev    = imgs.to(device)
    n_samples   = min(n_samples, imgs.size(0))

    with torch.no_grad():
        scratch_preds = torch.argmax(m_scratch(imgs_dev), dim=1).cpu()

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

    # Reverse ImageNet normalisation for display
    _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    col_labels = ["Input Image", "Ground Truth", "Scratch PSPNet", "Official (hszhao)"]

    for row in range(n_samples):
        rgb = (imgs[row] * _std + _mean).permute(1, 2, 0).clamp(0, 1).numpy()

        row_data = [
            rgb,
            masks[row].numpy(),
            scratch_preds[row].numpy(),
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
        "Qualitative Comparison: Scratch vs Official PSPNet",
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
    Average predictions across scales (and optionally horizontal flip)
    as described in the PSPNet paper (scales 0.5–1.75 in production).
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
    """Run multi-scale inference over the full validation set."""
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
# Summary table
# ====================================================================

def print_results_table(log_scratch, log_official,
                        ms_scratch=None, ms_official=None):
    """Tabulate final-epoch metrics for both models side by side."""
    print("\n" + "=" * 65)
    print("  FINAL METRICS COMPARISON (last epoch)")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Scratch PSPNet':>18} {'Official (hszhao)':>18}")
    print("-" * 65)

    rows = [
        ("Train Loss",     "train_loss"),
        ("Val Loss",       "val_loss"),
        ("Pixel Accuracy", "val_acc"),
        ("Mean IoU",       "val_miou"),
    ]
    for label, key in rows:
        s_val = log_scratch[key][-1]
        o_val = log_official[key][-1]
        print(f"  {label:<22} {s_val:>18.4f} {o_val:>18.4f}")

    if ms_scratch and ms_official:
        print("-" * 65)
        print(f"  {'MS Pixel Accuracy':<22} {ms_scratch[0]:>18.4f} {ms_official[0]:>18.4f}")
        print(f"  {'MS Mean IoU':<22} {ms_scratch[1]:>18.4f} {ms_official[1]:>18.4f}")

    print("=" * 65)


# ====================================================================
# Entry point
# ====================================================================

def main():
    # Open output.txt next to this script and mirror all output to it
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.txt")
    _log_fh   = open(_log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)
    print(f"All terminal output is also being saved to: {_log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyper-parameters (tuned for a 6 GB GPU)
    cfg = dict(
        epochs     = 15,
        batch_size = 2,        # 6 GB VRAM budget
        base_lr    = 0.01,
        num_train  = 500,
        num_val    = 100,
        crop_size  = 257,      # satisfies (x-1) % 8 == 0; fits in VRAM
        num_classes= VOC_NUM_CLASSES,
    )

    # ---- data ----
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
    # 1. Scratch PSPNet
    # ----------------------------------------------------------------
    print("\n>>> Initializing Scratch PSPNet with auxiliary branch...")
    model_scratch = PSPNetScratch(num_classes=cfg["num_classes"], use_aux=True)
    model_scratch, log_scratch = train_one_model(
        model_scratch, train_loader, val_loader,
        epochs  =cfg["epochs"],
        base_lr =cfg["base_lr"],
        device  =device,
        tag     ="Scratch PSPNet",
        scratch =True,
    )

    # ----------------------------------------------------------------
    # 2. Official hszhao PSPNet
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
    # 3. Multi-scale evaluation
    # ----------------------------------------------------------------
    print("\n>>> Running multi-scale inference (scales: 0.5-1.75 + flip)...")
    ms_scratch  = eval_multiscale(model_scratch,  val_loader, device, cfg["num_classes"])
    print(f"  Scratch  MS — Pixel Acc: {ms_scratch[0]:.4f}, mIoU: {ms_scratch[1]:.4f}")

    ms_official = eval_multiscale(model_official, val_loader, device, cfg["num_classes"])
    print(f"  Official MS — Pixel Acc: {ms_official[0]:.4f}, mIoU: {ms_official[1]:.4f}")

    # ----------------------------------------------------------------
    # 4. Visualisations
    # ----------------------------------------------------------------
    print("\n>>> Generating comparison plots...")
    plot_training_curves(log_scratch, log_official, cfg["epochs"])

    print("\n>>> Generating qualitative predictions...")
    show_qualitative_results(model_scratch, model_official, val_loader, device)

    print_results_table(log_scratch, log_official, ms_scratch, ms_official)

    # Persist weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model_scratch.state_dict(),  "checkpoints/my_pspnet.pth")
    torch.save(model_official.state_dict(), "checkpoints/pspnet_official.pth")
    print("\nSaved model checkpoints to 'checkpoints/'")


if __name__ == "__main__":
    main()
