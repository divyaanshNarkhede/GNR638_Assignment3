# run_training.py
# ----------------
# Main script that trains both my PSPNet and the SMP library version,
# then compares them. The goal is to see how close my implementation
# gets to the "official" one on the same data.
#
# I tried to follow the paper's training recipe:
#   - SGD with momentum 0.9, weight decay 1e-4
#   - poly LR schedule (lr decays as training progresses)
#   - separate LR for backbone vs new layers (10x higher for new stuff)
#   - auxiliary loss from an intermediate layer (weight=0.4)
#   - cross entropy loss, ignoring the 255 boundary label

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # no GUI needed
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

from model import MyPSPNet
from data_loader import build_loaders, N_CLASSES, IGNORE_LBL


# -----------------------------------------------
#  Metrics
# -----------------------------------------------

def calc_miou(pred, target, nclass, ignore=255):
    """
    Mean IoU — this is THE metric for segmentation.
    For each class: IoU = intersection / union
    Then average over classes that actually show up.
    """
    pred = pred.flatten()
    target = target.flatten()

    # throw out ignored pixels
    valid = (target != ignore)
    pred = pred[valid]
    target = target[valid]

    class_ious = []
    for c in range(nclass):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            class_ious.append(inter / union)

    return np.mean(class_ious) if class_ious else 0.0


def calc_pixel_acc(pred, target, ignore=255):
    """Simple accuracy — what fraction of valid pixels are correct?"""
    valid = (target != ignore)
    correct = (pred[valid] == target[valid]).sum().item()
    total = valid.sum().item()
    return correct / total if total > 0 else 0.0


# -----------------------------------------------
#  LR schedule (poly decay from the paper)
# -----------------------------------------------

def get_poly_scheduler(optim_obj, total_steps, power=0.9):
    """
    lr at step t = base_lr * (1 - t/T)^0.9
    The paper uses this everywhere. Its nice because the LR smoothly
    goes to zero by the end of training — no sudden drops.
    """
    def _lambda(step):
        return max((1.0 - step / total_steps) ** power, 0)
    return optim.lr_scheduler.LambdaLR(optim_obj, _lambda)


# -----------------------------------------------
#  Training loop
# -----------------------------------------------

def run_training(model, train_dl, val_dl, epochs=10, base_lr=0.01,
                 dev='cpu', label='model', mine=False):
    """
    Trains one model and returns it along with its training history.

    If mine=True, we use differential LR (backbone gets base_lr,
    PPM and heads get 10x that) and handle the auxiliary output.
    """
    model.to(dev)
    ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LBL)

    # setup optimizer — paper says new layers should learn faster
    if mine:
        new_layer_names = ['ppm', 'cls_head', 'aux_cls']
        backbone_p, new_p = [], []
        for n, p in model.named_parameters():
            if any(k in n for k in new_layer_names):
                new_p.append(p)
            else:
                backbone_p.append(p)
        groups = [
            {'params': backbone_p, 'lr': base_lr},
            {'params': new_p, 'lr': base_lr * 10}
        ]
    else:
        groups = [{'params': model.parameters(), 'lr': base_lr}]

    optimizer = optim.SGD(groups, momentum=0.9, weight_decay=1e-4)
    total_steps = epochs * len(train_dl)
    scheduler = get_poly_scheduler(optimizer, total_steps)

    # stuff to track
    log = {'train_loss': [], 'val_loss': [], 'acc': [], 'miou': []}

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  epochs={epochs}  lr={base_lr}  device={dev}")
    print(f"{'='*50}")

    for ep in range(epochs):
        # --- train ---
        model.train()
        running_loss = 0
        pbar = tqdm(train_dl, desc=f"[{label}] ep {ep+1}/{epochs}")
        for imgs, segs in pbar:
            imgs, segs = imgs.to(dev), segs.to(dev)
            optimizer.zero_grad()

            out = model(imgs)

            # my model returns a tuple (main, aux) during training
            if isinstance(out, tuple):
                main_pred, aux_pred = out
                loss = ce_loss(main_pred, segs) + 0.4 * ce_loss(aux_pred, segs)
            else:
                loss = ce_loss(out, segs)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss_avg = running_loss / len(train_dl)

        # --- validate ---
        model.eval()
        vloss_sum = 0
        pred_list, gt_list = [], []
        with torch.no_grad():
            for imgs, segs in val_dl:
                imgs, segs = imgs.to(dev), segs.to(dev)
                out = model(imgs)
                vloss_sum += ce_loss(out, segs).item()
                preds = out.argmax(1)
                pred_list.append(preds.cpu())
                gt_list.append(segs.cpu())

        val_loss_avg = vloss_sum / len(val_dl)
        all_pred = torch.cat(pred_list)
        all_gt = torch.cat(gt_list)
        acc = calc_pixel_acc(all_pred, all_gt, IGNORE_LBL)
        miou = calc_miou(all_pred, all_gt, N_CLASSES, IGNORE_LBL)

        log['train_loss'].append(train_loss_avg)
        log['val_loss'].append(val_loss_avg)
        log['acc'].append(acc)
        log['miou'].append(miou)

        print(f"  ep {ep+1}/{epochs}  "
              f"train_loss={train_loss_avg:.4f}  "
              f"val_loss={val_loss_avg:.4f}  "
              f"acc={acc:.4f}  miou={miou:.4f}")

    return model, log


# -----------------------------------------------
#  Post-training multi-scale evaluation
# -----------------------------------------------

def multiscale_eval(model, val_dl, dev, scales=(0.75, 1.0, 1.25)):
    """
    Runs multi-scale + flip inference on the entire val set.
    Returns pixel accuracy and mIoU.
    Only works for models that have multiscale_predict method.
    """
    model.eval()
    pred_list, gt_list = [], []
    print(f"  running multi-scale eval with scales={scales}...")
    with torch.no_grad():
        for imgs, segs in tqdm(val_dl, desc='MS eval'):
            imgs, segs = imgs.to(dev), segs.to(dev)
            if hasattr(model, 'multiscale_predict'):
                preds = model.multiscale_predict(imgs, scales=scales)
            else:
                # fallback to single scale if model doesnt support it
                preds = model(imgs).argmax(1)
            pred_list.append(preds.cpu())
            gt_list.append(segs.cpu())

    all_pred = torch.cat(pred_list)
    all_gt = torch.cat(gt_list)
    ms_acc = calc_pixel_acc(all_pred, all_gt, IGNORE_LBL)
    ms_miou = calc_miou(all_pred, all_gt, N_CLASSES, IGNORE_LBL)
    return ms_acc, ms_miou


# -----------------------------------------------
#  Plotting
# -----------------------------------------------

def plot_curves(log_mine, log_smp, epochs, savepath='comparison.png'):
    """Side by side curves: loss, accuracy, miou for both models."""
    ep_range = range(1, epochs + 1)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    stuff = [
        (ax[0,0], 'Training Loss', 'train_loss', 'Loss'),
        (ax[0,1], 'Validation Loss', 'val_loss', 'Loss'),
        (ax[1,0], 'Pixel Accuracy', 'acc', 'Accuracy'),
        (ax[1,1], 'Mean IoU', 'miou', 'mIoU'),
    ]
    for a, title, key, yl in stuff:
        a.plot(ep_range, log_mine[key], 'o-', label='My PSPNet', linewidth=2)
        a.plot(ep_range, log_smp[key], 's--', label='SMP PSPNet', linewidth=2)
        a.set_title(title)
        a.set_xlabel('Epoch')
        a.set_ylabel(yl)
        a.legend()
        a.grid(alpha=0.3)

    fig.suptitle('My PSPNet vs Official (SMP) PSPNet', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nsaved plot to {savepath}")


def show_samples(my_model, smp_model, val_dl, dev, n=4,
                 savepath='predictions.png'):
    """Visual comparison on a few val images."""
    my_model.eval()
    smp_model.eval()

    imgs, segs = next(iter(val_dl))
    n = min(n, imgs.size(0))
    with torch.no_grad():
        my_pred = my_model(imgs.to(dev)).argmax(1).cpu()
        smp_pred = smp_model(imgs.to(dev)).argmax(1).cpu()

    # undo normalization so we can actually see the images
    mean = torch.tensor([.485, .456, .406]).view(3,1,1)
    std = torch.tensor([.229, .224, .225]).view(3,1,1)

    fig, axes = plt.subplots(n, 4, figsize=(15, 4*n))
    titles = ['Image', 'Ground Truth', 'My PSPNet', 'SMP PSPNet']

    for i in range(n):
        disp = (imgs[i] * std + mean).permute(1,2,0).clamp(0,1).numpy()
        axes[i,0].imshow(disp)
        axes[i,1].imshow(segs[i].numpy(), cmap='tab20', vmin=0, vmax=20)
        axes[i,2].imshow(my_pred[i].numpy(), cmap='tab20', vmin=0, vmax=20)
        axes[i,3].imshow(smp_pred[i].numpy(), cmap='tab20', vmin=0, vmax=20)
        for j in range(4):
            axes[i,j].axis('off')
            if i == 0:
                axes[i,j].set_title(titles[j])

    fig.suptitle('Qualitative Comparison', fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved predictions to {savepath}")


def print_results(log_mine, log_smp, ms_mine=None, ms_smp=None):
    """Quick summary table at the end."""
    print("\n" + "="*65)
    print("  Final Results (last epoch)")
    print("="*65)
    print(f"  {'Metric':30s} {'Mine':>14s} {'SMP':>14s}")
    print("-"*65)
    for name, k in [('Train Loss', 'train_loss'), ('Val Loss', 'val_loss'),
                    ('Pixel Acc', 'acc'), ('mIoU', 'miou')]:
        m = log_mine[k][-1]
        s = log_smp[k][-1]
        print(f"  {name:30s} {m:14.4f} {s:14.4f}")
    # multi-scale rows
    if ms_mine is not None and ms_smp is not None:
        ms_acc_m, ms_miou_m = ms_mine
        ms_acc_s, ms_miou_s = ms_smp
        print(f"  {'MS Pixel Accuracy':30s} {ms_acc_m:14.4f} {ms_acc_s:14.4f}")
        print(f"  {'MS Mean IoU':30s} {ms_miou_m:14.4f} {ms_miou_s:14.4f}")
    print("="*65)


# -----------------------------------------------
#  main
# -----------------------------------------------

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {dev}")

    # hyperparams — paper uses 473 crops but our GPU (6GB) cant handle that
    # so we use 256 with batch size 2 to stay within VRAM limits
    epochs = 15
    bs = 2
    lr = 0.01         # paper's base LR
    n_train = 500
    n_val = 50
    crop = 256

    print("\nloading data...")
    train_dl, val_dl = build_loaders(
        data_path='./data', bs=bs, n_train=n_train,
        n_val=n_val, crop_sz=crop
    )
    print(f"  train: {len(train_dl.dataset)} images")
    print(f"  val:   {len(val_dl.dataset)} images")

    # ---- 1) my pspnet ----
    print("\n>> building my pspnet...")
    my_net = MyPSPNet(num_classes=N_CLASSES, use_aux=True)
    my_net, my_log = run_training(
        my_net, train_dl, val_dl,
        epochs=epochs, base_lr=lr, dev=dev,
        label='My PSPNet', mine=True
    )

    # ---- 2) smp pspnet (official-ish baseline) ----
    print("\n>> building smp pspnet...")
    smp_net = smp.PSPNet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=N_CLASSES,
    )
    smp_net, smp_log = run_training(
        smp_net, train_dl, val_dl,
        epochs=epochs, base_lr=lr, dev=dev,
        label='SMP PSPNet', mine=False
    )

    # ---- 3) multi-scale evaluation (paper reports these numbers) ----
    print("\n>> running multi-scale evaluation...")
    ms_mine = multiscale_eval(my_net, val_dl, dev, scales=(0.75, 1.0, 1.25))
    ms_smp = multiscale_eval(smp_net, val_dl, dev, scales=(0.75, 1.0, 1.25))
    print(f"  my pspnet  — MS acc: {ms_mine[0]:.4f}, MS mIoU: {ms_mine[1]:.4f}")
    print(f"  smp pspnet — MS acc: {ms_smp[0]:.4f}, MS mIoU: {ms_smp[1]:.4f}")

    # ---- 4) compare ----
    print("\n>> generating plots...")
    plot_curves(my_log, smp_log, epochs)

    print(">> generating sample predictions...")
    show_samples(my_net, smp_net, val_dl, dev)

    print_results(my_log, smp_log, ms_mine=ms_mine, ms_smp=ms_smp)

    # save the weights just in case
    os.makedirs('saved_models', exist_ok=True)
    torch.save(my_net.state_dict(), 'saved_models/my_pspnet.pth')
    torch.save(smp_net.state_dict(), 'saved_models/smp_pspnet.pth')
    print("\nweights saved to saved_models/")


if __name__ == '__main__':
    main()
