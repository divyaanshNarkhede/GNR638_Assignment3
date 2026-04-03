"""
PSPNet built from scratch — Pyramid Scene Parsing Network.

Based on: Zhao et al., "Pyramid Scene Parsing Network", CVPR 2017.

Standard FCNs lose global context because they only see local receptive
fields.  PSPNet fixes this with a Pyramid Pooling Module (PPM) that pools
the feature map at four coarse grids, projects each to a compact
representation, then fuses the up-sampled context back with the original
features before final classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ====================================================================
# Shared building blocks
# ====================================================================

def _cbr(in_ch, out_ch, kernel, padding=0, bias=False):
    """Conv -> BatchNorm -> ReLU convenience wrapper."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=bias),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _seg_classifier(in_ch, bottleneck_ch, num_cls, dropout=0.1):
    """
    Shared design for the main and auxiliary segmentation heads:
      3x3 conv -> BN -> ReLU -> spatial dropout -> 1x1 classifier
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, bottleneck_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(bottleneck_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout),
        nn.Conv2d(bottleneck_ch, num_cls, kernel_size=1),
    )


# ====================================================================
# Pyramid Pooling Module components
# ====================================================================

class _PyramidLevel(nn.Module):
    """
    A single level of the PPM: pool to a fixed grid size, compress
    channels with a 1x1 conv, then bilinearly upsample back to the
    input spatial dimensions.
    """

    def __init__(self, in_ch, out_ch, pool_grid):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=pool_grid)
        self.compress     = _cbr(in_ch, out_ch, kernel=1)

    def forward(self, feat):
        h_in, w_in = feat.shape[2], feat.shape[3]
        ctx = self.global_pool(feat)
        ctx = self.compress(ctx)
        ctx = F.interpolate(ctx, size=(h_in, w_in),
                            mode="bilinear", align_corners=True)
        return ctx


class PyramidPoolingModule(nn.Module):
    """
    PPM — the centrepiece of PSPNet.

    Four pooling levels (1x1, 2x2, 3x3, 6x6) each project the
    2048-channel backbone output down to `branch_ch` channels.
    All four context maps are upsampled and concatenated with the
    original feature map, yielding:
        out_channels = in_ch + num_levels * branch_ch
    """

    def __init__(self, in_ch, branch_ch, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.levels = nn.ModuleList(
            [_PyramidLevel(in_ch, branch_ch, ps) for ps in pool_sizes]
        )
        self.out_channels = in_ch + branch_ch * len(pool_sizes)

    def forward(self, feat):
        ctx_maps = [lvl(feat) for lvl in self.levels]
        return torch.cat([feat] + ctx_maps, dim=1)


# ====================================================================
# Modified ResNet backbone
# ====================================================================

class DeepStem(nn.Module):
    """
    Paper-style entry block: three stacked 3x3 convolutions replace
    the single 7x7 conv in vanilla ResNet.  Produces 128 channels and
    downsamples 4x (two stride-2 ops: first conv + max-pool).

    Channel sequence: 3 -> 64 -> 64 -> 128 -> max-pool
    """

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3,   64,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  64,  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.block(x)


def _dilated_resnet50():
    """
    ImageNet-pretrained ResNet-50 with dilated convolutions in the
    last two residual stages, keeping the output stride at 8 rather
    than the default 32.  Spatial resolution is preserved for the PPM.
    """
    return resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V1,
        replace_stride_with_dilation=[False, True, True],
    )


# ====================================================================
# Full PSPNet
# ====================================================================

class PSPNetScratch(nn.Module):
    """
    From-scratch PSPNet implementation.

    Forward path:
        input -> DeepStem -> ResNet stages 1-4 (dilated) -> PPM -> seg head -> output
                                      |
                               stage-3 features -> aux head  (train only)

    The deep stem outputs 128 channels, but pretrained layer1 expects 64.
    We patch the first Bottleneck of layer1 (both its 1x1 conv and the
    skip-connection downsampler) to accept 128 channels — matching the
    strategy used in the hszhao reference implementation.
    """

    def __init__(self, num_classes=21, use_aux=False):
        super().__init__()
        self.use_aux = use_aux

        # --- backbone ---
        base = _dilated_resnet50()

        self.stem = DeepStem()

        # Patch layer1's first bottleneck to accept 128-ch stem output
        base.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        base.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=1, bias=False)

        self.stage1 = base.layer1   # 128 -> 256  ch
        self.stage2 = base.layer2   # 256 -> 512  ch
        self.stage3 = base.layer3   # 512 -> 1024 ch  (dilated)
        self.stage4 = base.layer4   # 1024-> 2048 ch  (dilated)

        # --- pyramid pooling: 2048 in, 512 per branch, 4096 total out ---
        self.ppm = PyramidPoolingModule(
            in_ch=2048, branch_ch=512, pool_sizes=(1, 2, 3, 6)
        )

        # --- main head: 4096 -> 512 -> num_classes ---
        self.head = _seg_classifier(
            in_ch=self.ppm.out_channels,
            bottleneck_ch=512,
            num_cls=num_classes,
            dropout=0.1,
        )

        # --- auxiliary head taps off stage-3 features ---
        if self.use_aux:
            self.aux_head = _seg_classifier(
                in_ch=1024,
                bottleneck_ch=256,
                num_cls=num_classes,
                dropout=0.1,
            )

        self._init_weights()

    # ----------------------------------------------------------------
    # Weight initialisation
    # ----------------------------------------------------------------

    def _init_weights(self):
        """Kaiming-normal for conv weights; ones/zeros for BN in new modules."""
        new_modules = [self.stem, self.ppm, self.head]
        if self.use_aux:
            new_modules.append(self.aux_head)

        for module in new_modules:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ----------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]

        # Stem + backbone
        f = self.stem(x)
        f = self.stage1(f)
        f = self.stage2(f)
        f3 = self.stage3(f)       # stage-3 output  (for aux branch)
        f4 = self.stage4(f3)

        # Pyramid pooling + classification head
        f_ppm = self.ppm(f4)
        out = self.head(f_ppm)
        out = F.interpolate(out, size=(input_h, input_w),
                            mode="bilinear", align_corners=True)

        # Auxiliary prediction (training only)
        if self.training and self.use_aux:
            aux_out = self.aux_head(f3)
            aux_out = F.interpolate(aux_out, size=(input_h, input_w),
                                    mode="bilinear", align_corners=True)
            return out, aux_out

        return out


# ====================================================================
# Smoke test
# ====================================================================
if __name__ == "__main__":
    net = PSPNetScratch(num_classes=21, use_aux=True)

    net.train()
    sample = torch.randn(2, 3, 473, 473)
    main, aux = net(sample)
    print(f"[train] main: {main.shape}, aux: {aux.shape}")

    net.eval()
    with torch.no_grad():
        pred = net(sample)
    print(f"[eval]  pred: {pred.shape}")
