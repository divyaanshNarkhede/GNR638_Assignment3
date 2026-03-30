# model.py
# --------
# My PSPNet implementation based on the CVPR 2017 paper by Zhao et al.
# I read through the paper and coded up each part myself.
# The whole idea is that normal segmentation networks dont really "see" the
# bigger picture — like they might label a boat on a road because they only
# look at local patches. PSPNet adds this pyramid pooling thing that forces
# the network to also look at the whole image at once, which helps a lot.
#
# Key things from the paper that I implemented:
#   - Deep stem (three 3x3 convs instead of one 7x7) — Section 3.1
#   - Dilated ResNet50 backbone (output stride 8)
#   - Pyramid Pooling Module with bin sizes 1,2,3,6 — Section 3.2
#   - Auxiliary loss from layer3 features — Section 3.3
#   - Multi-scale inference with flipping — Section 4.1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CBR(nn.Module):
    """Conv -> BatchNorm -> ReLU. Got tired of writing this out every time lol"""
    def __init__(self, in_c, out_c, ks, stride=1, pad=0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, ks, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)


class SinglePoolBin(nn.Module):
    """
    Handles one bin size in the pyramid pooling.
    So for example if bin_size=6, this pools the feature map down to 6x6,
    runs a 1x1 conv to cut the channels, then upsamples back to original size.
    Pretty straightforward once you get the idea.
    """
    def __init__(self, in_c, out_c, bin_sz):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(bin_sz)
        self.conv = CBR(in_c, out_c, ks=1)  # 1x1 to reduce channels

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        y = self.pool(x)
        y = self.conv(y)
        # scale it back up so we can concat later
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


class PPM(nn.Module):
    """
    Pyramid Pooling Module — the main contribution of PSPNet.

    The paper (fig. 3) shows 4 different pooling levels: 1x1, 2x2, 3x3, 6x6.
    Each one captures context at a different granularity:
      - 1x1 gives you the global average (coarsest)
      - 6x6 is more local but still bigger than typical conv receptive fields

    After pooling + 1x1 conv + upsample for each level, we just concatenate
    everything together with the original features. So the output channels
    become: original_channels + (num_bins * reduced_channels)
    """
    def __init__(self, in_c, reduced_c=512, bins=(1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList([
            SinglePoolBin(in_c, reduced_c, b) for b in bins
        ])
        # keep track of how many channels come out after concat
        self.out_c = in_c + reduced_c * len(bins)

    def forward(self, x):
        pooled = [branch(x) for branch in self.branches]
        # stick the original features in front, then all the pooled ones
        return torch.cat([x] + pooled, dim=1)


class Classifier(nn.Module):
    """
    The segmentation head — takes features and outputs class predictions.
    Its just: 3x3 conv -> bn -> relu -> dropout -> 1x1 conv
    Nothing fancy here, same structure for both main and auxiliary heads.
    """
    def __init__(self, in_c, hidden_c, num_cls, drop=0.1):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Conv2d(in_c, hidden_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
            nn.Conv2d(hidden_c, num_cls, 1)   # final 1x1 to get class scores
        )

    def forward(self, x):
        return self.pipe(x)


class DeepStem(nn.Module):
    """
    The paper mentions using a modified ResNet with three 3x3 convs
    in the stem instead of the standard single 7x7 conv.
    This is what they call the "deep stem" or "resnet-v1c" variant.
    Channels go: 3 -> 64 -> 64 -> 128, with BN+ReLU after each.
    Then a 3x3 maxpool to get stride 4 total (same as normal ResNet).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x


def _load_dilated_resnet():
    """
    Grab a pretrained ResNet50 but modify it so the last two stages
    use dilated convolutions instead of strided ones.

    The paper says the feature extractor should give output stride = 8.
    Normal ResNet50 has stride 32 which loses too much spatial info.
    The trick is to replace stride-2 downsampling in layer3 and layer4
    with dilation — keeps resolution but widens receptive field.
    """
    backbone = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1,
        replace_stride_with_dilation=[False, True, True]
    )
    return backbone


def _init_stem_from_pretrained(deep_stem, resnet_conv1_weight):
    """
    Try to initialize the deep stem's first conv from the pretrained 7x7.
    We take the center 3x3 patch of the 7x7 kernel as a rough initialization
    for conv1. The other two convs get kaiming init.
    Not perfect but better than fully random — the paper probably trained
    the deep stem from scratch on ImageNet but we dont have time for that.
    """
    # the pretrained conv1 is [64, 3, 7, 7] — crop center 3x3
    w = resnet_conv1_weight
    center = w[:, :, 2:5, 2:5].clone()  # [64, 3, 3, 3]
    deep_stem.conv1.weight.data.copy_(center)

    # kaiming for the other two convs
    nn.init.kaiming_normal_(deep_stem.conv2.weight, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_normal_(deep_stem.conv3.weight, mode='fan_out', nonlinearity='relu')

    # batchnorm init
    for bn in [deep_stem.bn1, deep_stem.bn2, deep_stem.bn3]:
        nn.init.ones_(bn.weight)
        nn.init.zeros_(bn.bias)


class MyPSPNet(nn.Module):
    """
    Full PSPNet model — now with the deep stem from the paper.

    Pipeline:
        input image
            -> deep stem: 3x3 conv(64) -> 3x3 conv(64) -> 3x3 conv(128) -> pool
            -> layer1 (256ch)
            -> layer2 (512ch)
            -> layer3 (1024ch, dilated)  <-- aux head taps here
            -> layer4 (2048ch, dilated)
            -> PPM (2048 -> 4096ch)
            -> main classifier -> upsample to original size

    Note: layer1 expects 128 input channels now (from deep stem's conv3)
    instead of 64 from the normal stem. So we need to adjust that.
    """
    def __init__(self, num_classes=21, use_aux=False):
        super().__init__()
        self.use_aux = use_aux

        # load pretrained resnet for the residual stages + weight init
        resnet = _load_dilated_resnet()

        # use our deep stem instead of the standard 7x7 conv
        self.stem = DeepStem()
        _init_stem_from_pretrained(self.stem, resnet.conv1.weight.data)

        # layer1 normally expects 64ch input (from standard stem)
        # our deep stem outputs 128ch, so we need to fix the first conv in layer1
        # layer1's first block has a downsample[0] conv that goes 64->256
        # and the block's conv1 that goes 64->64. We need to change those to accept 128.
        old_l1 = resnet.layer1
        first_block = old_l1[0]

        # fix the main path: conv1 input channels 64 -> 128
        old_conv1 = first_block.conv1
        new_conv1 = nn.Conv2d(128, old_conv1.out_channels,
                              kernel_size=old_conv1.kernel_size,
                              stride=old_conv1.stride,
                              padding=old_conv1.padding, bias=False)
        nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
        first_block.conv1 = new_conv1

        # fix the skip connection: downsample conv 64 -> 256 becomes 128 -> 256
        old_ds = first_block.downsample[0]
        new_ds = nn.Conv2d(128, old_ds.out_channels,
                           kernel_size=old_ds.kernel_size,
                           stride=old_ds.stride, bias=False)
        nn.init.kaiming_normal_(new_ds.weight, mode='fan_out', nonlinearity='relu')
        first_block.downsample[0] = new_ds

        self.stage1 = old_l1           # -> 256 ch
        self.stage2 = resnet.layer2   # -> 512 ch
        self.stage3 = resnet.layer3   # -> 1024 ch (dilated)
        self.stage4 = resnet.layer4   # -> 2048 ch (dilated)

        # pyramid pooling on top of stage4 features
        self.ppm = PPM(in_c=2048, reduced_c=512, bins=(1, 2, 3, 6))

        # main classification head (takes PPM output = 2048 + 4*512 = 4096)
        self.cls_head = Classifier(self.ppm.out_c, 512, num_classes, drop=0.1)

        # auxiliary head — branches off from stage3 for deep supervision
        if use_aux:
            self.aux_cls = Classifier(1024, 256, num_classes, drop=0.1)

        # init weights for the new layers (backbone is already pretrained)
        self._weight_init()

    def _weight_init(self):
        """Kaiming init for the new conv layers, standard init for batchnorm."""
        to_init = [self.ppm, self.cls_head]
        if self.use_aux:
            to_init.append(self.aux_cls)

        for group in to_init:
            for m in group.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        orig_size = (x.shape[2], x.shape[3])

        # push through backbone stages
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        s3_feat = self.stage3(x)        # need this for aux
        s4_feat = self.stage4(s3_feat)

        # pyramid pooling to get global context
        enriched = self.ppm(s4_feat)

        # get per-pixel predictions and resize back
        out = self.cls_head(enriched)
        out = F.interpolate(out, size=orig_size, mode='bilinear',
                            align_corners=True)

        # aux branch only runs during training
        if self.training and self.use_aux:
            aux_out = self.aux_cls(s3_feat)
            aux_out = F.interpolate(aux_out, size=orig_size, mode='bilinear',
                                    align_corners=True)
            return out, aux_out

        return out

    def multiscale_predict(self, x, scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75)):
        """
        Multi-scale inference as described in the paper (Section 4.1).
        They say they average predictions across multiple scales AND
        their horizontal flips. This boosts mIoU by a decent amount.

        For each scale:
          1. Resize input to that scale
          2. Get prediction
          3. Also get prediction on horizontally flipped input
          4. Average all of them

        Kinda slow but gives better results — the paper always reports
        numbers with multi-scale testing.
        """
        self.eval()
        H, W = x.shape[2], x.shape[3]
        n_classes = self.cls_head.pipe[-1].out_channels  # grab from the 1x1 conv

        total_probs = torch.zeros(x.shape[0], n_classes, H, W, device=x.device)

        for s in scales:
            sH, sW = int(H * s), int(W * s)
            # make sure dimensions are valid
            if sH < 32 or sW < 32:
                continue

            scaled = F.interpolate(x, size=(sH, sW), mode='bilinear',
                                   align_corners=True)

            with torch.no_grad():
                # normal prediction
                logits = self.forward(scaled)
                probs = F.softmax(logits, dim=1)
                probs = F.interpolate(probs, size=(H, W), mode='bilinear',
                                      align_corners=True)
                total_probs += probs

                # flipped prediction — flip, predict, flip back
                flipped = torch.flip(scaled, dims=[3])
                logits_f = self.forward(flipped)
                probs_f = F.softmax(logits_f, dim=1)
                probs_f = torch.flip(probs_f, dims=[3])  # flip back
                probs_f = F.interpolate(probs_f, size=(H, W), mode='bilinear',
                                        align_corners=True)
                total_probs += probs_f

        # average over all (scales * 2) predictions
        total_probs /= (2 * len(scales))
        return total_probs.argmax(dim=1)


# just a quick sanity check to make sure everything connects properly
if __name__ == '__main__':
    model = MyPSPNet(num_classes=21, use_aux=True)

    x = torch.randn(2, 3, 473, 473)

    model.train()
    main_out, aux_out = model(x)
    print(f"train mode: main={main_out.shape}, aux={aux_out.shape}")

    model.eval()
    with torch.no_grad():
        pred = model(x)
    print(f"eval mode: pred={pred.shape}")

    # test multi-scale inference
    ms_pred = model.multiscale_predict(x, scales=(0.75, 1.0, 1.25))
    print(f"multiscale pred: {ms_pred.shape}")
    print("looks good!")
