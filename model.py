# model.py
# --------
# My PSPNet implementation based on the CVPR 2017 paper by Zhao et al.
# I basically read through the paper and tried to code up each part myself.
# The whole idea is that normal segmentation networks dont really "see" the
# bigger picture — like they might label a boat on a road because they only
# look at local patches. PSPNet adds this pyramid pooling thing that forces
# the network to also look at the whole image at once, which helps a lot.

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


def _load_dilated_resnet():
    """
    Grab a pretrained ResNet50 but modify it so the last two stages
    use dilated convolutions instead of strided ones.

    Why? Because the paper says the feature extractor should have an
    output stride of 8 (meaning the feature map is 1/8 of input size).
    Normal ResNet50 has stride 32 which throws away too much spatial info.
    The trick is to replace the stride-2 downsampling in layer3 and layer4
    with dilation instead — this keeps the resolution but widens the
    receptive field. Torchvision makes this easy with replace_stride_with_dilation.
    """
    backbone = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1,
        replace_stride_with_dilation=[False, True, True]
    )
    return backbone


class MyPSPNet(nn.Module):
    """
    Full PSPNet model.

    The pipeline goes:
        input image
            -> resnet stem (conv + pool, stride 4)
            -> layer1 (256ch)
            -> layer2 (512ch)
            -> layer3 (1024ch, dilated)  <-- aux head taps here
            -> layer4 (2048ch, dilated)
            -> PPM (2048 -> 4096ch)
            -> main classifier -> upsample to original size

    During training, the aux head also produces predictions from layer3
    output. The paper says this helps with gradient flow (deep supervision).
    """
    def __init__(self, num_classes=21, use_aux=False):
        super().__init__()
        self.use_aux = use_aux

        # load the backbone
        resnet = _load_dilated_resnet()

        # break it into stages so i can grab intermediate features
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.stage1 = resnet.layer1   # -> 256 ch
        self.stage2 = resnet.layer2   # -> 512 ch
        self.stage3 = resnet.layer3   # -> 1024 ch (dilated, no downsampling)
        self.stage4 = resnet.layer4   # -> 2048 ch (dilated, no downsampling)

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


# just a quick sanity check to make sure everything connects properly
if __name__ == '__main__':
    model = MyPSPNet(num_classes=21, use_aux=True)

    x = torch.randn(2, 3, 256, 256)

    model.train()
    main_out, aux_out = model(x)
    print(f"train mode: main={main_out.shape}, aux={aux_out.shape}")

    model.eval()
    with torch.no_grad():
        pred = model(x)
    print(f"eval mode: pred={pred.shape}")
    print("looks good!")
