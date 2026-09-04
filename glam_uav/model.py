"""
GLAM-UAV — UAV Imagery Real-Time Semantic Segmentation with Global-Local Information
Attention  (Sensors 2025, 25(6), 1786).  https://doi.org/10.3390/s25061786

*** UNOFFICIAL reimplementation from the paper (Fig 2-4, Eq 1-13). No code was released. ***

Verified against the full paper:
  - Encoder: ResNet18, 4 stages at strides 4/8/16/32 (64/128/256/512 ch).
  - GLAM global branch = Coordinate Attention (Hou et al. 2021), Eq 1-6 exactly:
    X/Y-axis avg-pool -> concat -> 1x1 reduce to C/r -> BN + non-linear -> split ->
    per-axis 1x1 -> sigmoid gates -> Y = X (.) Wx (.) Wy   (channel-preserving).
  - GLAM local branch, Eq 7-9: A = BN(Conv3x3(X)), B = BN(Conv1x1(X)), Z = A + B.
  - GLAM fusion, Eq 10: Output = BN(DWConv3x3(Y + Z)).  GLAM preserves the channel count.
  - SLFM (Fig 4): align f1..f4 to stride-4 (upsample + 1x1 conv); per-scale channel
    descriptor = Sigmoid(1x1(GAP(.)));  softmax across the 4 scales;  weighted sum;
    1x1 conv -> fused feature; added into the top decoder level only.
  - Training target: plain cross-entropy, UAVid 8 classes, 1024x1024 crops.
  - Reported: 12.1 M params, 48.24 GFLOPs, 72.4 FPS @ 1024^2 (RTX 3090).

Underspecified in the paper (choices here — may differ from the authors' model):
  - Decoder channel widths: never tabulated. The whole decoder is only ~0.9 M params
    (12.1 M total - 11.2 M for ResNet18), so GLAM cannot run at 256/512; here f4 is
    1x1-reduced to 128 and the decoder runs at 128 / 128 / 64 (s32 / s16 / s8).
  - Skip fusion done as concat + 1x1 (UNet-style, "the most straightforward connection
    operation"); plain addition is an equally valid reading.
  - A trailing ReLU is kept after Eq-10 fusion for trainability (Eq 10 itself is BN only).
  - CoordAttention reduction ratio r = 32 (the CVPR-2021 default; paper says only "C/r").

    from glam_uav import glam_uav
    model = glam_uav(num_classes=8)
    logits = model(images)                     # (B, num_classes, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(i, o, k=1, s=1, g=1):
    return nn.Sequential(
        nn.Conv2d(i, o, k, s, k // 2, groups=g, bias=False), nn.BatchNorm2d(o)
    )


def conv_bn_relu(i, o, k=1, s=1):
    return nn.Sequential(
        nn.Conv2d(i, o, k, s, k // 2, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------- ResNet18
class BasicBlock(nn.Module):
    def __init__(self, inp, out, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + idt)


class ResNet18(nn.Module):
    """Stem (stride 4) + layer1..4 -> features at strides 4/8/16/32, ch
    64/128/256/512."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.inp = 64
        self.layer1 = self._make(64, 2, 1)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)

    def _make(self, out, n, stride):
        ds = (
            conv_bn(self.inp, out, 1, stride)
            if (stride != 1 or self.inp != out)
            else None
        )
        layers = [BasicBlock(self.inp, out, stride, ds)]
        self.inp = out
        for _ in range(1, n):
            layers.append(BasicBlock(out, out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f1 = self.layer1(x)  # 64,  s4
        f2 = self.layer2(f1)  # 128, s8
        f3 = self.layer3(f2)  # 256, s16
        f4 = self.layer4(f3)  # 512, s32
        return f1, f2, f3, f4


# ---------------------------------------------------------------- GLAM
class CoordAtt(nn.Module):
    """GLAM global branch = Coordinate Attention (Eq 1-6).

    Channel-preserving reweight.
    """

    def __init__(self, ch, reduction=32):
        super().__init__()
        mid = max(8, ch // reduction)
        self.reduce = nn.Conv2d(ch, mid, 1)  # Eq 3
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)  # Eq 4 "non-linear"
        self.conv_h = nn.Conv2d(mid, ch, 1)  # Eq 5
        self.conv_w = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        xh = x.mean(dim=3, keepdim=True)  # (b,c,h,1)  Eq 1
        xw = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (b,c,w,1)  Eq 2
        y = self.act(self.bn(self.reduce(torch.cat([xh, xw], dim=2))))  # Eq 3-4
        yh, yw = torch.split(y, [h, w], dim=2)
        ah = torch.sigmoid(self.conv_h(yh))  # Eq 5
        aw = torch.sigmoid(self.conv_w(yw.permute(0, 1, 3, 2)))
        return x * ah * aw  # Eq 6


class GLAM(nn.Module):
    """Global-Local Attention Module (Eq 1-10).

    Channel-preserving (in == out).
    """

    def __init__(self, ch, reduction=32):
        super().__init__()
        self.ca = CoordAtt(ch, reduction)
        self.local_3x3 = conv_bn(ch, ch, 3)  # Eq 7
        self.local_1x1 = conv_bn(ch, ch, 1)  # Eq 8
        self.fuse = conv_bn(ch, ch, 3, g=ch)  # Eq 10: BN(DWConv3x3(.))
        self.act = nn.ReLU(inplace=True)  # kept for trainability

    def forward(self, x):
        y = self.ca(x)  # Eq 6
        z = self.local_3x3(x) + self.local_1x1(x)  # Eq 9
        return self.act(self.fuse(y + z))  # Eq 10


# ---------------------------------------------------------------- SLFM
class SLFMBlock(nn.Module):
    """Fig 4 'Block': per-scale channel descriptor = Sigmoid(1x1(GAP(x)))."""

    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x.mean(dim=(2, 3), keepdim=True)))  # (b,ch,1,1)


class SLFM(nn.Module):
    """Shallow Layer Feature Fusion (Fig 4): align f1..f4 to stride-4, softmax-weight
    across the 4 scales, weighted sum, 1x1 conv -> fused feature."""

    def __init__(self, in_chs, ch=64):
        super().__init__()
        self.align = nn.ModuleList(conv_bn_relu(c, ch, 1) for c in in_chs)
        self.block = nn.ModuleList(SLFMBlock(ch) for _ in in_chs)
        self.out = conv_bn_relu(ch, ch, 1)

    def forward(self, feats):
        size = feats[0].shape[-2:]
        aligned = []
        for i, f in enumerate(feats):
            a = self.align[i](f)
            if a.shape[-2:] != size:
                a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
            aligned.append(a)
        w = torch.stack(
            [self.block[i](a) for i, a in enumerate(aligned)], dim=1
        )  # (b,N,ch,1,1)
        w = torch.softmax(w, dim=1)
        fused = sum(w[:, i] * aligned[i] for i in range(len(aligned)))
        return self.out(fused)


# ---------------------------------------------------------------- full model
class GLAMUAV(nn.Module):
    """UNet-style: ResNet18 encoder + (3x GLAM + SLFM) decoder.

    ~11.8 M params (paper 12.1 M); decoder runs at 128 / 128 / 64 — see module
    docstring.
    """

    DEC = (128, 128, 64)  # decoder widths at s32 / s16 / s8

    def __init__(self, num_classes=8):
        super().__init__()
        self.enc = ResNet18()
        c1, c2, c3, c4 = 64, 128, 256, 512
        w32, w16, w8 = self.DEC

        self.reduce4 = conv_bn_relu(c4, w32, 1)
        self.glam1 = GLAM(w32)

        self.reduce3 = conv_bn_relu(c3, w16, 1)
        self.fuse16 = conv_bn_relu(w32 + w16, w16, 1)
        self.glam2 = GLAM(w16)

        self.reduce2 = conv_bn_relu(c2, w8, 1)
        self.fuse8 = conv_bn_relu(w16 + w8, w8, 1)
        self.glam3 = GLAM(w8)

        self.slfm = SLFM([c1, c2, c3, c4], ch=w8)
        self.head = nn.Conv2d(w8, num_classes, 1)

    def forward(self, x):
        f1, f2, f3, f4 = self.enc(x)

        d = self.glam1(self.reduce4(f4))  # w32, s32
        d = F.interpolate(d, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        d = self.glam2(self.fuse16(torch.cat([d, self.reduce3(f3)], 1)))  # w16, s16
        d = F.interpolate(d, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.glam3(self.fuse8(torch.cat([d, self.reduce2(f2)], 1)))  # w8,  s8
        d = F.interpolate(d, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        d = d + self.slfm([f1, f2, f3, f4])  # w8,  s4
        out = self.head(d)
        return F.interpolate(
            out, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


def glam_uav(num_classes=8):
    return GLAMUAV(num_classes=num_classes)


if __name__ == "__main__":
    m = glam_uav(num_classes=8).eval()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad():
        y = m(torch.randn(1, 3, 1024, 1024))
    print(
        f"glam_uav: {n:.2f}M params, out {tuple(y.shape)}  (paper: 12.1M, 48.24 GFLOPs)"
    )
