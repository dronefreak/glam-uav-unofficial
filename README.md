# GLAM-UAV: unofficial PyTorch reimplementation

> ⚠️ **Unofficial. Not affiliated with the authors. No trained weights (random init).** See [`DISCLAIMER.md`](DISCLAIMER.md).

A single-file, dependency-light (`torch` only) reimplementation of:

**UAV Imagery Real-Time Semantic Segmentation with Global–Local Information Attention**
Zikang Zhang, Gongquan Li. *Sensors 2025, 25(6), 1786*. doi:[10.3390/s25061786](https://doi.org/10.3390/s25061786)
→ [paper](https://www.mdpi.com/1424-8220/25/6/1786)

Intended for latency / parameter benchmarking and as a starting point for training.
train it to get the paper's accuracy (no weights included).

## Install

```bash
pip install -e .        # or: just copy the glam_uav/ folder into your project
```

## Use

```python
import torch
from glam_uav import glam_uav

model = glam_uav(num_classes=8)     # random init
logits = model(torch.randn(1, 3, 1024, 1024))  # (B, num_classes, H, W), at input resolution
```

Factory: `glam_uav`. Returns logits at the input resolution.

Smoke test: `python test_forward.py` · ONNX export: `python export_onnx.py --help`

## What is faithful / what is a guess

**Confidence: Medium–High.** The full paper (text + Fig. 2–4 + Eq. 1–13) was worked through
page by page. The encoder and the two named modules are implemented as specified; the only
free choices are the decoder widths (and those are tightly bounded; see below).

**Implemented as specified in the paper:**
- **Encoder:** ResNet-18, four stages at strides 4 / 8 / 16 / 32 (64 / 128 / 256 / 512 ch).
- **GLAM global branch = Coordinate Attention** (Eq. 1–6, exact): H- and W-axis average
  pooling → concat → 1×1 reduction to `C/r` → BN + non-linear → split → per-axis 1×1 →
  sigmoid gates → `Y = X ⊙ W_x ⊙ W_y`. **Channel-preserving** (it is a reweighting of `X`).
- **GLAM local branch** (Eq. 7–9): `A = BN(Conv3×3(X))`, `B = BN(Conv1×1(X))`, `Z = A + B`.
- **GLAM fusion** (Eq. 10): `Out = BN(DWConv3×3(Y + Z))`. GLAM as a whole preserves the
  channel count; it does **not** reduce channels.
- **SLFM** (Fig. 4): align encoder stages 1–4 to stride-4; per-scale channel descriptor
  `Sigmoid(1×1(GAP(·)))`; softmax across the four scales; weighted sum; 1×1 conv → fused
  feature. Added into the top decoder level.
- **Decoder:** three stacked GLAM blocks with skip connections from the encoder, bilinear
  upsampling between levels, 1×1 seg head, bilinear ×4 to input resolution.
- Training target in the paper: plain cross-entropy, UAVid 8 classes, 1024×1024 crops.

**Choices where the paper is underspecified (may differ from the authors' model):**
- **Decoder widths.** The paper never tabulates them, but the whole decoder is only
  ~0.9 M params (12.1 M total − 11.2 M for ResNet-18), which rules out running GLAM at
  256/512. Here `f4` is 1×1-reduced to 128 and the decoder runs at **128 / 128 / 64**
  (s32 / s16 / s8). Result: **11.79 M params** vs the paper's 12.1 M.
- **Skip fusion** is done as concat + 1×1 (UNet-style; the paper says only "the most
  straightforward connection operation"); plain addition is an equally valid reading.
- A trailing ReLU is kept after the Eq.-10 fusion for trainability (Eq. 10 itself is BN only).
- CoordAttention reduction ratio `r = 32` (the CVPR-2021 default; the paper says only `C/r`).
- ResNet-18 is random-initialised here (the paper's initialisation is not specified).

## Measured

**11.79 M params · 1.52 ms** TRT FP16 (RTX 4070 SUPER, batch 1, 1024², median; strongly-typed,
plugin-free, 100 % of nodes in FP16). Paper reports 12.1 M params / 48.24 GFLOPs / 72.4 FPS
@ 1024² on an RTX 3090.

Random weights (no checkpoints in this repo). Latency and parameter counts reflect this
implementation's configuration; accuracy requires training on UAVid (or UDD6 / LoveDA).

## Roadmap (not in this repo)

- Training recipe + UAVid / UDD6 / LoveDA dataloaders
- Trained checkpoints
- A verified accuracy/latency comparison

## Development

Code style is kept tidy with [pre-commit](https://pre-commit.com) hooks: `ruff`
(lint + format) and `docformatter`, plus the standard whitespace / YAML / merge-conflict
checks. Config: [`.pre-commit-config.yaml`](.pre-commit-config.yaml) + `[tool.ruff]` /
`[tool.docformatter]` in [`pyproject.toml`](pyproject.toml).

```bash
pip install -e ".[dev]"
pre-commit install          # run automatically on every commit
pre-commit run --all-files  # run now over the whole tree
```

## License & citation

Reimplementation code: **Apache-2.0** ([`LICENSE`](LICENSE)).
If you use this, **cite the original paper** (see [`CITATION.cff`](CITATION.cff)), not this
repository. The paper is open access under CC BY 4.0 (Sensors / MDPI).
