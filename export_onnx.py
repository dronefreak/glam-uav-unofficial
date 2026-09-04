"""Export GLAM-UAV to ONNX.

    python export_onnx.py --out glam_uav.onnx --size 1024 --classes 8 [--half]

--half writes a pure-FP16 graph (model.half()). For a mixed-precision graph that keeps
sensitive ops in FP32 (recommended for accuracy), convert the FP32 ONNX afterwards with
e.g. `modelopt.onnx.autocast` or `onnxconverter-common.float16`.
"""

import argparse

import torch

from glam_uav import glam_uav

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="glam_uav.onnx")
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--classes", type=int, default=8)
ap.add_argument("--half", action="store_true")
ap.add_argument("--opset", type=int, default=17)
a = ap.parse_args()

m = glam_uav(num_classes=a.classes).eval()
x = torch.randn(1, 3, a.size, a.size)
if a.half:
    m, x = m.half(), x.half()
with torch.no_grad():
    _ = m(x)
torch.onnx.export(
    m,
    (x,),
    a.out,
    opset_version=a.opset,
    input_names=["input"],
    output_names=["logits"],
)
print(
    f"wrote {a.out}  ({sum(p.numel() for p in m.parameters()) / 1e6:.2f} M params, "
    f"input 1x3x{a.size}x{a.size}{' fp16' if a.half else ''})"
)
