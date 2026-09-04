"""Minimal smoke test: construct, forward (eval), forward+backward (train)."""

import torch

from glam_uav import glam_uav

FACTORIES = [glam_uav]


def test_forward_backward():
    for fac in FACTORIES:
        m = fac(num_classes=8)
        m.eval()
        with torch.no_grad():
            y = m(torch.randn(1, 3, 256, 256))
        y0 = y[0] if isinstance(y, (list, tuple)) else y
        assert y0.shape == (1, 8, 256, 256), (fac.__name__, y0.shape)
        assert torch.isfinite(y0).all()
        m.train()
        out = m(torch.randn(2, 3, 128, 128))
        outs = out if isinstance(out, (list, tuple)) else (out,)
        loss = sum(o.float().pow(2).mean() for o in outs)
        loss.backward()
        n_grad = sum(
            p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
        )
        assert n_grad > 0
        print(
            f"{fac.__name__}: OK  ({sum(p.numel() for p in m.parameters()) / 1e6:.2f} M params)"
        )


if __name__ == "__main__":
    test_forward_backward()
