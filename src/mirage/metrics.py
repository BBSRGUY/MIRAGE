from __future__ import annotations

import torch
from torch.nn import functional as F


def psnr(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    mse = F.mse_loss(candidate, reference).item()
    return float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(4.0 / mse)).item()


def global_ssim(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Fast global SSIM proxy for regression testing; not a replacement for FVD."""
    x, y = candidate.float(), reference.float()
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(unbiased=False), y.var(unbiased=False)
    covariance = ((x - mx) * (y - my)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return (
        ((2 * mx * my + c1) * (2 * covariance + c2))
        / ((mx.square() + my.square() + c1) * (vx + vy + c2))
    ).item()


def temporal_consistency(video: torch.Tensor) -> float:
    if video.shape[1] < 2:
        return 1.0
    return (1.0 - (video[:, 1:] - video[:, :-1]).abs().mean() / 2.0).item()


def compare(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    return {
        "psnr_db": psnr(candidate, reference),
        "ssim_proxy": global_ssim(candidate, reference),
        "temporal_consistency": temporal_consistency(candidate),
    }
