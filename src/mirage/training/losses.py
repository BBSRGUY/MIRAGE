from __future__ import annotations

import torch
from torch.nn import functional as F


def behavior_signature(value: torch.Tensor) -> torch.Tensor:
    flat = value.float().flatten(1)
    return torch.stack(
        (flat.mean(1), flat.std(1, unbiased=False), flat.square().mean(1).sqrt(), flat.abs().mean(1)),
        dim=1,
    )


def temporal_loss(predicted: torch.Tensor, target: torch.Tensor, frames: int) -> torch.Tensor:
    predicted = predicted.view(predicted.shape[0], frames, -1, predicted.shape[-1])
    target = target.view(target.shape[0], frames, -1, target.shape[-1])
    return F.l1_loss(predicted[:, 1:] - predicted[:, :-1], target[:, 1:] - target[:, :-1])


def identity_loss(predicted: torch.Tensor, target: torch.Tensor, frames: int) -> torch.Tensor:
    predicted = predicted.view(predicted.shape[0], frames, -1, predicted.shape[-1]).mean(2)
    target = target.view(target.shape[0], frames, -1, target.shape[-1]).mean(2)
    predicted_identity = predicted.mean(1)
    target_identity = target.mean(1)
    return F.mse_loss(predicted_identity, target_identity)


def av_sync_loss(audio: torch.Tensor, motion: torch.Tensor, frames: int) -> torch.Tensor:
    audio_energy = audio.float().view(audio.shape[0], frames, -1).square().mean(2).sqrt()
    motion_energy = motion.float().view(motion.shape[0], frames, -1).square().mean(2).sqrt()
    audio_energy = (audio_energy - audio_energy.mean(1, keepdim=True)) / audio_energy.std(
        1, keepdim=True, unbiased=False
    ).clamp_min(1e-5)
    motion_energy = (motion_energy - motion_energy.mean(1, keepdim=True)) / motion_energy.std(
        1, keepdim=True, unbiased=False
    ).clamp_min(1e-5)
    return F.mse_loss(motion_energy, audio_energy)
