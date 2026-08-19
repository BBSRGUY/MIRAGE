from __future__ import annotations

from typing import Any

import torch


def _quality(source: torch.Tensor, scene: torch.Tensor, motion: torch.Tensor) -> dict[str, float]:
    total_energy = source.square().sum().clamp_min(1e-12)
    reconstruction = scene + motion
    temporal_difference = motion[:, 1:] - motion[:, :-1] if source.shape[1] > 1 else motion * 0
    return {
        "scene_energy_ratio": (scene.square().sum() / total_energy).item(),
        "motion_energy_ratio": (motion.square().sum() / total_energy).item(),
        "reconstruction_error": (
            (reconstruction - source).norm() / source.norm().clamp_min(1e-12)
        ).item(),
        "temporal_smoothness": temporal_difference.square().mean().sqrt().item(),
        "motion_sparsity": (motion.abs() < 0.01 * source.square().mean().sqrt())
        .float()
        .mean()
        .item(),
    }


def temporal_mean_decomposition(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scene = features.mean(dim=1, keepdim=True).expand_as(features)
    return scene, features - scene


def temporal_pca_decomposition(
    features: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if rank < 1 or rank > features.shape[1]:
        raise ValueError(f"temporal PCA rank {rank} invalid for {features.shape[1]} frames")
    temporal = features.permute(1, 0, 2, 3).reshape(features.shape[1], -1).float()
    u, s, vh = torch.linalg.svd(temporal, full_matrices=False)
    reconstructed = (u[:, :rank] * s[:rank]) @ vh[:rank]
    scene = reconstructed.reshape(
        features.shape[1], features.shape[0], features.shape[2], features.shape[3]
    ).permute(1, 0, 2, 3)
    coverage = (s[:rank].square().sum() / s.square().sum().clamp_min(1e-12)).item()
    return scene.to(features.dtype), features - scene.to(features.dtype), coverage


def exponential_lowpass_decomposition(
    features: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 <= alpha < 1:
        raise ValueError("low-pass alpha must be in [0, 1)")
    states = [features[:, 0]]
    for index in range(1, features.shape[1]):
        states.append(alpha * states[-1] + (1 - alpha) * features[:, index])
    scene = torch.stack(states, dim=1)
    return scene, features - scene


def analyze_scene_motion(features: torch.Tensor, ranks: list[int], alpha: float) -> dict[str, Any]:
    if features.ndim != 4:
        raise ValueError("scene/motion features must have shape [batch, time, spatial, channels]")
    mean_scene, mean_motion = temporal_mean_decomposition(features)
    low_scene, low_motion = exponential_lowpass_decomposition(features, alpha)
    pca = {}
    for rank in ranks:
        if rank > features.shape[1]:
            pca[str(rank)] = {"status": "skipped_invalid"}
            continue
        scene, motion, coverage = temporal_pca_decomposition(features, rank)
        pca[str(rank)] = {
            "status": "ok",
            "explained_energy": coverage,
            **_quality(features, scene, motion),
        }
    return {
        "temporal_mean": _quality(features, mean_scene, mean_motion),
        "exponential_lowpass": {"alpha": alpha, **_quality(features, low_scene, low_motion)},
        "temporal_pca": pca,
    }
