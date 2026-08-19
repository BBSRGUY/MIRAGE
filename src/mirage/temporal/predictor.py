from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class TinyResidualPredictor(nn.Module):
    """Tokenwise correction P(delta_x, timestep) added to the preceding residual."""

    def __init__(self, width: int, bottleneck: int | None = None):
        super().__init__()
        bottleneck = bottleneck or min(64, max(1, width // 8))
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.raw_down = nn.Linear(width, bottleneck, bias=False)
        self.time = nn.Sequential(nn.Linear(2, bottleneck), nn.SiLU())
        self.up = nn.Linear(bottleneck, width)

    def forward(self, delta_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim == 0:
            timestep = timestep[None]
        time_features = torch.stack((timestep, timestep.square()), dim=-1).float()
        while time_features.ndim < delta_x.ndim:
            time_features = time_features.unsqueeze(-2)
        hidden = F.silu(self.down(self.norm(delta_x)) + self.raw_down(delta_x)) + self.time(
            time_features
        )
        return self.up(hidden)


@dataclass(frozen=True)
class PredictorFit:
    model: TinyResidualPredictor
    train_relative_error: float
    validation_relative_error: float
    validation_cosine: float
    parameter_count: int


def _prediction_error(
    model: TinyResidualPredictor,
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
) -> tuple[float, float]:
    differences, targets, predictions = [], [], []
    model.eval()
    with torch.no_grad():
        for delta_x, previous_residual, target_residual, timestep in samples:
            t = torch.tensor(timestep, device=delta_x.device)
            predicted = previous_residual + model(delta_x, t)
            differences.append((predicted - target_residual).flatten())
            targets.append(target_residual.flatten())
            predictions.append(predicted.flatten())
    difference = torch.cat(differences)
    target = torch.cat(targets)
    prediction = torch.cat(predictions)
    return (
        (difference.norm() / target.norm().clamp_min(1e-12)).item(),
        F.cosine_similarity(prediction, target, dim=0).item(),
    )


def fit_predictor(
    width: int,
    train_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
    validation_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
    *,
    steps: int = 200,
    learning_rate: float = 1e-3,
    seed: int = 0,
    bottleneck: int | None = None,
) -> PredictorFit:
    if not train_samples or not validation_samples:
        raise ValueError(
            "predictor fitting requires disjoint non-empty train and validation samples"
        )
    device = train_samples[0][0].device
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        model = TinyResidualPredictor(width, bottleneck).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(steps):
        delta_x, previous, target, timestep = train_samples[step % len(train_samples)]
        optimizer.zero_grad(set_to_none=True)
        prediction = previous + model(delta_x, torch.tensor(timestep, device=device))
        loss = (prediction - target).square().mean() / target.square().mean().clamp_min(1e-12)
        loss.backward()
        optimizer.step()
    train_error, _ = _prediction_error(model, train_samples)
    validation_error, validation_cosine = _prediction_error(model, validation_samples)
    return PredictorFit(
        model=model,
        train_relative_error=train_error,
        validation_relative_error=validation_error,
        validation_cosine=validation_cosine,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
