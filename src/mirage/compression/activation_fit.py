from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from .factorization import FactorizationResult


@dataclass(frozen=True)
class ActivationMetrics:
    relative_activation_error: float
    cosine_similarity: float
    mean_token_cosine: float
    mean_absolute_error: float
    normalized_rmse: float
    p95_token_error: float
    max_token_error: float
    token_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def apply_projection(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    if inputs.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"activation width {inputs.shape[-1]} does not match projection input {weight.shape[-1]}"
        )
    output = inputs.float() @ weight.float().T
    return activation(output) if activation is not None else output


def activation_metrics(
    inputs: torch.Tensor,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> ActivationMetrics:
    """Measure local projection behavior on real or controlled held-out inputs."""
    y = apply_projection(inputs, original, activation).reshape(-1, original.shape[0])
    y_hat = apply_projection(inputs, reconstructed, activation).reshape_as(y)
    difference = y_hat - y
    token_norm = y.norm(dim=-1).clamp_min(1e-12)
    token_error = difference.norm(dim=-1) / token_norm
    cosine = F.cosine_similarity(y_hat.flatten(), y.flatten(), dim=0)
    token_cosine = F.cosine_similarity(y_hat, y, dim=-1)
    rmse = difference.square().mean().sqrt()
    reference_rms = y.square().mean().sqrt().clamp_min(1e-12)
    return ActivationMetrics(
        relative_activation_error=(difference.norm() / y.norm().clamp_min(1e-12)).item(),
        cosine_similarity=cosine.item(),
        mean_token_cosine=token_cosine.mean().item(),
        mean_absolute_error=difference.abs().mean().item(),
        normalized_rmse=(rmse / reference_rms).item(),
        p95_token_error=torch.quantile(token_error, 0.95).item(),
        max_token_error=token_error.max().item(),
        token_count=y.shape[0],
    )


def optimize_for_activations(
    initial: FactorizationResult,
    weights: torch.Tensor,
    train_inputs: list[torch.Tensor],
    validation_inputs: list[torch.Tensor],
    *,
    steps: int,
    learning_rate: float,
    lambda_weight: float = 0.1,
    lambda_activation: float = 1.0,
) -> tuple[FactorizationResult, dict[str, object]]:
    """Behavior-fit a factorization on train features and report disjoint validation fidelity."""
    if len(train_inputs) != weights.shape[0] or len(validation_inputs) != weights.shape[0]:
        raise ValueError("one train and validation activation tensor is required per layer")
    if steps < 1:
        raise ValueError("behavior fitting requires at least one optimization step")
    device = weights.device
    basis = torch.nn.Parameter(initial.basis.detach().to(device))
    alpha = torch.nn.Parameter(initial.alpha.detach().to(device))
    u = torch.nn.Parameter(initial.u.detach().to(device))
    v = torch.nn.Parameter(initial.v.detach().to(device))
    optimizer = torch.optim.Adam([basis, alpha, u, v], lr=learning_rate)
    source = weights.float()
    denominator = source.square().mean().clamp_min(1e-12)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstructed = torch.einsum("lk,koi->loi", alpha, basis)
        if u.shape[-1]:
            reconstructed = reconstructed + torch.einsum("lor,lri->loi", u, v)
        weight_loss = (reconstructed - source).square().mean() / denominator
        activation_losses = []
        for layer, inputs in enumerate(train_inputs):
            x = inputs.to(device).float().reshape(-1, inputs.shape[-1])
            target = x @ source[layer].T
            predicted = x @ reconstructed[layer].T
            activation_losses.append(
                (predicted - target).square().mean() / target.square().mean().clamp_min(1e-12)
            )
        activation_loss = torch.stack(activation_losses).mean()
        loss = lambda_weight * weight_loss + lambda_activation * activation_loss
        loss.backward()
        optimizer.step()
    fitted = FactorizationResult(
        basis.detach(), alpha.detach(), u.detach(), v.detach(), dict(initial.metrics)
    )
    reconstructed = fitted.reconstruct()
    train = [
        activation_metrics(train_inputs[i], source[i], reconstructed[i]).to_dict()
        for i in range(source.shape[0])
    ]
    validation = [
        activation_metrics(validation_inputs[i], source[i], reconstructed[i]).to_dict()
        for i in range(source.shape[0])
    ]
    report = {
        "objective": {"lambda_weight": lambda_weight, "lambda_activation": lambda_activation},
        "steps": steps,
        "train": train,
        "validation": validation,
        "train_mean_relative_error": sum(x["relative_activation_error"] for x in train)
        / len(train),
        "validation_mean_relative_error": sum(x["relative_activation_error"] for x in validation)
        / len(validation),
        "train_mean_cosine": sum(x["cosine_similarity"] for x in train) / len(train),
        "validation_mean_cosine": sum(x["cosine_similarity"] for x in validation) / len(validation),
    }
    fitted.metrics["behavior_fit"] = report
    return fitted, report
