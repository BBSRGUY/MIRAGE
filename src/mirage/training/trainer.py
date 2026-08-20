from __future__ import annotations

import json
import random
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ..datasets import StreamingAVDataset, collate_av
from ..m3_config import M3Config
from ..metrics import compare, temporal_consistency
from ..model import MirageGenerator
from ..quantization import apply_m2_mixed_precision
from ..telemetry import parameter_bytes, resident_state_bytes
from .ema import EMA
from .losses import av_sync_loss, behavior_signature, identity_loss, temporal_loss
from .provenance import (
    build_run_provenance,
    comparable_run_inputs,
    hash_model_state,
    sha256_file,
)


def _device(requested: str | torch.device | None) -> torch.device:
    return torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    step: int,
    epoch: int,
    config: M3Config,
    provenance: dict[str, Any],
) -> None:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    temporary = path.with_suffix(".tmp")
    checkpoint_provenance = {
        **provenance,
        "model_state_sha256": hash_model_state(module),
    }
    torch.save(
        {
            "format": "mirage_m3_v1",
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": config.to_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(),
            "provenance": checkpoint_provenance,
        },
        temporary,
    )
    temporary.replace(path)
    sidecar = {
        **checkpoint_provenance,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
    }
    sidecar_path = path.with_suffix(path.suffix + ".provenance.json")
    sidecar_temporary = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    sidecar_temporary.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    sidecar_temporary.replace(sidecar_path)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    provenance: dict[str, Any],
    strict: bool,
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "mirage_m3_v1":
        raise ValueError("unsupported M3 checkpoint format")
    stored_provenance = state.get("provenance")
    if stored_provenance is None:
        if strict:
            raise ValueError("checkpoint has no provenance record")
    elif strict and comparable_run_inputs(stored_provenance) != comparable_run_inputs(provenance):
        raise ValueError(
            "checkpoint provenance does not match config, dataset, teacher features, or code"
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    ema.load_state_dict(state["ema"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    if stored_provenance is not None and hash_model_state(model) != stored_provenance[
        "model_state_sha256"
    ]:
        raise ValueError("checkpoint model-state provenance hash does not match loaded weights")
    return int(state["step"]), int(state["epoch"])


def _autocast(config: M3Config, device: torch.device):
    if config.training.mixed_precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_m3(config: M3Config, device: str | torch.device | None = None) -> dict[str, Any]:
    config.validate()
    device = _device(device)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    torch.manual_seed(config.training.seed + rank)
    random.seed(config.training.seed + rank)
    output = Path(config.training.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = build_run_provenance(config)
    model = MirageGenerator(config.model).to(device)
    if model.resident_estimate(config.training.batch_size) > int(config.model.vram_budget_gb * 2**30):
        raise MemoryError("configured M3 model exceeds inference residency budget before training")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    ema = EMA(model, config.training.ema_decay)
    step, start_epoch = 0, 0
    if config.training.resume:
        step, start_epoch = _load_checkpoint(
            Path(config.training.resume),
            model,
            optimizer,
            ema,
            provenance,
            config.training.strict_resume_provenance,
        )
    training_model: torch.nn.Module = model
    if torch.distributed.is_initialized():
        training_model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    loader = DataLoader(
        StreamingAVDataset(config, "train"),
        batch_size=config.training.batch_size,
        collate_fn=collate_av,
        num_workers=0,
    )
    optimizer.zero_grad(set_to_none=True)
    telemetry_rows = []
    started = perf_counter()
    stop = step >= config.training.max_steps
    last_epoch = start_epoch
    for epoch in range(start_epoch, config.training.epochs):
        last_epoch = epoch
        if stop:
            break
        for batch_index, batch in enumerate(loader):
            video = batch["video"].to(device)
            audio = batch["audio"].to(device) if batch["audio"] is not None else None
            target, _audio_latent = model.encode_targets(video, audio)
            noise = torch.randn_like(target)
            t = torch.rand(target.shape[0], device=device)
            xt = (1 - t[:, None, None]) * noise + t[:, None, None] * target
            flow_target = target - noise
            with _autocast(config, device):
                predicted_velocity, states = training_model(
                    xt,
                    batch["prompt"],
                    t,
                    config.training.gradient_checkpointing,
                )
                predicted_target = xt + (1 - t[:, None, None]) * predicted_velocity
                losses = {
                    "flow": F.mse_loss(predicted_velocity.float(), flow_target.float()),
                    "temporal": temporal_loss(predicted_target, target, config.model.frames),
                    "identity": identity_loss(predicted_target, target, config.model.frames),
                }
                teacher = batch["teacher_feature"]
                losses["behavior"] = (
                    F.mse_loss(
                        behavior_signature(predicted_velocity), teacher[:, :4].to(device)
                    )
                    if teacher is not None
                    else predicted_velocity.new_zeros(())
                )
                if audio is not None:
                    _video_out, predicted_audio = model.decode_latents(predicted_target)
                    losses["av_sync"] = av_sync_loss(
                        predicted_audio, states["motion"], config.model.frames
                    )
                else:
                    losses["av_sync"] = predicted_velocity.new_zeros(())
                total = sum(
                    losses[name] * getattr(config.losses, name) for name in losses
                ) / config.training.gradient_accumulation
            total.backward()
            if (batch_index + 1) % config.training.gradient_accumulation:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                training_model.parameters(), config.training.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            step += 1
            row = {
                "step": step,
                "epoch": epoch,
                "loss": float(total.detach()) * config.training.gradient_accumulation,
                **{f"loss_{name}": float(value.detach()) for name, value in losses.items()},
                "grad_norm": float(grad_norm),
                "elapsed_s": perf_counter() - started,
                "allocated_vram_bytes": (
                    torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                ),
                "peak_vram_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
            }
            telemetry_rows.append(row)
            if rank == 0 and step % config.training.checkpoint_every == 0:
                _save_checkpoint(
                    output / f"step-{step:08d}.pt",
                    model,
                    optimizer,
                    ema,
                    step,
                    epoch,
                    config,
                    provenance,
                )
            if step >= config.training.max_steps:
                stop = True
                break
        if stop:
            break
    if rank == 0:
        checkpoint = output / "last.pt"
        _save_checkpoint(
            checkpoint, model, optimizer, ema, step, last_epoch, config, provenance
        )
        report = {
            "variant": config.variant,
            "steps_completed": step,
            "checkpoint": str(checkpoint),
            "parameter_bytes_bf16_equivalent": parameter_bytes(model) // 2,
            "inference_resident_estimate_bytes": model.resident_estimate(1),
            "projection_backend": config.model.projection_backend,
            "m2_allocation_baseline": "grouped INT4 / rowwise INT8",
            "provenance": {
                **provenance,
                "model_state_sha256": hash_model_state(model),
                "checkpoint_sha256": sha256_file(checkpoint),
                "sidecar": str(checkpoint.with_suffix(checkpoint.suffix + ".provenance.json")),
            },
            "telemetry": telemetry_rows,
        }
        (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    return {"rank": rank, "steps_completed": step}


@torch.inference_mode()
def evaluate_m3(
    config: M3Config,
    checkpoint: str | Path,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    device = _device(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MirageGenerator(config.model).to(device)
    model.load_state_dict(state["model"])
    if "provenance" in state and hash_model_state(model) != state["provenance"][
        "model_state_sha256"
    ]:
        raise ValueError("evaluation checkpoint model-state provenance verification failed")
    ema = EMA(model, config.training.ema_decay)
    ema.load_state_dict(state["ema"])
    ema.copy_to(model)
    model.eval()
    independent_reference = model
    quantized = deepcopy(model)
    compression = apply_m2_mixed_precision(
        quantized,
        config.compression.allocation_report,
        config.compression.int4_group_size,
    )
    quantized.to(device).eval()
    budget = int(config.model.vram_budget_gb * 2**30)
    estimated_residency = quantized.resident_estimate(1)
    if estimated_residency >= budget:
        raise MemoryError("quantized M3 inference does not fit the configured residency budget")
    rows = []
    for prompt, seed in zip(config.evaluation.prompts, config.evaluation.seeds):
        reference = independent_reference.generate(prompt, seed=seed, device=device, use_cache=False)
        generated = quantized.generate(prompt, seed=seed, device=device, use_cache=False)
        rows.append(
            {
                "prompt": prompt,
                "seed": seed,
                "temporal_consistency": temporal_consistency(generated.video),
                "quality_retention_vs_bf16_independent": compare(
                    generated.video, reference.video
                ),
                "telemetry": generated.telemetry.to_dict(),
            }
        )
    report = {
        "variant": config.variant,
        "checkpoint": str(checkpoint),
        "held_out_prompts": rows,
        "compression": compression,
        "bf16_independent_state_bytes": resident_state_bytes(independent_reference),
        "quantized_resident_estimate_bytes": estimated_residency,
        "configured_budget_bytes": budget,
        "headroom_bytes": budget - estimated_residency,
        "cache_predict_default": False,
        "quality_scope": "initial trained smoke metrics; VBench/FVD require a real evaluation corpus",
        "no_teacher_runtime": True,
        "checkpoint_provenance": state.get("provenance"),
    }
    output = Path(config.training.output_dir)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
