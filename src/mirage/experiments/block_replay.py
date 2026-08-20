from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ..compression.factorization import load_factorization
from ..datasets import FeatureStore, load_prompt_split
from ..m2_config import M2Config
from ..teacher.ltx import LTXTeacherAdapter
from ..teacher.mapping import LTX25_PROJECTION_PATHS, resolve_module
from .independent_precision import quantize_int4_groups, quantize_int8_rows


def _prepare_weights(
    config: M2Config, allocation: dict[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    root = Path(config.output_dir)
    store = FeatureStore(root)
    decisions = allocation["portfolio"]["decisions"]
    by_scheme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    independent = []
    for decision in decisions:
        (independent if decision["choice"] in {"INT8_ROW", "INT4_GROUP64"} else by_scheme[decision["scheme"]]).append(
            decision
        )
    weights: dict[str, torch.Tensor] = {}
    for scheme, rows in by_scheme.items():
        source = root / "reports" / "adaptive_fits" / f"{scheme}.safetensors"
        fitted = source.with_name(f"{source.stem}_activation_fit.safetensors")
        artifact = fitted if fitted.exists() else source
        if not artifact.exists():
            raise FileNotFoundError(f"selected allocation artifact is missing: {artifact}")
        fit = load_factorization(artifact, device=device)
        index_by_name = {name: index for index, name in enumerate(fit.metrics["layer_names"])}
        for row in rows:
            weights[row["layer"]] = (
                fit.reconstruct_layer(index_by_name[row["layer"]]).to(torch.bfloat16).cpu()
            )
        del fit
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for row in independent:
        source = store.load(f"weight/{row['layer']}", device=device)["weight"].float()
        function = quantize_int8_rows if row["choice"] == "INT8_ROW" else quantize_int4_groups
        reconstructed, _bytes = function(source)
        weights[row["layer"]] = reconstructed.to(torch.bfloat16).cpu()
        del source, reconstructed
    return weights


def run_block_replay(
    config: M2Config, device: str | torch.device | None = None
) -> dict[str, Any]:
    device = torch.device(device or config.teacher.device)
    root = Path(config.output_dir)
    allocation = json.loads(
        (root / "reports" / "heterogeneous_allocation.json").read_text()
    )
    compressed = _prepare_weights(config, allocation, device)
    prompts = load_prompt_split(
        config.data.prompts_file,
        config.data.num_train_prompts,
        config.data.num_eval_prompts,
        config.data.seed,
    )
    evaluation = [prompt for prompt in prompts if prompt.split == "eval"]
    adapter = LTXTeacherAdapter(config.teacher)
    rows: list[dict[str, Any]] = []
    replaying = False
    applied_swaps = 0
    maximum_weight_delta = 0.0

    def no_capture(_name: str, _tensor: torch.Tensor, _metadata: dict[str, Any]) -> None:
        return None

    def replay(
        block_index: int,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
        step_index: int,
        context: dict[str, Any],
    ) -> None:
        nonlocal replaying, applied_swaps, maximum_weight_delta
        if replaying:
            return
        video = kwargs.get("video", args[0] if args else None)
        audio = kwargs.get("audio", args[1] if len(args) > 1 else None)
        baseline_video = output[0] if isinstance(output, tuple) else None
        if video is None or baseline_video is None:
            return
        candidates: list[tuple[torch.nn.Module, torch.Tensor]] = []
        try:
            replaying = True
            for family in config.teacher.projection_families:
                normalized = f"block.{block_index:02d}.{family}"
                candidate = compressed.get(normalized)
                if candidate is None:
                    continue
                projection = resolve_module(module, LTX25_PROJECTION_PATHS[family])
                candidates.append((projection, candidate))

            # The official disk-streaming wrapper reloads the teacher block in its
            # own pre-hook on every invocation.  Install this hook after that hook
            # so the replay candidates are applied to the tensors that are
            # actually consumed by the nested forward.
            def apply_candidates(
                _module: torch.nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any]
            ) -> None:
                nonlocal applied_swaps, maximum_weight_delta
                for projection, candidate in candidates:
                    value = candidate.to(
                        device=projection.weight.device, dtype=projection.weight.dtype
                    )
                    maximum_weight_delta = max(
                        maximum_weight_delta,
                        (projection.weight.detach().float() - value.float()).abs().max().item(),
                    )
                    projection.weight.data = value
                    applied_swaps += 1

            candidate_hook = module.register_forward_pre_hook(apply_candidates, with_kwargs=True)
            video_copy = replace(video, x=video.x.clone())
            audio_copy = replace(audio, x=audio.x.clone()) if audio is not None else None
            try:
                candidate_output = module(video_copy, audio_copy)[0].x.float()
            finally:
                candidate_hook.remove()
            baseline = baseline_video.x.float()
            difference = candidate_output - baseline
            rows.append(
                {
                    "sample_id": context["sample_id"],
                    "block_index": block_index,
                    "step_index": step_index,
                    "relative_error": (difference.norm() / baseline.norm().clamp_min(1e-12)).item(),
                    "cosine": F.cosine_similarity(
                        candidate_output.flatten(), baseline.flatten(), dim=0
                    ).item(),
                    "token_count": baseline.shape[-2],
                }
            )
        finally:
            replaying = False

    adapter.load()
    adapter.install_capture_hooks(no_capture)
    adapter.install_block_replay(replay)
    try:
        for index, prompt in enumerate(prompts):
            if prompt.split != "eval":
                continue
            adapter.run_prompt(
                prompt.text,
                sample_id=prompt.sample_id,
                split=prompt.split,
                seed=config.data.seed + index,
                frames=config.generation.frames,
                height=config.generation.height,
                width=config.generation.width,
                steps=config.generation.steps,
                guidance_scale=config.generation.guidance_scale,
                max_sequence_length=config.generation.max_sequence_length,
            )
    finally:
        adapter.unload()
        compressed.clear()
    report = {
        "teacher": config.teacher.model_id,
        "evaluation_prompts": [prompt.to_dict() for prompt in evaluation],
        "train_prompts_used_for_scoring": False,
        "compression_ratio": allocation["portfolio"]["compression_ratio"],
        "applied_projection_swaps": applied_swaps,
        "maximum_weight_delta": maximum_weight_delta,
        "mean_block_relative_error": sum(row["relative_error"] for row in rows) / len(rows),
        "mean_block_cosine": sum(row["cosine"] for row in rows) / len(rows),
        "worst_block_relative_error": max(row["relative_error"] for row in rows),
        "worst_block_cosine": min(row["cosine"] for row in rows),
        "measurements": rows,
        "scope": "local complete-block replay on identical teacher block inputs",
    }
    if not rows or applied_swaps == 0 or maximum_weight_delta == 0.0:
        raise RuntimeError("block replay did not apply non-identical compressed weights")
    (root / "reports" / "block_replay.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
