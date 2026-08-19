from __future__ import annotations

import gc
import json
import os
import platform
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from ..m2_config import TeacherConfig
from .base import CaptureCallback, TeacherAdapter
from .hooks import HookSet, first_tensor
from .mapping import ProjectionSpec, map_ltx25_projections, projection_specs_from_keys

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class LTXTeacherAdapter(TeacherAdapter):
    """Official LTX-2.5 22B adapter with disk/CPU block-streamed execution."""

    def __init__(self, config: TeacherConfig):
        self.config = config
        if config.dtype not in _DTYPES:
            raise ValueError(f"unsupported LTX-2.5 dtype: {config.dtype}")
        self._device = torch.device(config.device)
        self._dtype = _DTYPES[config.dtype]
        self._checkpoint: Path | None = None
        self._specs: list[ProjectionSpec] = []
        self._checkpoint_metadata: dict[str, str] = {}
        self._pipeline_objects: dict[str, Any] = {}
        self._live_transformer: nn.Module | None = None
        self._hooks = HookSet()
        self._callback: CaptureCallback | None = None
        self._context: dict[str, Any] = {}
        self._step_index = -1
        self._timestep = float("nan")
        self._block_inputs: dict[tuple[int, str], torch.Tensor] = {}

    @property
    def model_identifier(self) -> str:
        suffix = f"::{Path(self.config.transformer_file).name}" if self.config.transformer_file else ""
        return f"{self.config.model_id}{suffix}"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def block_count(self) -> int:
        return max((spec.block_index for spec in self._specs), default=-1) + 1

    def _add_official_sources(self) -> None:
        root = Path(self.config.ltx_repo_path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"official LTX-2 checkout not found: {root}")
        for package in ("ltx-core", "ltx-pipelines"):
            source = str(root / "packages" / package / "src")
            if source not in sys.path:
                sys.path.insert(0, source)

    def _required_path(self, value: str | None, label: str) -> Path:
        if value is None:
            raise ValueError(f"LTX-2.5 teacher requires {label}")
        path = Path(os.path.expandvars(value))
        if not path.is_absolute() and self.config.model_root is not None:
            path = Path(os.path.expandvars(self.config.model_root)) / path
        if not path.is_file():
            raise FileNotFoundError(f"LTX-2.5 {label} not found: {path}")
        return path

    def load(self) -> None:
        if self._checkpoint is not None:
            raise RuntimeError("LTX-2.5 teacher is already loaded")
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("LTX-2.5 teacher requested CUDA, but CUDA is unavailable")
        checkpoint = self._required_path(self.config.transformer_file, "BF16 transformer")
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            self._checkpoint_metadata = dict(handle.metadata() or {})
            self._specs = projection_specs_from_keys(handle.keys())
            if not self._specs:
                raise ValueError("checkpoint has no recognized LTX-2.5 AV transformer projections")
            probe = handle.get_tensor(self._specs[0].checkpoint_key)
        version = self._checkpoint_metadata.get("model_version")
        config_raw = self._checkpoint_metadata.get("config", "{}")
        checkpoint_config = json.loads(config_raw)
        class_name = checkpoint_config.get("transformer", {}).get("_class_name")
        if version != "2.5.0" or class_name != "AVTransformer3DModel":
            raise ValueError(
                f"expected LTX-2.5 AVTransformer3DModel, found version={version!r}, "
                f"class={class_name!r}"
            )
        if probe.dtype != torch.bfloat16:
            raise ValueError(
                f"structural analysis requires dense BF16 weights; found {probe.dtype}. "
                "NVFP4/INT8 checkpoints are runtime baselines only."
            )
        if self.block_count != 48:
            raise ValueError(f"expected 48 LTX-2.5 blocks, discovered {self.block_count}")
        self._checkpoint = checkpoint
        self._add_official_sources()
        self._build_runtime_shells()

    def _build_runtime_shells(self) -> None:
        from ltx_pipelines.utils.blocks import DiffusionStage, PromptEncoder
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode

        text_encoder = self._required_path(self.config.text_encoder_file, "Gemma4 BF16 text encoder")
        paths = ModelPaths.from_split(
            transformer_path=str(self._checkpoint),
            text_encoder_path=str(text_encoder),
            video_vae_path=self.config.video_vae_file,
            audio_vae_path=self.config.audio_vae_file,
            duration_head_path=self.config.duration_head_file,
        )
        offload = OffloadMode(self.config.offload_mode)
        self._pipeline_objects["prompt_encoder"] = PromptEncoder(
            paths, self._dtype, self._device, offload_mode=offload
        )
        self._pipeline_objects["stage"] = DiffusionStage.from_checkpoint(
            str(self._checkpoint), self._dtype, self._device, offload_mode=offload
        )

    def named_projections(self) -> Mapping[str, nn.Linear]:
        if self._live_transformer is None:
            return {}
        return map_ltx25_projections(self._live_transformer.transformer_blocks)

    def iter_projection_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        if self._checkpoint is None:
            raise RuntimeError("LTX-2.5 teacher is not loaded")
        with safe_open(str(self._checkpoint), framework="pt", device="cpu") as handle:
            for spec in self._specs:
                tensor = handle.get_tensor(spec.checkpoint_key)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(f"non-BF16 structural tensor at {spec.checkpoint_key}")
                yield spec.name, tensor

    def _sample_tokens(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        value = tensor.detach()
        if value.ndim < 3 or value.shape[-2] <= self.config.max_capture_tokens:
            return value.to(device="cpu", dtype=torch.float32), None
        indices = torch.linspace(
            0, value.shape[-2] - 1, self.config.max_capture_tokens, device=value.device
        ).round().long()
        return value.index_select(-2, indices).to(device="cpu", dtype=torch.float32), indices.cpu()

    def _emit(self, name: str, tensor: torch.Tensor, **metadata: Any) -> torch.Tensor:
        if self._callback is None:
            raise RuntimeError("capture callback is not installed")
        sampled, indices = self._sample_tokens(tensor)
        payload = {
            **self._context,
            "step_index": self._step_index,
            "timestep": self._timestep,
            "original_shape": list(tensor.shape),
            **metadata,
        }
        if indices is not None:
            payload["token_indices"] = indices.tolist()
        self._callback(name, sampled, payload)
        return sampled

    def install_capture_hooks(self, callback: CaptureCallback) -> None:
        if self._checkpoint is None:
            raise RuntimeError("LTX-2.5 teacher is not loaded")
        self._callback = callback

    @staticmethod
    def _modality_tensor(value: Any) -> torch.Tensor | None:
        tensor = getattr(value, "x", None)
        return tensor if isinstance(tensor, torch.Tensor) else None

    def _attach_live_hooks(self, velocity_model: nn.Module) -> None:
        self._hooks.clear()
        self._live_transformer = velocity_model
        blocks = velocity_model.transformer_blocks
        selected = set(self.config.capture_blocks or range(len(blocks)))
        invalid = selected.difference(range(len(blocks)))
        if invalid:
            raise ValueError(f"capture block indices out of range: {sorted(invalid)}")

        def transformer_pre(_module: nn.Module, args: tuple[Any, ...]) -> None:
            self._step_index += 1
            video = args[0] if args else None
            latent = getattr(video, "latent", None)
            timesteps = getattr(video, "timesteps", None)
            context = getattr(video, "context", None)
            if isinstance(timesteps, torch.Tensor):
                self._timestep = float(timesteps.detach().flatten()[0].float().cpu())
            if isinstance(latent, torch.Tensor):
                self._emit("latent.transformer_input", latent, hook="latent", modality="video")
            if isinstance(context, torch.Tensor):
                self._emit(
                    "conditioning.transformer_input", context, hook="conditioning", modality="video"
                )

        self._hooks.extend([velocity_model.register_forward_pre_hook(transformer_pre)])

        for index, block in enumerate(blocks):
            if index not in selected:
                continue

            def block_pre(
                _module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                idx: int = index,
            ) -> None:
                for modality, position in (("video", 0), ("audio", 1)):
                    item = kwargs.get(modality, args[position] if len(args) > position else None)
                    value = self._modality_tensor(item)
                    if value is not None:
                        sampled = self._emit(
                            f"block.{idx:02d}.{modality}.input",
                            value,
                            hook="block_input",
                            block_index=idx,
                            modality=modality,
                        )
                        self._block_inputs[(idx, modality)] = sampled

            def block_post(
                _module: nn.Module, _args: tuple[Any, ...], output: Any, idx: int = index
            ) -> None:
                for modality, position in (("video", 0), ("audio", 1)):
                    item = output[position] if isinstance(output, tuple) and len(output) > position else None
                    value = self._modality_tensor(item)
                    if value is None:
                        continue
                    sampled = self._emit(
                        f"block.{idx:02d}.{modality}.output",
                        value,
                        hook="block_output",
                        block_index=idx,
                        modality=modality,
                    )
                    prior = self._block_inputs.pop((idx, modality), None)
                    if prior is not None and prior.shape == sampled.shape:
                        self._emit(
                            f"block.{idx:02d}.{modality}.residual",
                            sampled - prior,
                            hook="block_residual",
                            block_index=idx,
                            modality=modality,
                        )

            self._hooks.extend(
                [
                    block.register_forward_pre_hook(block_pre, with_kwargs=True),
                    block.register_forward_hook(block_post),
                ]
            )

            def attention_post(
                _module: nn.Module, _args: tuple[Any, ...], output: Any, idx: int = index
            ) -> None:
                value = first_tensor(output)
                if value is not None:
                    self._emit(
                        f"block.{idx:02d}.video.attention",
                        value,
                        hook="attention_output",
                        block_index=idx,
                        modality="video",
                    )

            self._hooks.extend([block.attn1.register_forward_hook(attention_post)])

        for normalized_name, projection in self.named_projections().items():
            block_index = int(normalized_name.split(".")[1])
            if block_index not in selected:
                continue

            def projection_pre(
                _module: nn.Module,
                args: tuple[Any, ...],
                projection_name: str = normalized_name,
                idx: int = block_index,
            ) -> None:
                value = first_tensor(args)
                if value is not None:
                    self._emit(
                        f"projection.{projection_name}.input",
                        value,
                        hook="projection_input",
                        projection_name=projection_name,
                        block_index=idx,
                    )

            self._hooks.extend([projection.register_forward_pre_hook(projection_pre)])

    @torch.inference_mode()
    def run_prompt(
        self,
        prompt: str,
        *,
        sample_id: str,
        split: str,
        seed: int,
        frames: int,
        height: int,
        width: int,
        steps: int,
        guidance_scale: float,
        max_sequence_length: int,
    ) -> None:
        del max_sequence_length
        if self._callback is None:
            raise RuntimeError("capture hooks must be installed before inference")
        if steps != 8:
            raise ValueError("LTX-2.5 distilled teacher uses its trained 8-step schedule")
        if guidance_scale != 1.0:
            raise ValueError("M2 extraction currently records the distilled no-CFG teacher path")

        from ltx_core.components.noisers import GaussianNoiser
        from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.types import ModalitySpec

        prompt_encoder = self._pipeline_objects["prompt_encoder"]
        stage = self._pipeline_objects["stage"]
        latent_frames = (frames - 1) // 8 + 1
        self._context = {
            "sample_id": sample_id,
            "split": split,
            "seed": seed,
            "frames": frames,
            "height": height,
            "width": width,
            "latent_num_frames": latent_frames,
        }
        self._step_index = -1
        (encoded,) = prompt_encoder([prompt])
        self._emit("conditioning.prompt.video", encoded.video_encoding, hook="prompt_conditioning")
        if encoded.audio_encoding is not None:
            self._emit("conditioning.prompt.audio", encoded.audio_encoding, hook="prompt_conditioning")

        original_ctx = stage._transformer_ctx

        @contextmanager
        def hooked_ctx(**kwargs: Any) -> Iterator[nn.Module]:
            with original_ctx(**kwargs) as transformer:
                self._attach_live_hooks(transformer.velocity_model)
                try:
                    yield transformer
                finally:
                    self._hooks.clear()
                    self._live_transformer = None

        stage._transformer_ctx = hooked_ctx
        try:
            generator = torch.Generator(device=self._device).manual_seed(seed)
            stage(
                denoiser=SimpleDenoiser(encoded.video_encoding, encoded.audio_encoding),
                sigmas=DISTILLED_SIGMAS.to(self._device),
                noiser=GaussianNoiser(generator=generator),
                width=width,
                height=height,
                frames=frames,
                fps=24.0,
                video=ModalitySpec(context=encoded.video_encoding),
                audio=ModalitySpec(context=encoded.audio_encoding),
            )
        finally:
            stage._transformer_ctx = original_ctx

    def metadata(self) -> dict[str, Any]:
        if self._checkpoint is None:
            raise RuntimeError("LTX-2.5 teacher is not loaded")
        checkpoint_config = json.loads(self._checkpoint_metadata.get("config", "{}"))
        import transformers

        return {
            "adapter": "official-ltx-2.5-block-streaming",
            "model_identifier": self.model_identifier,
            "model_version": self._checkpoint_metadata.get("model_version"),
            "checkpoint": str(self._checkpoint),
            "checkpoint_bytes": self._checkpoint.stat().st_size,
            "checkpoint_weight_dtype": "bfloat16",
            "device": str(self._device),
            "compute_dtype": str(self._dtype).replace("torch.", ""),
            "dtype": str(self._dtype).replace("torch.", ""),
            "offload_mode": self.config.offload_mode,
            "block_count": self.block_count,
            "projection_count": len(self._specs),
            "transformer_class": checkpoint_config.get("transformer", {}).get("_class_name"),
            "scheduler_config": checkpoint_config.get("scheduler", {}),
            "distilled_sigma_values": [
                1.0,
                0.99375,
                0.9875,
                0.98125,
                0.975,
                0.909375,
                0.725,
                0.421875,
                0.0,
            ],
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version(),
        }

    def unload(self) -> None:
        self._hooks.clear()
        self._callback = None
        self._block_inputs.clear()
        self._live_transformer = None
        self._pipeline_objects.clear()
        self._checkpoint = None
        self._specs.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
