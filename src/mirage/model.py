from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import spatiotemporal_mask
from .basis import SharedBasisBank
from .blocks import DenseBlock, IndependentMirageBlock, MirageBlock
from .codec import CompactAVCodec
from .config import MirageConfig
from .precision import DynamicPrecisionPolicy
from .telemetry import Measure, RunTelemetry, estimate_resident_bytes, parameter_bytes
from .text import HashTextEncoder


@dataclass
class GenerationOutput:
    video: torch.Tensor
    audio: torch.Tensor
    telemetry: RunTelemetry


def timestep_embedding(t: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    phase = t[:, None] * frequencies[None]
    out = torch.cat((phase.cos(), phase.sin()), dim=-1)
    return torch.nn.functional.pad(out, (0, width - out.shape[-1]))


class MirageGenerator(nn.Module):
    """Minimal trainable MIRAGE generator with all state resident on one device."""

    def __init__(self, config: MirageConfig):
        super().__init__()
        config.validate()
        self.config = config
        d = config.hidden_size
        self.text = HashTextEncoder(config.vocabulary_size, d, config.text_tokens)
        self.scene_seed = nn.Parameter(
            torch.randn(config.latent_height * config.latent_width, d) * 0.02
        )
        self.motion_seed = nn.Parameter(torch.randn(config.video_tokens, d) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.bank = (
            SharedBasisBank(config.basis_count, d)
            if config.projection_backend == "shared_basis"
            else None
        )
        if self.bank is not None:
            self.blocks = nn.ModuleList(
                [
                    MirageBlock(
                        self.bank,
                        d,
                        config.heads,
                        config.residual_rank,
                        config.cache_threshold,
                        config.max_cache_age,
                    )
                    for _ in range(config.depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    IndependentMirageBlock(
                        d, config.heads, config.cache_threshold, config.max_cache_age
                    )
                    for _ in range(config.depth)
                ]
            )
        self.scene_state = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.motion_state = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.velocity = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.codec = CompactAVCodec(d, config.patch_size)
        self.precision_policy = DynamicPrecisionPolicy(config.precision, config.steps)

    def clear_caches(self) -> None:
        for block in self.blocks:
            block.clear_cache()

    def resident_estimate(self, batch_size: int = 1) -> int:
        return estimate_resident_bytes(self, self.config, batch_size)

    def encode_targets(
        self, video: torch.Tensor, audio: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latent = self.codec.encode_video(video)
        audio_latent = self.codec.encode_audio(audio, self.config.frames) if audio is not None else None
        return latent, audio_latent

    def predict_velocity(
        self,
        x: torch.Tensor,
        prompts: list[str],
        t: torch.Tensor,
        *,
        telemetry: RunTelemetry | None = None,
        gradient_checkpointing: bool = False,
        allow_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Differentiable MIRAGE-native flow field used by M3 training."""
        if x.shape[1:] != (self.config.video_tokens, self.config.hidden_size):
            raise ValueError("latent tensor does not match configured token geometry")
        telemetry = telemetry or RunTelemetry(parameter_bytes=parameter_bytes(self))
        device = x.device
        text = self.text(prompts, device).mean(dim=1)
        condition = text + self.time_mlp(timestep_embedding(t, self.config.hidden_size))
        frames = x.view(
            x.shape[0], self.config.frames, -1, self.config.hidden_size
        )
        scene = frames.mean(dim=1)
        motion = frames - scene[:, None]
        scene = self.scene_state(scene)
        motion = self.motion_state(motion)
        state = self.config.scene_ratio * scene[:, None] + (1 - self.config.scene_ratio) * motion
        state = state.reshape_as(x)
        allowed = spatiotemporal_mask(
            self.config.frames,
            self.config.latent_height,
            self.config.latent_width,
            self.config.attention_window,
            self.config.attention_stride,
            device,
        )
        for block in self.blocks:
            if gradient_checkpointing and self.training:
                state = checkpoint(
                    lambda value, cond: block(value, cond, allowed, telemetry, False),
                    state,
                    condition,
                    use_reentrant=False,
                )
            else:
                state = block(state, condition, allowed, telemetry, allow_cache)
                n, d = self.config.video_tokens, self.config.hidden_size
                telemetry.estimated_flops += int(
                    12 * n * d * d + 4 * allowed.sum().item() * d
                )
        velocity = self.velocity(state)
        return velocity, {"scene": scene, "motion": motion}

    def forward(
        self,
        x: torch.Tensor,
        prompts: list[str],
        t: torch.Tensor,
        gradient_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.predict_velocity(
            x, prompts, t, gradient_checkpointing=gradient_checkpointing, allow_cache=False
        )

    def decode_latents(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frames = latent.view(
            latent.shape[0], self.config.frames, -1, self.config.hidden_size
        )
        scene = frames.mean(1)
        motion = frames - scene[:, None]
        return self.codec(
            latent,
            motion.reshape_as(latent),
            self.config.frames,
            self.config.height,
            self.config.width,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompts: str | list[str],
        *,
        seed: int | None = None,
        device: str | torch.device | None = None,
        use_cache: bool = False,
    ) -> GenerationOutput:
        prompts = [prompts] if isinstance(prompts, str) else prompts
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if next(self.parameters()).device != device:
            self.to(device)
        budget = int(self.config.vram_budget_gb * 2**30)
        estimate = self.resident_estimate(len(prompts))
        if estimate > budget:
            raise MemoryError(
                f"estimated residency {estimate / 2**30:.2f} GiB exceeds "
                f"{self.config.vram_budget_gb:.2f} GiB budget"
            )
        generator = torch.Generator(device=device).manual_seed(
            self.config.seed if seed is None else seed
        )
        telemetry = RunTelemetry(parameter_bytes=parameter_bytes(self))
        self.clear_caches()
        c = self.config
        with Measure(telemetry, device):
            x = torch.randn(
                len(prompts), c.video_tokens, c.hidden_size, generator=generator, device=device
            )
            for step in range(c.steps):
                t = torch.full((len(prompts),), step / max(c.steps - 1, 1), device=device)
                dtype = self.precision_policy.dtype_for(step, device)
                telemetry.record_precision(str(dtype).replace("torch.", ""))
                with self.precision_policy.context(step, device):
                    velocity, _states = self.predict_velocity(
                        x,
                        prompts,
                        t,
                        telemetry=telemetry,
                        allow_cache=use_cache and step > 0,
                    )
                x = x + velocity.float() / c.steps
            video, audio = self.decode_latents(x)
        return GenerationOutput(video.float().cpu(), audio.float().cpu(), telemetry)


class DenseBaseline(nn.Module):
    def __init__(self, config: MirageConfig):
        super().__init__()
        config.validate()
        self.config = config
        d = config.hidden_size
        self.text = HashTextEncoder(config.vocabulary_size, d, config.text_tokens)
        self.seed = nn.Parameter(torch.randn(config.video_tokens, d) * 0.02)
        self.blocks = nn.ModuleList([DenseBlock(d, config.heads) for _ in range(config.depth)])
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.velocity = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.codec = CompactAVCodec(d, config.patch_size)

    @torch.inference_mode()
    def generate(
        self, prompts: str | list[str], *, seed: int = 0, device: str | torch.device | None = None
    ) -> GenerationOutput:
        prompts = [prompts] if isinstance(prompts, str) else prompts
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(device)
        c, telemetry = self.config, RunTelemetry(parameter_bytes=parameter_bytes(self))
        generator = torch.Generator(device=device).manual_seed(seed)
        with Measure(telemetry, device):
            text = self.text(prompts, device).mean(1)
            x = self.seed[None].expand(len(prompts), -1, -1).clone()
            x += torch.randn(x.shape, generator=generator, device=device) * 0.5
            for step in range(c.steps):
                t = torch.full((len(prompts),), step / max(c.steps - 1, 1), device=device)
                condition = text + self.time_mlp(timestep_embedding(t, c.hidden_size))
                for block in self.blocks:
                    x = block(x, condition)
                    n, d = c.video_tokens, c.hidden_size
                    telemetry.estimated_flops += int(24 * n * d * d + 4 * n * n * d)
                x = x - self.velocity(x) / c.steps
            video, audio = self.codec(x, x, c.frames, c.height, c.width)
        return GenerationOutput(video.float().cpu(), audio.float().cpu(), telemetry)
