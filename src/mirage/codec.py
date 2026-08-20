from __future__ import annotations

import torch
from torch import nn


class CompactAVCodec(nn.Module):
    """Small patch decoder with a synchronized latent-audio head."""

    def __init__(self, width: int, patch_size: int, audio_hop: int = 320):
        super().__init__()
        self.patch_size = patch_size
        self.audio_hop = audio_hop
        self.video_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 3 * patch_size * patch_size), nn.Tanh()
        )
        self.audio_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, audio_hop), nn.Tanh())
        self.video_encoder = nn.Sequential(
            nn.Linear(3 * patch_size * patch_size, width), nn.LayerNorm(width)
        )
        self.audio_encoder = nn.Sequential(nn.Linear(audio_hop, width), nn.LayerNorm(width))

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Patchify [B,F,3,H,W] into the model's spatiotemporal latent sequence."""
        b, frames, channels, height, width = video.shape
        p = self.patch_size
        if channels != 3 or height % p or width % p:
            raise ValueError("video must be RGB and spatial dimensions divisible by patch_size")
        patches = video.reshape(b, frames, channels, height // p, p, width // p, p)
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6).reshape(
            b, frames * (height // p) * (width // p), channels * p * p
        )
        return self.video_encoder(patches)

    def encode_audio(self, audio: torch.Tensor, frames: int) -> torch.Tensor:
        required = frames * self.audio_hop
        if audio.shape[-1] < required:
            audio = torch.nn.functional.pad(audio, (0, required - audio.shape[-1]))
        return self.audio_encoder(audio[..., :required].reshape(audio.shape[0], frames, -1))

    def decode_video(
        self, tokens: torch.Tensor, frames: int, height: int, width: int
    ) -> torch.Tensor:
        b = tokens.shape[0]
        h, w, p = height // self.patch_size, width // self.patch_size, self.patch_size
        pixels = self.video_head(tokens).view(b, frames, h, w, 3, p, p)
        return pixels.permute(0, 1, 4, 2, 5, 3, 6).reshape(b, frames, 3, height, width)

    def decode_audio(self, motion: torch.Tensor, frames: int) -> torch.Tensor:
        b = motion.shape[0]
        frame_features = motion.view(b, frames, -1, motion.shape[-1]).mean(dim=2)
        return self.audio_head(frame_features).reshape(b, frames * self.audio_hop)

    def forward(
        self, tokens: torch.Tensor, motion: torch.Tensor, frames: int, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decode_video(tokens, frames, height, width), self.decode_audio(motion, frames)
