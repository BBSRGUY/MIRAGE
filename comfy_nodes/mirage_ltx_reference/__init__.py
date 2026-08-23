"""ComfyUI nodes for the MIRAGE LTX multi-reference experiment.

This node ports the local Wan2GP EditAnything reference adapter to ComfyUI's
LTX-2.5 model implementation.  It deliberately uses the existing Comfy runtime
and existing local weights; it performs no downloads.
"""

from __future__ import annotations

import logging
import types
from pathlib import Path

import comfy.ldm.common_dit
import comfy.ldm.modules.attention
import comfy.sd
import comfy.utils
import folder_paths
import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_MODULE_PATH = (
    "C:/Users/rashm/OneDrive/Desktop/videogen/Wan2GP/ckpts/"
    "edit_anything_reference_v0.1_r128_ref_adaln_proj-role_embedding-"
    "ref_attn-ref_visual_proj.module.safetensors"
)
WAN2GP_LORA_ROOT = "C:/Users/rashm/OneDrive/Desktop/videogen/Wan2GP/loras/ltx2"
DEFAULT_LORA_HINT = "edit_anything_reference_v0.1_r128"

# Reuse the user's existing Wan2GP LoRA directory in ComfyUI.  This only adds a
# search path; it neither copies nor downloads model files.
if Path(WAN2GP_LORA_ROOT).is_dir():
    folder_paths.add_model_folder_path("loras", WAN2GP_LORA_ROOT)


def _subset(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return F.linear(x, weight.to(device=x.device, dtype=x.dtype), None if bias is None else bias.to(device=x.device, dtype=x.dtype))


def _project_reference(
    ref_latent: torch.Tensor,
    state: dict[str, torch.Tensor],
    token_scale: float,
    adaln_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the 32 reference tokens and 9x4096 AdaLN vector on CPU."""
    latent = ref_latent.detach().to(device="cpu", dtype=torch.float32)
    if latent.ndim != 5 or latent.shape[1] != 128:
        raise ValueError(f"EditAnything expects an LTX video latent [B,128,F,H,W], got {tuple(latent.shape)}")
    ref_frame = latent.mean(dim=2)

    visual = _subset(state, "ref_visual_proj.")
    local = F.adaptive_avg_pool2d(ref_frame, (4, 8)).permute(0, 2, 3, 1).reshape(ref_frame.shape[0], 32, -1)
    mean = ref_frame.mean(dim=(-2, -1))
    std = ref_frame.std(dim=(-2, -1), unbiased=False)
    stats = torch.cat([mean, std], dim=-1).unsqueeze(1).expand(-1, 32, -1)
    tokens = torch.cat([local, stats], dim=-1)
    tokens = _linear(F.silu(_linear(tokens, visual["fc1.weight"], visual.get("fc1.bias"))), visual["proj.weight"], visual.get("proj.bias"))
    tokens = F.layer_norm(
        tokens,
        (tokens.shape[-1],),
        visual["norm.weight"].float(),
        visual["norm.bias"].float(),
    )
    tokens = (tokens + visual["pos_embed"].float()[:, : tokens.shape[1]]) * float(token_scale)

    adaln = _subset(state, "ref_adaln_proj.")
    pooled = torch.cat(
        [
            F.adaptive_avg_pool2d(ref_frame, (1, 1)).flatten(1),
            F.adaptive_avg_pool2d(ref_frame, (2, 2)).flatten(1),
            F.adaptive_max_pool2d(ref_frame, (1, 1)).flatten(1),
        ],
        dim=-1,
    )
    adaln_out = _linear(F.silu(_linear(pooled, adaln["fc1.weight"], adaln.get("fc1.bias"))), adaln["proj.weight"], adaln.get("proj.bias"))
    return tokens.to(torch.bfloat16), (adaln_out * float(adaln_scale)).to(torch.bfloat16)


class _ReferenceAttention(nn.Module):
    def __init__(self, base_attn: nn.Module, state: dict[str, torch.Tensor], prefix: str, context: torch.Tensor) -> None:
        super().__init__()
        object.__setattr__(self, "base_attn", base_attn)
        self.register_buffer("context", context, persistent=False)
        for name in ("to_q", "to_k", "to_v", "to_out.0"):
            safe = name.replace(".", "_")
            self.register_buffer(f"{safe}_a", state[f"{prefix}{name}.lora_A.weight"], persistent=False)
            self.register_buffer(f"{safe}_b", state[f"{prefix}{name}.lora_B.weight"], persistent=False)

    def _projection(self, name: str, base: nn.Module, x: torch.Tensor) -> torch.Tensor:
        safe = name.replace(".", "_")
        out = base(x)
        a = getattr(self, f"{safe}_a")
        b = getattr(self, f"{safe}_b")
        delta = F.linear(F.linear(x.to(a.dtype), a), b)
        return out + delta.to(device=out.device, dtype=out.dtype)

    def forward(self, x: torch.Tensor, transformer_options: dict) -> torch.Tensor:
        base = self.base_attn
        context = self.context.to(device=x.device, dtype=x.dtype)
        if context.shape[0] == 1 and x.shape[0] != 1:
            context = context.expand(x.shape[0], -1, -1)
        q = base.q_norm(self._projection("to_q", base.to_q, x))
        k = base.k_norm(self._projection("to_k", base.to_k, context))
        v = self._projection("to_v", base.to_v, context)
        out = comfy.ldm.modules.attention.optimized_attention(
            q,
            k,
            v,
            base.heads,
            attn_precision=base.attn_precision,
            transformer_options=transformer_options,
        )
        return self._projection("to_out.0", base.to_out[0], out)


class _ReferenceAdaLN(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value, persistent=False)


def _patch_model(
    model,
    state: dict[str, torch.Tensor],
    ref_context: torch.Tensor,
    ref_adaln: torch.Tensor,
    start_block: int,
    end_block: int,
    context_scale: float,
):
    patched = model.clone()
    diffusion_model = patched.model.diffusion_model
    blocks = list(getattr(diffusion_model, "transformer_blocks", ()))
    if not blocks:
        raise TypeError("loaded MODEL is not an LTX transformer")
    if len(blocks) <= end_block:
        raise ValueError(f"adapter requests block {end_block}, but model has only {len(blocks)} blocks")

    for index, block in enumerate(blocks):
        adaln_holder = _ReferenceAdaLN(ref_adaln)
        patched.add_object_patch(f"diffusion_model.transformer_blocks.{index}.mirage_ref_adaln", adaln_holder)
        original_forward = block.forward

        def block_forward(this, *args, _original=original_forward, **kwargs):
            key = "v_timestep" if "v_timestep" in kwargs else "timestep"
            if key in kwargs and kwargs[key] is not None:
                value = this.mirage_ref_adaln.value.to(device=kwargs[key].device, dtype=kwargs[key].dtype)
                if value.ndim == 2:
                    value = value.unsqueeze(1)
                kwargs[key] = kwargs[key] + value
            return _original(*args, **kwargs)

        patched.add_object_patch(
            f"diffusion_model.transformer_blocks.{index}.forward",
            types.MethodType(block_forward, block),
        )

        if not start_block <= index <= end_block:
            continue
        prefix = f"diffusion_model.transformer_blocks.{index}.ref_attn."
        if f"{prefix}to_q.lora_A.weight" not in state:
            raise KeyError(f"sidecar has no reference attention for block {index}")
        ref_attn = _ReferenceAttention(block.attn2, state, prefix, ref_context)
        patched.add_object_patch(f"diffusion_model.transformer_blocks.{index}.mirage_ref_attn", ref_attn)
        original_attn = block.attn2.forward

        def attn_forward(this, x, *args, _original=original_attn, _block=block, **kwargs):
            out = _original(x, *args, **kwargs)
            options = kwargs.get("transformer_options", {})
            return out + _block.mirage_ref_attn(x, options) * float(context_scale)

        patched.add_object_patch(
            f"diffusion_model.transformer_blocks.{index}.attn2.forward",
            types.MethodType(attn_forward, block.attn2),
        )
    return patched


class MIRAGELTXEditAnythingReference:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        preferred = [name for name in loras if DEFAULT_LORA_HINT in name.lower()]
        ordered = preferred + [name for name in loras if name not in preferred]
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "reference_sheet": ("IMAGE",),
                "lora_name": (ordered,),
                "module_path": ("STRING", {"default": DEFAULT_MODULE_PATH}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "context_scale": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.005}),
                "token_scale": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 4.0, "step": 0.05}),
                "adaln_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.1}),
                "start_block": ("INT", {"default": 12, "min": 0, "max": 47}),
                "end_block": ("INT", {"default": 35, "min": 0, "max": 47}),
            }
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "reference_latent")
    FUNCTION = "apply"
    CATEGORY = "MIRAGE/LTX Reference"

    def apply(
        self,
        model,
        vae,
        reference_sheet,
        lora_name,
        module_path,
        lora_strength,
        context_scale,
        token_scale,
        adaln_scale,
        start_block,
        end_block,
    ):
        if start_block > end_block:
            raise ValueError("start_block must not exceed end_block")
        sidecar = Path(module_path).expanduser()
        if not sidecar.is_file():
            raise FileNotFoundError(f"EditAnything sidecar not found: {sidecar}")
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, lora_strength, 0)

        pixels = reference_sheet[:1, :, :, :3]
        ref_latent = vae.encode(pixels)
        state = comfy.utils.load_torch_file(str(sidecar), safe_load=True)
        context, adaln = _project_reference(ref_latent, state, token_scale, adaln_scale)
        model_ref = _patch_model(
            model_lora,
            state,
            context,
            adaln,
            int(start_block),
            int(end_block),
            float(context_scale),
        )
        logging.info(
            "MIRAGE EditAnything installed: %s, blocks %d-%d, reference tokens %s",
            Path(lora_path).name,
            start_block,
            end_block,
            tuple(context.shape),
        )
        return (model_ref, {"samples": ref_latent})


NODE_CLASS_MAPPINGS = {"MIRAGELTXEditAnythingReference": MIRAGELTXEditAnythingReference}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MIRAGELTXEditAnythingReference": "MIRAGE LTX EditAnything Reference (Local)"
}
