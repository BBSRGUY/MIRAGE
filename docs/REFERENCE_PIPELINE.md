# MIRAGE local multi-reference pipeline

This is an inference experiment, not a training pipeline. It uses the weights already present on
this machine and does not download models or create another Python environment.

The current adapter is the local Wan2GP EditAnything release. It consists of two required files:

- the standard LoRA in `Wan2GP/loras/ltx2`;
- the sidecar module in `Wan2GP/ckpts`, containing the reference visual projection, reference AdaLN,
  role embedding, and per-block reference attention.

MIRAGE ports both parts into a ComfyUI custom node. Loading the LoRA without the sidecar is not an
equivalent implementation.

## What multi-reference means in this first experiment

The released local adapter consumes one reference image. MIRAGE accepts up to 12 source images and
packs them into one deterministic, text-free contact sheet. The prompt records the semantic role and
sheet position of every reference. The sheet is then passed through the real EditAnything reference
projection and appended reference-latent path.

This is a compatibility experiment, not a claim that LTX 2.5 was trained for independent 5–10 image
slots. The experiment must be compared against first-frame and ordinary IC-LoRA baselines before its
identity retention is considered established.

## Run

1. Put the source image paths and descriptions in `configs/ref_ltx25.example.json`.
2. Build the assets and workflow with the existing Comfy Python:

   ```powershell
   & "C:\Users\rashm\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe" scripts\mirage_ref.py ref-deploy --config configs\ref_ltx25.example.json --output artifacts\reference
   ```

3. Restart ComfyUI once after the node is first deployed, then open `MIRAGE LTX2.5 Ref2V`.

The generated workflow uses the local NVFP4 LTX 2.5 transformer, local INT8 Gemma text encoder,
local video/audio VAEs, local EditAnything LoRA, and local sidecar. No teacher or training weights are
needed.

## First controlled run

Start with 768×448, 121 frames, one reference, and a fixed seed. Then repeat with two and three
references in the composite sheet. Record wall time, peak VRAM, and identity similarity per reference.
Do not call this native Ref2V support unless the multi-reference cases beat the first-frame/IC-LoRA
controls on held-out inputs.
