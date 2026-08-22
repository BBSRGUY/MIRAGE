# MIRAGE

MIRAGE (Motion-Invariant Residual Adaptive Generative Engine) is a research harness for
video generation designed around a hard constraint: the complete inference path must remain
resident on one GPU with at most 24 GiB of VRAM and no CPU offload.

The repository now includes the active M3 trainable generator pipeline. The first checkpoint is
a synthetic systems smoke result, not a perceptual-quality claim.

Milestone 2 is an empirical study against the official **LTX-2.5 22B audio/video teacher**.
It streams dense BF16 teacher blocks for activation capture, reads trained BF16 projection
weights directly from Safetensors, and rejects older LTX checkpoints and quantized structural
surrogates. NVFP4 may be benchmarked as a runtime baseline, but it is never used to infer
dense-weight redundancy.

## What runs today

- persistent scene state plus per-frame motion state;
- independent transformer projections with an M2-derived grouped-INT4/rowwise-INT8 inference policy;
- local/anchor spatiotemporal attention with measured mask density;
- predictive block residual reuse across generation steps;
- an executable timestep precision schedule (FP32 at sensitive edge steps, BF16 in the middle);
- two-to-four-step flow-style generation;
- one compact latent decoder for RGB video and synchronized mono audio;
- a dense, independent-weight baseline;
- JSON telemetry for latency, allocated and peak VRAM, analytical FLOPs, attention density,
  cache hit rate, precision decisions, and parameter bytes;
- PSNR, a lightweight SSIM proxy, and temporal consistency for regression comparisons.

Low-bit FP8/FP4 basis kernels and block-sparse compute are deliberately not claimed yet.
PyTorch's reference attention consumes the sparse mask correctly, but a custom Triton kernel
is required to turn that logical sparsity into proportional wall-clock savings.

## Run it

Use the existing CUDA Conda runtime on this machine:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
C:\Users\rashm\miniconda3\python.exe -m mirage.cli doctor
C:\Users\rashm\miniconda3\python.exe -m pytest
C:\Users\rashm\miniconda3\python.exe -m mirage.cli generate `
  --config configs/smoke.json --device cuda --output artifacts/sample
C:\Users\rashm\miniconda3\python.exe -m mirage.cli benchmark `
  --config configs/smoke.json --device cuda --output artifacts/benchmark
C:\Users\rashm\miniconda3\python.exe -m mirage.cli ablate-cache `
  --config configs/milestone_24gb.json --threshold 1.0 --device cuda `
  --output artifacts/cache_ablation
```

`generate` writes an animated GIF, a lossless tensor bundle containing video and audio, and
telemetry JSON. `benchmark` warms both paths, then writes both GIFs and a comparison report.
`ablate-cache` uses the exact same weights, prompt, and noise for cached and uncached runs, so
its quality-retention numbers isolate predictive execution instead of comparing random models.

The larger `configs/milestone_24gb.json` configuration is the residency validation target.
Every generation performs a conservative preflight estimate and fails before allocation if
the configured VRAM budget would be exceeded.

## Architectural contract

The default M3 runtime stores each independent layer projection using the heterogeneous policy
selected by the held-out M2.2 block gate:

```text
W_l = grouped-INT4       for robust projections
W_l = rowwise-INT8       for sensitive projections
```

The PyTorch reference runtime dequantizes each projection transiently on the GPU. Native fused
low-bit kernels are deferred to M4. The earlier shared-basis plus low-rank representation remains
available only through an explicit ablation config because M2.2 selected zero basis projections.

Generation maintains spatial scene tokens across the entire clip and separate spatiotemporal
motion tokens. At each flow step, the scene changes only through the time-averaged update;
motion receives the full update. Cache/predict execution is disabled in the default trained path;
the M1 cache implementation remains available solely for controlled ablation until trained
trajectory evidence supports it.

## Experimental discipline

Treat `benchmark.json` as a systems regression artifact, not as a trained quality result.
Quality comparisons become meaningful only when candidate and baseline are trained or
distilled on the same data and evaluation seed set. Change one feature at a time, retain its
configuration and checkpoint, and report median latency after warmup together with peak VRAM,
FLOPs, sparsity, cache rate, and held-out perceptual/temporal metrics.

M3 defaults to the M2-selected heterogeneous grouped-INT4/rowwise-INT8 policy with independent
weights. Shared bases are retained only as a controlled ablation, and temporal cache/predict
execution is disabled until a trained native trajectory supports it.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.venv\Scripts\python.exe -m mirage.cli m3-train --config configs/m3_mirage_s.json --device cuda
.venv\Scripts\python.exe -m mirage.cli m3-eval --config configs/m3_mirage_s.json `
  --checkpoint artifacts/m3/mirage-s-smoke/last.pt --device cuda
```

The staged path is documented in [docs/ROADMAP.md](docs/ROADMAP.md).
The LTX-2.5 study protocol and artifact schema are documented in
[docs/M2_METHOD.md](docs/M2_METHOD.md), [docs/M2_SCHEMA.md](docs/M2_SCHEMA.md), and the measured
[M2 results](docs/M2_RESULTS.md). The active recovery method is documented in
[docs/M21_METHOD.md](docs/M21_METHOD.md).
[M3 method](docs/M3_METHOD.md) and [M3 status](docs/M3_STATUS.md) document the active training
milestone and clearly separate its foundation smoke pass from the still-unpassed quality gate.
The bounded [M3 real AV data protocol](docs/M3_DATA.md) defines the frozen Panda-70M subset,
source-isolated splits, deterministic shards, and audit required before the first real run.
