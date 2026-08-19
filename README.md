# MIRAGE

MIRAGE (Motion-Invariant Residual Adaptive Generative Engine) is a research harness for
video generation designed around a hard constraint: the complete inference path must remain
resident on one GPU with at most 24 GiB of VRAM and no CPU offload.

This repository contains Milestone 1: a small, end-to-end, trainable generator and an
independent dense reference model. The initialized checkpoints produce noise-like videos;
the milestone validates architecture, execution, measurement, and memory residency—not
perceptual quality before training.

## What runs today

- persistent scene state plus per-frame motion state;
- transformer projections composed from one shared basis bank and layer-local low-rank deltas;
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

For each layer-local projection, MIRAGE stores

```text
W_l = sum_i alpha[l, i] B_i + U_l V_l
```

where `B` is shared by all blocks and `UV` is a small layer-local correction. Dense matrices
are composed transiently by this reference implementation and are never stored per layer.
The next kernel milestone will fuse basis composition with GEMM so even that transient matrix
does not exist.

Generation maintains spatial scene tokens across the entire clip and separate spatiotemporal
motion tokens. At each flow step, the scene changes only through the time-averaged update;
motion receives the full update. Blocks may reuse a cached residual when normalized feature
drift remains below the configured threshold and the cache age limit has not expired.

## Experimental discipline

Treat `benchmark.json` as a systems regression artifact, not as a trained quality result.
Quality comparisons become meaningful only when candidate and baseline are trained or
distilled on the same data and evaluation seed set. Change one feature at a time, retain its
configuration and checkpoint, and report median latency after warmup together with peak VRAM,
FLOPs, sparsity, cache rate, and held-out perceptual/temporal metrics.

The staged path is documented in [docs/ROADMAP.md](docs/ROADMAP.md).
