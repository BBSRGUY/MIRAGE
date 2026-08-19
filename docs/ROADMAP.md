# MIRAGE research roadmap

## M1 — Resident executable and measurement (current)

Acceptance: end-to-end video and audio tensors on a single CUDA device; no `.cpu()` operation
until final output transfer; preflight below 24 GiB; deterministic smoke test; dense comparison;
machine-readable telemetry. The repository meets this contract with untrained weights.

## M2 — Training and distillation

Add WebDataset ingestion, rectified-flow training, EMA checkpoints, gradient checkpointing,
and teacher-behavior losses for features, motion, temporal consistency, and output. Establish
a fixed evaluation set and report CLIP similarity, LPIPS, VBench subsets, FVD, audio-video sync,
and human preference confidence intervals. Train the dense reference and then ablate scene
decomposition, basis rank/count, sparse masks, cache thresholds, and step count individually.

## M3 — Teacher adapters

Create offline teacher-extraction adapters for LTX and MiniMax. Persist only dataset features;
teachers are never part of MIRAGE inference residency. Map teacher layers to shared bases via
low-rank initialization, then behavior-distill rather than minimizing weight reconstruction.

## M4 — Kernels

Fuse basis composition, low-rank residual application, quantized codebook lookup, and GEMM in
Triton/CUDA. Replace the masked dense SDPA reference with true block-sparse spatiotemporal
attention. Add FP8 basis storage, selected BF16 outliers, and calibrated FP4/FP6 residual paths.

## M5 — Benchmark and ablation release

Publish exact hardware/software manifests, peak allocated and reserved memory, cold and warm
latency distributions, generated frames per second, analytical and profiler FLOPs, kernel
occupancy, logical and effective sparsity, cache decisions by layer/step, and full quality
metrics. A change lands only with an apples-to-apples baseline and confidence intervals.

