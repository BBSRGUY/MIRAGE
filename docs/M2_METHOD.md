# M2: LTX-2.5 22B empirical study

## Scope

M2 tests whether MIRAGE's proposed compression and temporal mechanisms preserve the internal
behavior of the official LTX-2.5 22B distilled audio/video transformer. It does not claim that
weight reconstruction alone proves perceptual equivalence.

The structural source is the official dense checkpoint
`ltx-2.5-22b-distilled-transformer-bf16.safetensors`. The adapter requires metadata
`model_version=2.5.0`, transformer class `AVTransformer3DModel`, exactly 48 blocks, and BF16
projection tensors. INT8, GGUF, FP8, and NVFP4 files are rejected for structural analysis.

## Execution model

The official Lightricks LTX-2 code is used for the teacher. Dense block weights are streamed
from disk or pinned CPU memory into a small number of GPU slots; this makes the 22B teacher a
measurement instrument on a 24 GB GPU. This offload is confined to the teacher experiment.
The MIRAGE inference target remains fully GPU-resident with no CPU offload.

For every prompt and seed, M2 records sampled video/audio block inputs, outputs, residuals,
projection inputs, prompt conditioning, latent state, and timestep. Train and held-out prompt
splits are fixed before fitting. All random seeds, software versions, checkpoint identity,
hardware, and capture settings are written alongside the features.

## Experiments

1. `teacher-extract` captures trained weights and internal features in resumable chunks.
2. `m2-basis-sweep` fits shared bases independently per shape-compatible projection family.
3. `m2-activation-fit` measures held-out output cosine and normalized error, optionally fitting
   coefficients and low-rank residuals to behavior.
4. `m2-temporal-probe` measures feature and residual drift across the trained eight-step schedule.
5. `m2-cache-analysis` evaluates reuse/predict/compute policies without executing a fake model.
6. `m2-predictor-fit` trains a residual predictor and evaluates held-out prediction error.
7. `m2-scene-motion` measures static low-rank energy and temporal residual energy.
8. `m2-report` applies configured PASS/PARTIAL/FAIL thresholds and records missing evidence.

Each basis count, residual rank, and cache threshold is a separate artifact. Failed or
memory-infeasible combinations are recorded rather than silently replaced with synthetic data.

## Reproduction

Use the isolated runtime created from the existing CUDA environment:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe -m mirage.cli teacher-extract --config configs/m2_ltx_teacher.json
.\.venv\Scripts\python.exe -m mirage.cli m2-basis-sweep --config configs/m2_ltx_teacher.json
.\.venv\Scripts\python.exe -m mirage.cli m2-report --config configs/m2_ltx_teacher.json
```

The default extraction is deliberately small (six prompts, 9 frames at 256x256) to validate
the complete measurement path before scaling the prompt set and resolution.
