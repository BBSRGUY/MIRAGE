# MIRAGE research roadmap

MIRAGE is developed as a sequence of falsifiable systems milestones. Each milestone must preserve
all prior functionality, emit machine-readable evidence, and avoid claiming perceptual quality
until trained-model evaluation supports it.

## M1 — Resident executable and measurement — COMPLETE

Acceptance: end-to-end video and audio tensors on a single CUDA device; no `.cpu()` operation
until final output transfer; conservative residency preflight below 24 GiB; deterministic smoke
test; dense comparison; machine-readable latency, VRAM, FLOPs, sparsity, cache, precision, and
quality-regression telemetry.

M1 established that the MIRAGE execution graph is viable with untrained weights: shared-basis
transformer projections, persistent scene/motion state, sparse spatiotemporal masking,
predictive block reuse, dynamic precision, few-step flow generation, compact synchronized AV
decoding, and strict single-GPU residency.

## M2 — Teacher structural compressibility and temporal redundancy — COMPLETE / QUALIFIED PASS

- **M2.0 — Measurement infrastructure: COMPLETE.**
- **M2.1 — Adaptive structural compression recovery: COMPLETE / FAIL.**
- **M2.2 — Heterogeneous compression and functional reconstruction: COMPLETE / PASS.**
- **M2 structural gate: PASSED.**
- **Universal shared-basis hypothesis: FALSIFIED.**

M2 uses the official dense BF16 LTX-2.5 22B audio/video transformer as an offline measurement
instrument. Teacher weights are never part of MIRAGE runtime residency.

Implemented capabilities include:

- deterministic teacher extraction with resumable feature storage;
- normalized projection discovery and trained dense-weight extraction;
- shared-basis plus low-rank reconstruction sweeps;
- held-out activation-space fidelity measurement;
- temporal block/input/output/residual drift analysis;
- cache reuse threshold analysis;
- tiny residual-predictor analysis for REUSE / PREDICT / EXECUTE routing;
- persistent scene/motion decomposition probes;
- layer/projection sensitivity mapping;
- automatic PASS / PARTIAL / FAIL decision reporting.

The structural hypothesis under test is

```text
W_l ~= sum_i alpha[l,i] * B_i + U_l V_l
```

M2 evidence must distinguish raw weight reconstruction from held-out activation fidelity and
must not treat either as proof of perceptual equivalence.

The real LTX-2.5 run is archived in `artifacts/m2/ltx25_22b`. M2.0 was partial and M2.1 failed,
but M2.2 passed the unchanged structural gate with a heterogeneous independent-precision
portfolio. Predictive execution did not pass on the LTX denoising trajectory and remains frozen
until a trained MIRAGE-native trajectory exists.

## M2.1 — Adaptive structural compression recovery — COMPLETE / FAIL

M2.1 must recover held-out local activation fidelity without weakening the M2 thresholds. It
replaces the skipped dense FF fit with a row-streamed layer-Gram solver, evaluates all `ff.in`
and `ff.out` families, clusters contiguous layers by trained sensitivity signatures, fits global
plus cluster-specific basis banks, and allocates residual ranks non-uniformly under a global
parameter budget.

Mandatory acceptance:

- worst-family held-out activation cosine ≥ 0.995;
- worst-family normalized activation error ≤ 0.05;
- aggregate compression ratio ≥ 3×;
- both FF projection families evaluated;
- held-out prompts used for every activation result.

M2.1 emits `M21_DECISION.json` and cannot unblock M3 by itself. Temporal execution has its own
gate in M2.2.

The completed real LTX-2.5 run evaluated all six projection families and achieved 3.1768×
aggregate compression, but worst-family held-out activation error was 0.4861 and cosine was
0.8249. M2.1 therefore failed without weakening either fidelity threshold.

## M2.2 — Heterogeneous compression and functional reconstruction — COMPLETE / PASS

M2.2 abandons universal shared-basis compression. It chooses representations per family and
layer from independent, precision-reduced, shared-basis, low-rank, and structured-sparse
options under one whole-model parameter budget. The first diagnostic measures the spectrum of
the remaining M2.1 reconstruction error. Broad residuals trigger tile/block-sparse candidates
rather than ever-larger low-rank factors.

The primary behavioral measurement moves from isolated projections to complete attention and
FFN subgraphs, allowing compressed projections to co-adapt against teacher block outputs. The
mandatory final gate is aggregate model compression ≥3×, held-out block cosine ≥0.995, held-out
block relative error ≤0.05, every projection family tested, and no evaluation data used during
fitting.

Temporal predictive execution is frozen after its compressed oracle reached only 11.43% coverage
at ≤5% error. Scene/motion decomposition remains supported. REUSE/PREDICT/DELTA will be revisited
on MIRAGE's trained native trajectory rather than forced onto LTX's denoising trajectory.

The final held-out complete-block replay passed at 3.0009× aggregate compression: mean/worst
relative block error was 0.0160/0.0393 and mean/worst cosine was 0.999839/0.999433. The replay
verified 480 live projection swaps across two evaluation prompts and all six projection families.

The selected portfolio contains 140 grouped-INT4 and 148 rowwise-INT8 projections, with zero
shared-basis projections. Therefore M2.2 passes the structural compression gate while falsifying
shared bases as the default representation for LTX-2.5. Tile-sparse residuals were also rejected.
M3 must use the measured mixed-precision allocation as its compact independent-weight baseline;
shared bases remain a controlled trainable ablation and must earn their place on quality-per-byte.

## M3 — MIRAGE behavior distillation and trainable generative model — ACTIVE / FOUNDATION PASS

Objective: turn the validated M1 architecture and M2 teacher measurements into the first trained
MIRAGE generator while preserving the single-GPU inference goal.

M3 should not copy the teacher architecture. It should use M2 sensitivity results to allocate
capacity non-uniformly and distill teacher behavior into the MIRAGE-native representation.

Required work:

- WebDataset or equivalent streaming video/audio dataset ingestion;
- offline teacher-feature dataset construction using the M2 extraction format;
- rectified-flow / flow-matching training for MIRAGE latent generation;
- behavior distillation from teacher activations, residuals, motion, and outputs;
- heterogeneous grouped-INT4/rowwise-INT8 allocation as the compact independent-weight baseline;
- optional shared-basis initialization from the best M2 basis/rank configurations for controlled
  trainable ablation only;
- layer-family-specific precision, basis counts, and residual ranks driven by sensitivity maps;
- trainable persistent scene state and dynamic motion state;
- temporal consistency and identity-preservation losses;
- AV synchronization losses where audio is enabled;
- EMA checkpoints and resumable distributed/single-GPU training;
- gradient checkpointing and memory telemetry;
- fixed train/validation/evaluation prompt and clip splits;
- controlled ablations for scene decomposition, shared bases, cache/predict routing, sparse
  attention, and generation step count.

Primary M3 acceptance criteria:

1. MIRAGE produces coherent trained video rather than architecture-validation noise.
2. A held-out evaluation set is used for every reported quality result.
3. The complete inference model remains within the configured GPU residency budget with no CPU
   offload during generation.
4. Teacher-behavior losses show measurable transfer without requiring teacher weights at runtime.
5. Shared-basis compression is retained only if it delivers materially better quality-per-byte
   than the equivalently sized independent mixed-precision MIRAGE baseline.
6. Scene/motion decomposition improves temporal consistency or compute efficiency in controlled
   ablation.
7. Predictive execution is evaluated on the trained trajectory, not only synthetic/untrained
   states.
8. Training and inference configs, seeds, checkpoints, and telemetry are reproducible.

Quality evaluation should include, where practical: CLIP/image-text similarity, LPIPS, selected
VBench dimensions, FVD or an equivalent distributional video metric, temporal consistency,
identity consistency, AV-sync metrics, and blinded human preference on a fixed sample set.

M3 should end with a trained checkpoint family rather than a single model size. Target at least:

```text
MIRAGE-S   aggressive efficiency target
MIRAGE-M   balanced quality / speed target
MIRAGE-L   highest-quality model that still leaves meaningful headroom on a 24-GB GPU
```

The runtime target should aim substantially below the physical 24-GiB limit so activations,
attention workspace, codec state, prompt conditioning, and longer/higher-resolution sequences
have room to execute without offload.

The M3 foundation is now executable: JSONL/synthetic streaming AV ingestion, compact offline
teacher-signature construction, rectified-flow training, scene/motion state, temporal/identity/
AV-sync losses, EMA/resumable checkpoints, DDP support, gradient checkpointing, memory telemetry,
MIRAGE-S/M/L configs, M2-derived INT4/INT8 inference, and a shared-basis-only ablation config.

The first 20-step CUDA smoke run reduced flow loss from 2.2877 to 1.2072 and verified teacher-free,
cache-disabled quantized inference. This does not pass M3: it uses synthetic data and has not yet
met the coherent-video, real held-out quality, behavior-transfer, or matched-budget ablation gates.

## M4 — Native sparse and heterogeneous low-bit kernels

Objective: convert MIRAGE's logical sparsity and structural compression into real wall-clock and
VRAM gains.

Implement Triton/CUDA kernels that:

- fuse grouped-INT4 and rowwise-INT8 dequantization with GEMM;
- fuse shared-basis composition only if the M3 ablation reverses the M2 quality-per-byte result;
- fuse low-rank residual application where beneficial;
- execute true block-sparse spatiotemporal attention rather than dense masked SDPA;
- support calibrated FP8 basis storage;
- support selected BF16/FP16 sensitive outliers;
- support validated FP4/FP6 residual/codebook paths only where M2/M3 sensitivity permits;
- expose kernel-level telemetry including occupancy, bandwidth, effective sparsity, and latency.

Kernel optimization must preserve numerical/quality regression thresholds from the trained M3
reference. Speed claims must use warm benchmarks and profiler-backed measurements.

## M5 — Few-step streaming generation and predictive execution

Objective: reduce the amount of expensive backbone computation per generated clip/frame.

Advance the binary cache mechanism into a trained three-level controller:

```text
REUSE -> PREDICT -> EXECUTE
```

The controller should use measured state drift and learned residual prediction, with per-layer
and timestep-sensitive thresholds rather than one global heuristic.

M5 should also investigate:

- one-to-four-step distilled generation;
- causal or chunked streaming generation where quality permits;
- persistent world/entity state that updates more slowly than motion state;
- dynamic token/block execution based on actual scene change;
- recurrent detail refinement rather than full-backbone recomputation;
- long-clip state management without re-encoding unchanged content.

Acceptance is based on trained-model quality retention, measured backbone-execution reduction,
and end-to-end latency rather than analytical FLOPs alone.

## M6 — Compression and precision specialization

Objective: minimize model bytes without repeating the quality loss of uniform low-bit
quantization.

Use M2/M3 sensitivity information to develop non-uniform storage and execution policies:

- FP8/BF16 for sensitive semantic or normalization paths;
- lower precision for demonstrably robust blocks/projections;
- codebook/vector-quantized residual representations where validated;
- sparse high-precision outlier storage;
- learned basis dictionaries shared across compatible layer families;
- activation-aware calibration and behavior-preserving compression objectives.

Every compressed checkpoint must be compared against the same trained BF16 MIRAGE reference,
with perceptual, temporal, and AV metrics in addition to weight/activation reconstruction.

## M7 — Production benchmark and ablation release

Publish exact hardware/software manifests and apples-to-apples benchmark artifacts covering:

- peak allocated and reserved VRAM;
- model resident bytes by component;
- cold and warm latency distributions;
- generated frames per second and seconds per output second;
- analytical and profiler-measured FLOPs;
- kernel occupancy and memory bandwidth;
- logical versus effective attention sparsity;
- per-layer REUSE/PREDICT/EXECUTE decisions;
- prompt encoder, backbone, codec, and audio timing breakdowns;
- quality metrics with confidence intervals;
- controlled architecture ablations;
- comparison against relevant LTX/MiniMax-class local pipelines at matched resolution, frame
  count, step count, and hardware.

A performance change lands only with an apples-to-apples baseline. A compression change lands
only when quality loss is measured. A quality claim lands only on held-out trained-model outputs.

## Long-term target

MIRAGE should ultimately make high-quality local video generation comfortable rather than merely
possible on a 24-GB GPU: no CPU offload during generation, meaningful VRAM headroom, few-step
inference, true sparse compute, compact resident weights, and generation speed driven by what
changes in the scene rather than repeatedly recomputing the entire video state.
