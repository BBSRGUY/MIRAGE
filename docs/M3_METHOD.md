# M3 method: trainable MIRAGE behavior distillation

M3 trains a MIRAGE-native rectified-flow generator. The official LTX-2.5 teacher is used only
to construct offline behavior signatures; neither teacher code nor weights are loaded by the
training data stream or inference runtime.

## Fixed contracts

- Independent projections are the default trainable backend.
- The M2.2 heterogeneous grouped-INT4/rowwise-INT8 policy is applied for inference.
- Shared-basis projections are available only through `m3_shared_basis_ablation.json`.
- Persistent scene state is separated from zero-mean per-frame motion state before every flow
  prediction.
- Cache/predict execution is disabled in default M3 training and inference.
- Every checkpoint contains model, optimizer, EMA, step/epoch, configuration, and RNG state.
- Every checkpoint embeds SHA-256 provenance for the canonical config, dataset manifest, complete
  offline teacher-feature set, Git revision, dirty source snapshot, and tensor-level model state.
  An atomic sidecar additionally hashes the completed checkpoint container.
- Strict resume rejects a checkpoint if its experiment inputs or code snapshot do not match.
- Inference fails if its conservative resident-state estimate exceeds the configured budget.

## Data format

Production data uses a JSONL manifest. Each row contains:

```json
{"sample_id":"clip-0001","split":"train","prompt":"...","video":"clip-0001.safetensors","audio":"clip-0001-audio.safetensors","teacher_feature":"teacher/clip-0001.safetensors"}
```

The loader is iterable, rank/worker sharded, and deterministic. Tensor containers hold `video`
as `[frames,3,height,width]`, optional `audio` as `[samples]`, and optional teacher `signature`.
The `synthetic://` source is strictly a systems smoke fixture and cannot pass the M3 quality gate.

## Objective

For data latent `x1`, Gaussian `x0`, and sampled `t`, training uses

```text
xt = (1-t)x0 + t x1
v* = x1 - x0
```

The primary loss is velocity MSE. Auxiliary losses cover offline teacher behavior statistics,
temporal latent differences, clip identity state, and audio/motion energy synchronization.

## Evaluation discipline

EMA checkpoints are evaluated on fixed held-out prompts and seeds. The compressed runtime is
compared with the same trained independent BF16 checkpoint before quantization. Initial smoke
reports include numerical retention, latency, VRAM, FLOPs, attention density, and cache rate.
CLIP, LPIPS, VBench, FVD, identity, AV-sync, and human-preference claims remain pending until a
real held-out AV corpus and the required metric runtimes are configured.

M3 does not pass merely because smoke training executes. It passes only after coherent video,
real held-out evaluation, teacher-behavior transfer, matched-budget ablations, and the residency
contract all pass together.
