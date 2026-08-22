# MIRAGE M3-v0 real AV corpus

M3-v0 is a bounded, reproducible 10,000-clip Panda-70M subset. It is intended to answer one
question before behavior distillation or architecture ablations: can the frozen MIRAGE-S model
learn coherent short video from real AV data within the single-GPU training and inference
envelope?

No corpus media or third-party metadata is committed to this repository. Panda-70M is published
for research use, and every source video's own license remains binding. Record and review source
licensing before redistributing clips or a trained checkpoint.

## Frozen selection contract

- 2–8 second clips, matching score at least 0.43, and the Panda `desirable` label;
- at most one shot segment and at most three clips from any source video;
- deterministic source-video hashing into train/validation/test, preventing source leakage;
- fixed motion/content quotas from `configs/m3_corpus_v0.json`;
- post-download duration, resolution, motion, exact-hash, perceptual-signature, video, and audio
  validation;
- square 256 px, 12 fps H.264 video plus mono 16 kHz AAC audio;
- deterministic WebDataset tar shards containing `<id>.mp4`, `<id>.txt`, and `<id>.json`;
- shard hashes, dataset/config hashes, split audit, audio coverage, and checkpoint provenance.

The model architecture remains identical to `m3_mirage_s.json`. The real baseline config disables
teacher behavior loss and retains independent weights, the scene/motion split, zero cache/predict
execution, and inference-only heterogeneous INT4/INT8 compression.

## Build stages

Download the official Panda-70M 2M metadata archive into `data/panda70m/train_2m.csv`. Use the
Panda-70M repository's fork of `video2dataset`; its README states that the upstream package is
not compatible with the Panda CSV schema. The selector emits a bounded request CSV accepted by
that fork and records the exact argument vector in `selection_report.json`.
The repository's `panda70m_10k_download.yaml` retains 360 px video and audio while limiting the
official fork to 4 processes × 4 threads. Output is incremental in 100-clip shards, so an
interrupted proof download resumes without requesting the 1.6 TB source corpus.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m mirage.cli m3-data-select --config configs/m3_corpus_v0.json

# From Panda-70M/dataset_dataloading, run the command recorded under
# video2dataset_command in data/mirage-m3-v0/selection/selection_report.json.

python -m mirage.cli m3-data-normalize --config configs/m3_corpus_v0.json
python -m mirage.cli m3-data-shard --config configs/m3_corpus_v0.json
python -m mirage.cli m3-data-audit --config configs/m3_corpus_v0.json
python -m mirage.cli m3-train --config configs/m3_mirage_s_real.json --device cuda
```

Do not start training unless `audit_report.json` has `"passed": true`. Selection and normalization
reports make shortfall explicit; the pipeline never silently substitutes rejected clips or mixes
validation/test sources into training. Teacher features are added only after this independent
real-data baseline is established.
