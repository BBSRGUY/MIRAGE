# MIRAGE M3 status

**ACTIVE / FOUNDATION PASS — final M3 gate NOT PASSED**

The first MIRAGE-S CUDA smoke run completed 20 optimizer steps and produced a resumable EMA
checkpoint. Flow loss decreased from 2.2877 to 1.2072. Peak allocated training VRAM was
52,386,816 bytes.

The cache-disabled, teacher-free heterogeneous inference runtime used 12 grouped-INT4 and 12
rowwise-INT8 projections. Its conservative resident estimate was 4,290,560 bytes. Across two
fixed held-out prompts it retained 40.14–41.31 dB PSNR and 0.99874–0.99913 global SSIM versus
the same trained independent BF16 checkpoint.

Six compact offline LTX-2.5 teacher signatures were constructed from 1,200 M2 activation
records. Teacher weights are not required by the M3 dataset or inference runtime.

Checkpoint provenance is active. The regenerated smoke checkpoint records Git commit
`f2be399ae2896d06c5393abd0bc90bed41b8f7b8`, a dirty source-snapshot hash, canonical config hash,
dataset and teacher-feature hashes, model-state hash, and checkpoint-container hash. Strict resume
and the external sidecar hash were both verified.

This is not a quality milestone pass. The current checkpoint learned only from the deterministic
synthetic systems fixture. Real AV training, behavior-transfer measurement, CLIP/LPIPS/VBench/
FVD evaluation, identity and AV-sync evaluation, and matched-budget ablations remain required.

The frozen M3-v0 real-data foundation is now implemented: deterministic Panda-70M selection,
source-isolated splits, AV normalization, exact/perceptual deduplication, WebDataset sharding,
shard auditing, direct tar-stream loading, and WebDataset checkpoint provenance. A generated
two-second AV fixture passes the complete path. This validates the machinery only; no Panda media
has been downloaded and no real-data training or quality claim has been made.
