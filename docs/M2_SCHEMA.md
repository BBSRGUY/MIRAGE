# M2 artifact schema

An experiment root contains:

```text
metadata.json
prompts.json
manifest.json
weights/*.safetensors
activations/*.safetensors
temporal/*.safetensors
reports/*.json
```

`manifest.json` is append-only and versioned. Each record has a stable ID, relative path,
kind (`weight`, `activation`, or `temporal`), sample ID, split, and metadata. Safetensors files
contain tensors only; experiment context stays in JSON so it can be inspected without loading
tensor payloads.

Normalized projection names encode block and LTX-2.5 modality, for example:

- `block.00.attn.q` — video self-attention query
- `block.00.cross_attn.k` — video-to-text cross-attention key
- `block.00.audio.ff.in` — audio feed-forward input projection
- `block.00.av.audio_to_video.out` — audio-to-video cross-modal output

Activation metadata includes prompt/sample split, seed, denoising step, timestep, hook type,
block/projection identity, modality, original shape, and sampled token indices. `metadata.json`
adds the git commit, complete configuration, checkpoint path/size/dtype/version, adapter type,
software versions, GPU identity, VRAM capacity, and train/evaluation sample counts.

Reports must distinguish direct measurements from derived estimates and must not fill absent
metrics with simulated values. The final decision is `PASS`, `PARTIAL`, or `FAIL`, accompanied
by every threshold, observed value, and missing prerequisite.
