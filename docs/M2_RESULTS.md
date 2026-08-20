# M2 LTX-2.5 empirical results

This is the first real MIRAGE M2 decision run against the official dense BF16 LTX-2.5 22B
audio/video transformer. It used four fixed training prompts and two held-out evaluation prompts,
eight distilled flow steps, 256×256 latent-video probes, and seed 1337 on an NVIDIA GeForce RTX
5090 Laptop GPU with 24 GiB VRAM.

## Decision

**PARTIAL — do not proceed to full M3 distillation.**

| Criterion | Measured | Required | Result |
| --- | ---: | ---: | --- |
| Held-out activation cosine | 0.891832 | ≥ 0.995 | FAIL |
| Normalized activation error | 0.442342 | ≤ 0.05 | FAIL |
| Compression ratio | 10.9714× | ≥ 3× | PASS |
| REUSE + PREDICT coverage | 0% | ≥ 40% | FAIL |
| Scene rank-1 energy | 82.3689% | ≥ 60% | PASS |

The best measured attention candidate was `attn.k`, four shared bases and rank-16 residuals,
after 100 behavior-fitting steps. Behavior fitting improved held-out error from 0.550656 to
0.442342 and cosine from 0.813136 to 0.891832, but remained far outside the acceptance limits.
No tested candidate met both held-out fidelity limits, so there is no acceptable compression
ratio to carry into M3.

## Temporal execution result

At the configured strict routing thresholds, the held-out policy selected 0% REUSE, 0% PREDICT,
and 100% EXECUTE across 70 block transitions. Mean predictor error was 0.460383. Direct cache
reuse was also weak: 0% hit rate at 1% error, 4.7619% at 2% error, and 14.7619% at 5% error.
The 50.9524% rate at a 10% threshold is recorded for analysis but is not accepted as fidelity
retention.

## Scene/motion result

Rank-1 temporal structure explained 82.3689% of measured video-block energy on average, passing
the structural threshold. Mean scene-state drift across adjacent denoising steps was 0.188748.
This supports continued scene/motion research, but it is an internal feature result rather than
evidence of generated-video quality.

## Known gap

Sixteen `ff.in` and `ff.out` candidates were skipped because the current dense reference fitter
estimated 39–40 GB peak fitting memory, above the configured 18-GiB budget. A streamed or
randomized solver is required before the feed-forward families can receive a scientific verdict.

## Archived evidence

The repository tracks the compact decision package under `artifacts/m2/ltx25_22b`: decision and
provenance, basis/activation sweep summaries, sensitivity map, temporal redundancy, cache,
predictor, and scene/motion reports. Raw activation chunks, dense extracted weights, fitted
factor tensors, and predictor tensors remain ignored because the local feature store is 37.9 GB.

These results measure teacher weight reconstruction, held-out local activation reconstruction,
and internal temporal behavior. They do not establish perceptual equivalence or full-generation
quality.
