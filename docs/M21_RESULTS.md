# M2.1 adaptive compression results

M2.1 was run against the same official BF16 LTX-2.5 22B extraction and fixed four-train/two-eval
prompt split as M2.0. It completed the previously missing `ff.in` and `ff.out` experiments using
the streamed solver, then evaluated hierarchical sharing, nonuniform residual ranks, and
activation-aware residual fitting.

## Decision

**FAIL — M2 gate blocked; do not proceed to M3.**

| Mandatory criterion | Measured | Required | Result |
| --- | ---: | ---: | --- |
| Worst-family held-out activation cosine | 0.824927 | ≥ 0.995 | FAIL |
| Worst-family normalized activation error | 0.486108 | ≤ 0.05 | FAIL |
| Aggregate parameter compression | 3.17683× | ≥ 3× | PASS |
| FF families evaluated | yes | yes | PASS |
| Held-out prompts | 2 | ≥ 1 | PASS |

## Strongest admissible configuration

The final configuration used one global basis, three bases in each of four behavior/depth groups,
and sensitivity-ranked residual tiers of 32, 64, 128, and 256. Aggregate parameter compression
was 3.1768×. A closed-form activation-metric residual refit used training activations only and
was scored on the two held-out prompts.

| Family | Compression | Held-out error | Held-out cosine |
| --- | ---: | ---: | ---: |
| `attn.k` | 3.03557× | 0.262144 | 0.953797 |
| `attn.out` | 3.03557× | 0.434058 | 0.850918 |
| `attn.q` | 3.03557× | 0.288778 | 0.942127 |
| `attn.v` | 3.03557× | 0.486108 | 0.824927 |
| `ff.in` | 3.25251× | 0.198215 | 0.971325 |
| `ff.out` | 3.25251× | 0.395492 | 0.875653 |

The FF result is informative: `ff.in` is substantially more compressible than the attention
output/value families, while `ff.out` is not. The missing FF study is resolved, but no family
reaches both strict fidelity limits.

## Delta-spectrum result

The residual-delta oracle was also constrained to at least 3× factor compression. Rank 8 was the
largest eligible tested rank. It explained 85.69% of held-out delta energy on average, produced
mean relative reconstruction error 0.3260, and met the ≤5% local-error limit on only 11.43% of
held-out transitions. Rank 32 was excluded from the decision because 32 sampled tokens make it
a full-rank, non-compressed reconstruction.

This oracle result does not justify a causal delta adapter yet, and no FLOP-reduction claim is
made.
