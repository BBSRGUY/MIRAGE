# M2.1: Adaptive structural compression recovery

M2.1 addresses the two unresolved findings from the first real LTX-2.5 study: attention-family
shared bases missed the held-out fidelity limits, and the dense reference solver could not fit
the two large feed-forward projection families within the configured GPU budget.

## Streamed FF solver

The solver never constructs `[layers, out_features, in_features]`. It reads bounded row slices
from each Safetensors weight and accumulates the exact layer-space Gram matrix in FP64:

```text
Gram[i,j] = sum_row_chunks dot(W_i[chunk], W_j[chunk])
```

The 48×48 eigensystem produces PCA mixing coefficients. A second streamed pass constructs only
the requested basis tensors. Layer-specific low-rank residuals are then fitted one teacher matrix
at a time. Peak fitting memory is therefore controlled by the row chunk, basis bank, one dense
teacher matrix, and one residual—not three copies of the complete family stack.

## Hierarchical basis

Each projection family uses a small global bank plus cluster-specific banks:

```text
W_l = sum_i alpha[l,i] G_i + sum_j beta[l,j] C[group(l),j] + U_l V_l
```

For families with M2.0 sensitivity evidence, contiguous depth groups minimize within-group
variance of held-out activation error, cosine loss, and sensitivity score. Families without prior
evidence use deterministic equal-depth groups and are re-clustered after their first held-out run.

## Adaptive rank budget

Layer residual ranks are assigned from four increasing tiers ordered by measured sensitivity.
The allocator then reduces the least-sensitive ranks until the combined global bases, group
bases, coefficients, and actual non-padded residual parameters meet the configured global
compression target. Reported compression counts actual ranks, not padded checkpoint storage.

## Hard gate

M2.1 passes only if every configured projection family is evaluated on held-out prompts, the
worst family reaches cosine ≥ 0.995 and normalized error ≤ 0.05, and aggregate parameter
compression is ≥ 3×. No threshold is relaxed. M3 remains blocked until M2.2 also passes.
