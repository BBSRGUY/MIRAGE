# M2.2: Heterogeneous compression and functional reconstruction

M2.2 tests the narrower hypothesis supported by M2.1: trained video-transformer projections are
not uniformly compressible. Shared bases are candidate representations for some families and
layers, not a universal MIRAGE requirement.

The experiment proceeds in this order:

1. Measure singular-energy curves of each remaining matrix error after the strongest admissible
   M2.1 reconstruction.
2. For broad residuals, test structured tile/block-sparse exceptions in addition to low rank.
3. Build per-layer candidates including FULL, precision-reduced independent, basis plus low-rank,
   and basis plus sparse residual representations.
4. Allocate candidates globally under a whole-transformer compression constraint rather than
   requiring every family to reach 3× independently.
5. Jointly fit attention and FFN functional subgraphs on training prompts.
6. Judge the final portfolio on complete held-out block outputs.

Mandatory M2.2 acceptance is aggregate compression ≥3×, held-out block cosine ≥0.995, held-out
block relative error ≤0.05, all projection families tested, and strict train/evaluation isolation.
Projection metrics remain diagnostics. M3 stays blocked until the generated decision artifact
returns PASS.
