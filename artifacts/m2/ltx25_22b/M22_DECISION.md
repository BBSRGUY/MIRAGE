# MIRAGE M2.2 decision

**PASS — M2 structural gate PASSED**

## Mandatory criteria

- PASS — `compression_ratio`: 3.0009005387969414 (threshold 3.0)
- PASS — `worst_block_relative_error`: 0.039324380457401276 (threshold 0.05)
- PASS — `worst_block_cosine`: 0.999433159828186 (threshold 0.995)
- PASS — `all_projection_families`: ['attn.k', 'attn.out', 'attn.q', 'attn.v', 'ff-in', 'ff-out'] (threshold ['attn.k', 'attn.out', 'attn.q', 'attn.v', 'ff-in', 'ff-out'])
- PASS — `evaluation_isolation`: True (threshold True)
- PASS — `live_weight_swap_verified`: 480 (threshold 1)

## Scientific conclusion

Shared-basis compression is falsified as the default LTX-2.5 representation; the passing portfolio uses independent grouped INT4 and rowwise INT8.
The selected portfolio contains 140 grouped-INT4 and 148 rowwise-INT8 projections.
Structured tile-sparse residuals were rejected, and temporal predictive execution remains frozen.

Proceed to M3: **YES**. Shared bases must be an ablation, not the M3 default.
