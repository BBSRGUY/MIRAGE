# MIRAGE M2 decision

**PARTIAL**

## Acceptance criteria

- FAIL — `validation_activation_cosine`: 0.927778 (threshold 0.995)
- FAIL — `normalized_activation_error`: 0.328203 (threshold 0.05)
- PASS — `compression_ratio`: 5.08504 (threshold 3.0)
- FAIL — `reuse_predict_coverage`: 0 (threshold 0.4)
- PASS — `scene_low_rank_energy`: 0.823689 (threshold 0.6)

## Decision

Proceed to full distillation: **NO**.

## Limitations

- 16 feed-forward candidates across ff-in, ff-out were not evaluated because the dense reference fitter exceeded the 18-GiB fitting budget.

## Scope

This decision covers trained-teacher weight reconstruction, held-out local activation fidelity, and internal temporal behavior. It does **not** demonstrate full-video perceptual equivalence.
