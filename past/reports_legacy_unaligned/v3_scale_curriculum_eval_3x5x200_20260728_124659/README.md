# V3 formal curriculum evaluation

Training: `v3_scale_curriculum_3x500_20260728_110541`

Evaluation: `v3_scale_curriculum_eval_3x5x200_20260728_124659`

Three clean-selected best checkpoints were evaluated with real gsplat rendering
on the same 200 episodes at each scale, for 3,000 episodes total.

| Scale | Pooled SR | Wilson 95% CI | Collision | Timeout | vs uniform V3 |
|------:|----------:|--------------:|----------:|--------:|--------------:|
| 1.0× | 78.33% | 74.86–81.44% | 21.67% | 0.00% | −3.50 pp |
| 0.75× | 78.00% | 74.51–81.13% | 21.83% | 0.17% | −3.17 pp |
| 0.5× | 68.00% | 64.16–71.61% | 22.17% | 9.83% | +10.83 pp |
| 0.25× | 78.17% | 74.69–81.29% | 21.83% | 0.00% | +2.00 pp |
| 0.1× | 63.33% | 59.40–67.09% | 24.17% | 12.50% | −14.00 pp |

The formal gate fails all four criteria. In particular, seed2 reaches only
53.5% at 0.5× and pooled 0.1× timeout is 12.5%.

This result also exposes a checkpoint-selection confound. Training saves only
the checkpoint with the highest clean success rate. The seed1 and seed2 files
were written before the robustness stage completed, so the evaluated models
are not consistently the final curriculum policies. The pilot's best clean
checkpoint happened to be its final checkpoint, which explains part of the
pilot/formal discrepancy.

Before spending another 3×500 run, always save the final checkpoint and add a
small multi-scale validation score for robust checkpoint selection.
