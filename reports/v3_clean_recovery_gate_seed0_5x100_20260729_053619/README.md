# V3 clean-recovery single-seed gate

Training run: `v3_clean_recovery_seed0_100_20260729_032959`

Evaluation run: `v3_clean_recovery_gate_seed0_5x100_20260729_053619`

The seed0 robust-best checkpoint was fine-tuned for 100 updates with a
`60/15/10/10/5%` clean-focused scale distribution and learning rate `3e-5`.
The recovered robust-best was evaluated with real gsplat rendering on 100
common-random-number episodes per scale.

| Scale | SR | Collision | Timeout | Gate |
|------:|---:|----------:|--------:|:----:|
| 1.0× | 75% | 25% | 0% | fail |
| 0.75× | 74% | 25% | 1% | pass |
| 0.5× | 78% | 21% | 1% | pass |
| 0.25× | 78% | 22% | 0% | pass |
| 0.1× | 73% | 27% | 0% | pass |

The clean target (`>=80%`) failed while all degradation and timeout targets
passed. Clean recovery did not improve the original seed0 robust-best clean
result (76% in the 200-episode formal evaluation). Per the predeclared stop
rule, this route is terminated and will not be expanded to three seeds.
