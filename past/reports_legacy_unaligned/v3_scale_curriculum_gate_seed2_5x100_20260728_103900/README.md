# V3 curriculum pilot gate

Training run: `v3_scale_curriculum_seed2_200_20260728_101116`

Evaluation run: `v3_scale_curriculum_gate_seed2_5x100_20260728_103900`

The seed2 curriculum pilot trained for 200 updates with three stages:

- foundation (0–30%): `35/25/20/10/10%`
- transition (30–70%): `25/20/30/15/10%`
- robustness (70–100%): `25/15/25/20/15%`

Final clean evaluation during training reached 85% and policy entropy was 2.80.
The best checkpoint then underwent a real-gsplat five-level gate using 100
common-random-number episodes per scale.

| Scale | SR | Wilson 95% CI | Collision | Timeout | Gate |
|------:|---:|--------------:|----------:|--------:|:----:|
| 1.0× | 83% | 74.45–89.11% | 17% | 0% | pass |
| 0.75× | 80% | 71.12–86.66% | 20% | 0% | pass |
| 0.5× | 82% | 73.33–88.30% | 18% | 0% | pass |
| 0.25× | 87% | 79.02–92.24% | 13% | 0% | pass |
| 0.1× | 78% | 68.93–85.00% | 22% | 0% | pass |

All four predeclared acceptance criteria passed. Proceed to formal
three-seed, 500-update curriculum training and the same 3×5×200 independent
evaluation. The pilot is encouraging but is not a substitute for the formal
multi-seed result.
