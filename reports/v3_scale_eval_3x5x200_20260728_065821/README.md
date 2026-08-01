# V3 depth-scale robustness evaluation

Run: `v3_scale_eval_3x5x200_20260728_065821`

Three best V3-Rand-Scale checkpoints were evaluated with real gsplat rendering on
the same 200 deterministic episode seeds at each of five depth scales. This is
3,000 episodes in total.

| Scale | Pooled SR (600 ep) | Wilson 95% CI | CR | Timeout | Baseline SR | Change |
|------:|-------------------:|--------------:|---:|--------:|------------:|-------:|
| 1.0× | 81.83% | 78.55–84.71% | 18.17% | 0.00% | 83.5% | −1.67 pp |
| 0.75× | 81.17% | 77.84–84.09% | 18.83% | 0.00% | 80.5% | +0.67 pp |
| 0.5× | 57.17% | 53.17–61.07% | 20.33% | 22.50% | 73.0% | −15.83 pp |
| 0.25× | 76.17% | 72.60–79.40% | 19.33% | 4.50% | 13.5% | +62.67 pp |
| 0.1× | 77.33% | 73.82–80.50% | 20.00% | 2.67% | 13.5% | +63.83 pp |

The intervention preserves clean performance and removes the catastrophic
failure at the two most extreme scales. It does not yet pass the stability
criterion: seed0 reaches only 14.5% SR at 0.5× with 66.5% timeouts, while seed1
and seed2 reach 75.0% and 82.0%. This non-monotonic, seed-specific robustness
hole makes uniform five-level sampling unsuitable as the final method.

Recommended next experiment: curriculum or reweighted sampling concentrated
around the 0.5× transition, with lower sampling probability for 0.25× and 0.1×.
Seed2 is the strongest current checkpoint across all five levels.
