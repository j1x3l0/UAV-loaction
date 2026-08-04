# V3 weighted depth-scale evaluation

Run: `v3_scale_weighted_eval_3x5x200_20260728_084229`

Three best checkpoints trained with scale probabilities
`20%/20%/40%/10%/10%` were evaluated using real gsplat rendering. Each
checkpoint saw the same 200 deterministic episodes at each of five scales,
for 3,000 episodes total.

| Scale | Pooled SR | Wilson 95% CI | CR | Timeout | Uniform V3 | Change |
|------:|----------:|--------------:|---:|--------:|-----------:|-------:|
| 1.0× | 77.00% | 73.47–80.19% | 21.17% | 1.83% | 81.83% | −4.83 pp |
| 0.75× | 76.83% | 73.29–80.03% | 22.17% | 1.00% | 81.17% | −4.33 pp |
| 0.5× | 73.67% | 70.00–77.03% | 24.50% | 1.83% | 57.17% | +16.50 pp |
| 0.25× | 51.17% | 47.17–55.15% | 26.00% | 22.83% | 76.17% | −25.00 pp |
| 0.1× | 60.17% | 56.20–64.01% | 26.00% | 13.83% | 77.33% | −17.17 pp |

The 0.5× per-seed results are 76.5%, 64.0%, and 80.5%, so the original
seed-specific hole is repaired. The experiment nevertheless fails three of
four acceptance gates: clean is below 80%, the two extreme pooled levels are
below 70%, and extreme-level timeout exceeds 10%.

The next run should be a staged curriculum rather than another fixed
distribution. Add probability logging and gate the schedule with one
200-episode-seed pilot before spending another 3×500 run.
