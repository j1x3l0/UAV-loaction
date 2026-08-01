# High-entropy baseline: 3 × 500 episodes

- Server run ID: `b0_entropy_sign_fixed_3x500_20260726_191516`
- Renderer: real gsplat
- Scene: `gate_mid_new_gs.ply`
- Seeds: 0, 1, 2
- Classification: high-entropy baseline

| Seed | Final SR | Final CR | Final SR 95% CI | Final entropy | Final alpha |
|---:|---:|---:|---:|---:|---:|
| 0 | 86% | 14% | 77.9–91.5% | 4.26 | 0.02087 |
| 1 | 83% | 17% | 74.5–89.1% | 4.26 | 0.02076 |
| 2 | 85% | 15% | 76.7–90.7% | 4.26 | 0.02078 |
| Mean | 84.7% | 15.3% | — | 4.26 | 0.02080 |

The adaptive-entropy update sign is correct, but the initial coefficient was
`0.1`. More importantly, the forward-pass hard clamp on `exp(log_std)` pinned
all three policies at `std=1`, where the clamp supplied zero gradient. This run
is retained as a high-entropy comparison baseline, not as validation of the
entropy fix.
