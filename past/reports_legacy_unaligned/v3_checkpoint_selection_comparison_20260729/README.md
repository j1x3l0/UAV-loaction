# V3 checkpoint selection comparison

Training run: `v3_curriculum_ckptfix_3x500_20260728_133555`

All variants come from the same three training seeds and use the same
evaluation base seed (`20260728`). Robust-best uses 200 episodes per seed and
scale; clean-best and final use 100.

| Scale | Clean-best SR | Final SR | Robust-best SR | Robust 95% CI |
|------:|--------------:|---------:|---------------:|--------------:|
| 1.0× | 73.67% | 70.00% | 77.83% | 74.34–80.97% |
| 0.75× | 72.00% | 72.67% | 78.83% | 75.39–81.91% |
| 0.5× | 69.00% | 71.00% | 77.00% | 73.47–80.19% |
| 0.25× | 65.00% | 75.33% | 75.83% | 72.25–79.09% |
| 0.1× | 72.33% | 77.00% | 76.00% | 72.42–79.25% |

Robust-best improves over clean-best at every scale by 3.67–10.83 percentage
points. It improves over final at the first four scales and is one point lower
at 0.1×. Its pooled timeout is at most 1.5% at every scale, and its per-seed
0.5× SR is 77.5%, 67.0%, and 86.5%.

The robust checkpoint passes three of four formal criteria:

- clean pooled SR ≥80%: **fail** (77.83%)
- every pooled scale ≥70%: **pass**
- every seed at 0.5× ≥60%: **pass**
- every pooled timeout ≤10%: **pass**

The checkpoint-selection fix is therefore effective. The remaining issue is a
small clean-performance gap rather than degradation robustness. A full new
training run is not justified yet; first test a conservative clean/robust
selection constraint or short clean fine-tuning from robust-best.
