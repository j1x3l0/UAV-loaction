# 同源Gaussian几何视觉必要性门控

## 设置

- 模型：`v3_gscloud_aligned_seed2_100_20260730/seed2_best.pth`
- 渲染与碰撞：同一份 `gate_mid_new_gs.ply`
- 真实gsplat、运动相机、确定性策略
- base seed：`20260728`
- baseline与const-depth各200 episodes，使用相同episode seeds

## 结果

| 配置 | 总SR | Wilson 95% CI | Clear SR | Avoidance SR | Collision | Timeout |
|------|-----:|--------------:|---------:|-------------:|----------:|--------:|
| baseline | 31.0% | 25.0–37.7% | 56.7% | 6.8% | 69.0% | 0.0% |
| const-depth | 9.5% | 6.2–14.4% | 15.5% | 3.9% | 86.5% | 4.0% |

baseline相对const-depth提升21.5个百分点，且总SR Wilson区间不重叠，证明
策略已经利用真实3DGS深度。另一方面，baseline在103个avoidance任务中只成功
7次，碰撞率93.2%，说明当前策略主要学会直达任务，尚未学会可靠绕障。

## 决策

视觉必要性通过，导航能力门控失败。禁止直接扩展3×500。下一步采用自由空间
测地进度奖励，并按10%/30%/50%逐步提高avoidance采样比例，先运行单seed
200-update小试。
