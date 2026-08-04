# V3d 场景对齐视觉必要性门控

日期：2026-07-29

## 协议

- 训练：`v3_aligned_geometry_seed2_100_20260729`
- checkpoint：`seed2_best.pth`
- 真实渲染：`gate_mid_new_gs.ply`
- 碰撞几何：`gate_mid_new.ply` 稠密点云 KD-tree
- 起终点：最大连通自由空间，50%直线路径受阻
- 相机：随速度方向，低速时朝向目标
- base seed：20260728
- 每配置：100 episodes

## 训练结果

- 100/100 updates，204,800 environment steps
- 训练用时：27分28秒
- 最佳内部 clean SR：20%
- 最终内部 clean SR：20%
- 最终五尺度小样本 SR：40% / 60% / 40% / 60% / 60%
- 最终 entropy：2.78

内部五尺度每档只有5 episodes，只用于checkpoint选择，不作为正式性能结论。

## 独立输入门控

| 配置 | SR | Wilson 95% CI | Collision | Timeout |
|------|---:|--------------:|----------:|--------:|
| baseline | 21% | 14.2–30.0% | 79% | 0% |
| const-depth | 24% | 16.7–33.2% | 76% | 0% |
| no-velocity | 5% | 2.2–11.2% | 95% | 0% |
| no-target-direction | 1% | 0.2–5.4% | 99% | 0% |
| no-depth + no-velocity | 5% | 2.2–11.2% | 95% | 0% |
| no-velocity + no-target-direction | 0% | 0–3.7% | 100% | 0% |
| all-inputs-ablated | 0% | 0–3.7% | 100% | 0% |

## 判定

门控失败，不扩展3×500：

1. 对齐baseline只有21%，尚未达到可用导航水平。
2. const-depth比正常depth高3个百分点，置信区间高度重叠；没有视觉必要性证据。
3. velocity和target-direction仍是主要有效输入。
4. 失败全部表现为碰撞而非timeout，下一步应诊断碰撞几何、相机运动、任务分层
   和奖励，而不是增加训练种子或训练轮数。

该结果证明场景闭环修复改变了任务难度，也否定了将旧82% SR直接解释为真实
3DGS场景导航性能的做法。
