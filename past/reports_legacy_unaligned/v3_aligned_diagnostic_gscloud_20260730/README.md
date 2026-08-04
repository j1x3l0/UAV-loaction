# V3d 深度—碰撞几何对齐诊断

日期：2026-07-30
渲染器：真实 gsplat
场景：`gate_mid_new_gs.ply`
采样：相同 base seed `20260728`，30 个相机位姿

## 结果

| 碰撞几何 | 可比较位姿 | 中位绝对误差 | P90误差 | 相关系数 | 判定 |
|---|---:|---:|---:|---:|---|
| 原 `gate_mid_new.ply` | 30/30 | 1.29m | 3.15m | 0.676 | 失败 |
| 3DGS Gaussian中心 | 30/30 | 0.37m | 1.45m | 0.949 | 通过 |

原碰撞点云的误差集中在“直线路径可达”样本：中位绝对误差2.24m、相关系数
0.497；需绕障样本为0.38m、相关系数0.869。这表明原几何漏掉了部分渲染器
可见表面，使策略观测与碰撞/任务标签冲突。

## 结论与下一步

后续实验使用3DGS Gaussian中心作为同源碰撞几何。已启动
`v3_gscloud_aligned_seed2_100_20260730` 单种子100-update小试。训练完成后
运行 baseline 与 const-depth 各200 episodes 的 clear/avoidance 分层门控；
门控未通过则不扩展3×500。

原始记录：

- 原碰撞点云：`../v3_aligned_diagnostic_20260730/scene_alignment.json`
- 同源Gaussian中心：`scene_alignment.json`
