# sv_1007 纯避障输入消融（深度必要性）

日期：2026-08-06。纯避障 episode（`--avoidance-probability 1.0`）下验证深度是否
必要。衔接旧 D1 实验（gate_mid_new 纯避障诱导失败），在新主场景 sv_1007 上重做。

## 方法

- **模型**：`saved_models/v3_sv1007/seed*_3000_final.pth`（clean 5 seeds）
- **评估**：`scripts/eval_v3_ablation.py --avoidance-probability 1.0 --ablation baseline,const_depth,no_velocity`，50ep/配置
- **GPU0 only**
- 原始数据：`reports/px4_sv1007_avoid_ablation_20260806/seed{0-4}.json`

## 结果（SR 均值 5 seeds，纯避障）

| 消融 | seed0 | seed1 | seed2 | seed3 | seed4 | 均值 | vs baseline |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **baseline** | 20 | 48 | 38 | 36 | 30 | **34.4%** | — |
| **const_depth**（去深度）| 6 | 4 | 4 | 4 | 12 | **6.0%** | **−28.4pp** |
| no_velocity（去速度）| 2 | 10 | 6 | 10 | 4 | **6.4%** | −28.0pp |

## 结论：深度必要性在 sv_1007 纯避障下强成立

- **baseline − const_depth = 28.4pp**（5 seeds 全部 const_depth ≤ 12%，baseline 普遍 20-48%）。
- **对比旧 D1（gate_mid_new）**：D1 在旧场景纯避障下 radius 扫描未能诱导深度必要性
  （baseline 与 const_depth 一起崩塌）；sv_1007 上 baseline 可学（34.4%）而
  const_depth 崩塌（6.0%），**深度必要性被实证支持**。
- 可能的解释：sv_1007 更大更复杂（完整 gate，22×25×9m），障碍密度和多样性足够，
  深度成为避障的实质信号；旧场景 gate_mid_new 太简单，深度可被其他线索替代。
- 与 D3-lite（合成稠密走廊，深度必要 54% vs 0%）和 sv_1007 混合任务（深度 16.8pp）
  一起，支持 **"深度必要性随障碍密度/场景复杂度增强"** 的论文叙事。

## 待办

- [ ] 纯避障结果与 D3-lite、混合任务消融合并，统一"深度必要性"叙事
- [ ] 可考虑配对统计（McNemar/bootstrap）确认 28.4pp 显著（5 seeds 数据足够）

## 文件

- `seed{0-4}.json`：各 seed 逐配置 SR
- 服务器日志：`/root/px4-deploy/sv1007_avoid_ablation.log`
