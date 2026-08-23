# sv_1007 视觉 RL 实验结果汇总（阶段总结）

日期：2026-08-23。汇总 sv_1007 主场景全部实验结果、结论与 PR 索引。
本目录收集所有实验报告（12 个），供快速审阅。

> **速览**：先看 `SUMMARY.md`（一页纸核心结论）。以下为完整实验矩阵。

## 一、实验矩阵总览

### 主场景训练（sv_1007，22×25×9m 完整 gate）

| 队列 | 5 seeds best_SR | 均值 |
|------|:---:|:---:|
| clean（3000ep）| 43/60/55/64/44 | **53.2%** |
| curriculum v1（scale_curriculum）| 41/40/51/34/36 | **40.4%** |
| **curriculum v2（clean-heavy）** | 50/51/43 | **48.0%** |

### 退化鲁棒性（7 轴 × 5 档 × 50ep）
> `px4_sv1007_degradation_20260805/`

**clean 全面优于 curriculum v1**（无退化档 +20.8pp），curriculum 鲁棒优势不泛化到复杂场景。

### 输入必要性（输入消融）
> `px4_sv1007_ablation_20260806/` + `avoid_ablation_20260806/`

| 条件 | 混合任务 | 纯避障 |
|------|:---:|:---:|
| baseline | 54.0% | 34.4% |
| const_depth | 37.2%（−16.8pp）| **6.0%（−28.4pp）** |
| no_velocity | 7.2% | 6.4% |
| no_target_dir | 0.8% | — |

**深度必要性在纯避障下强成立**（−28.4pp）。

### 结构消融
> `px4_v3_arch_ablation_20260808/`

| 消融 | 均值 | vs baseline 53.2% |
|------|:---:|:---:|
| rgb | 48.0% | −5.2pp |
| shallow_cnn | 48.0% | −5.2pp |
| no_privileged_critic | 40.3% | −12.9pp |

### 配对统计检验（固定 seed，逐 episode）
> `px4_sv1007_paired_stats_20260809/`

| 对比 | 显著 seed 数 |
|------|:---:|
| baseline vs const_depth | **4/5** |
| baseline vs no_velocity | 5/5 |
| baseline vs no_target_dir | 5/5 |
| PPO vs BC | 2/3 |
| clean vs curriculum v1 | 4/5 |

### V3c 跨场景
> `px4_v3c_cross_scene_20260807/` + `px4_v3c_multiscene_20260807/`

| 模型 | sv_1007 | left | right |
|------|:---:|:---:|:---:|
| 单场景 sv_1007 | 42% | **0%** | **0%** |
| 3 场景联合（均值）| **28.7%** | **26.7%** | **37.3%** |

**跨场景泛化弱**（27-37%），多场景联合部分缓解但未达 40-50%。4 场景尝试失败（见 `docs/cross-scene-conclusion.md`），结论收缩为"域间迁移困难"。

### BC 基线
> `px4_v3_bc_baseline_20260809/`

BC 32.0% < RL 53.2%（−21.2pp），RL 优于 BC。

### 数据清洗
> `px4_sv1007_clean_20260804/`

16 → 5 障碍簇 + 地面，清洗前后 SR 无显著差异（一致性通过）。

### PX4 真实位姿注册
> `px4_sv1007_registration_20260817/`

悬停遥测回放验证：中央像素深度 0.969m ≈ 悬停高度 0.95m，坐标映射一致。

### 可复现性
> `px4_reproducibility_20260810/`

完整配置、种子、统计摘要 + 70 个模型 SHA256 哈希。

### Curriculum v2
> `px4_curriculum_v2_20260815/`

正常性能降 ≤5pp 达成（46% > clean 40%），保留尺度退化优势。

## 二、核心结论

1. **深度必要性成立**：纯避障 −28.4pp，配对 4/5 显著
2. **深度 > RGB**，深 CNN > 浅 CNN，特权 Critic 必要
3. **RL > BC**（32% vs 53.2%）
4. **curriculum v2** 正常性能达标，保留尺度鲁棒性
5. **跨场景泛化弱**（27-37%），结论收缩为方法边界
6. **PX4 对齐**：相机一致化 + 悬停回放验证，非全飞行闭环

## 三、GitHub PR 索引

| PR | 内容 |
|:---:|------|
| #16 | sv_1007 清洗工具 + 一致性校验 |
| #17 | sv_1007 退化 + 输入消融分析 |
| #18 | V3c 跨场景 + 多场景联合 + 结构消融 |
| #19 | BC 基线 |
| #20 | 配对统计检验 + 可复现报告 + curriculum v2 |
| #21 | PX4 真实位姿注册验证 |

## 四、诚实边界

- 跨场景泛化失败（方法边界，需 domain randomization / 更通用表征）
- PX4 仅悬停回放，非在线闭环
- 4 场景训练存在 checkpoint 可复现性问题（未纳入）

## 五、文件

本目录收集全部 12 个实验报告。模型与原始数据在 `local_results/`（gitignored）。
