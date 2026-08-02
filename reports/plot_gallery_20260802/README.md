# 图集：reports 数据可视化（2026-08-02）

> 全部图表由 `rlproject-swift-improved/scripts/plot_v3_aggregate.py` 从已有
> reports CSV 批量生成（oracle 分布图由一次性脚本生成）。
> **⚠️ 所有数据均为 `legacy-unaligned`**（最终环境对齐前产生，见
> `reports/README.md` 有效性说明）：只用于方法学诊断与工具验证，
> **不得作为正式 V3 验收或论文主结果**。正式 V3 数据产生后，用同一工具
> 重新出图即可（脚本可复用）。

## 生成方法

```bash
python scripts/plot_v3_aggregate.py \
  --csv <aggregate 或 degradation CSV> \
  --axis-column depth_scale \   # 单轴聚合时指定
  --output <out.png> \
  --title "<实验名> (legacy-unaligned, diagnostic)"
```

- A 类（多轴退化 CSV，`axis` 列）：自动按轴分组 → 每轴一条曲线
- B 类（单轴聚合，`depth_scale,successes,...`）：`--axis-column` 指定 → 曲线 + Wilson CI 带
- C 类（消融，`ablation`/`label` 列）：自动检测 → 横向条形图 + CI 误差棒

## 图集索引（20 张）

### A. 多轴退化曲线（8）

| 文件 | 数据来源 | 轴 |
|------|---------|-----|
| `phase_v2_smoke_5x5x10_20260727_082037.png` | v2 五轴 smoke gate | gaussian/resolution/depth_noise/lighting/viewpoint_uncertainty |
| `phase_v2_formal_5x5x50_20260727_082212.png` | v2 五轴正式评估 | 同上 |
| `phase_v2_structural_smoke_4x5x10_20260727_083219.png` | 结构退化 smoke | depth_failure/occlusion/depth_scale/combined |
| `phase_v2_structural_formal_4x5x50_20260727_083353.png` | 结构退化正式 | 同上 |
| `phase_v2_depth_scale_confirm_5x200_20260727_100845.png` | 深度尺度确认 | depth_scale |
| `phase_v2_real_20260726.png` | v2 真实 GS 初始评估 | gaussian/resolution/depth_noise/lighting/viewpoint |
| `v3_clean_recovery_gate_seed0_5x100_20260729_053619.png` | V3 clean recovery gate | depth_scale |
| `v3_scale_curriculum_gate_seed2_5x100_20260728_103900.png` | scale curriculum gate | depth_scale |

### B. 聚合对比曲线（含 Wilson CI 带，4）

| 文件 | 数据来源 | 内容 |
|------|---------|------|
| `v3_scale_eval_3x5x200_20260728_065821.png` | 3 seeds 聚合 | depth_scale 退化 vs baseline |
| `v3_scale_weighted_eval_3x5x200_20260728_084229.png` | 3 seeds 聚合 | weighted 训练退化 |
| `v3_scale_curriculum_eval_3x5x200_20260728_124659.png` | 3 seeds 聚合 | curriculum 退化 vs uniform/weighted/baseline |
| `v3_checkpoint_selection_comparison_20260729.png` | checkpoint 对比 | clean_best/robust_best 等多 variant 曲线 |

### C. 消融对比条形图（7）

| 文件 | 数据来源 | 行数 |
|------|---------|-----:|
| `v3b_input_gate_seed2_7x100_20260729.png` | 输入消融 gate | 7 |
| `v3_aligned_visual_gate_seed2_7x100_20260729.png` | 对齐视觉 gate | 7 |
| `v3_geodesic_avoidance_gate_seed2_2x200_20260730.png` | 测地绕障 gate | 2 |
| `v3_gscloud_aligned_gate_seed2_2x200_20260730.png` | GS 点云对齐 gate | 2 |
| `v3_hybrid_geodesic_gate_seed2_2x200_20260730.png` | 混合测地 gate | 2 |
| `v3_waypoint_obs_gate_seed2_2x200_20260730.png` | waypoint 观测 gate | 2 |
| `entropy_clean_6x200_20260727_081142.png` | 熵对比 | 6 |

### D. 分布图（1）

| 文件 | 数据来源 | 内容 |
|------|---------|------|
| `v3_waypoint_oracle_200_20260730.png` | waypoint oracle 200 | path_efficiency + min_clearance 直方图（效率中位 1.013，净空中位 0.21m） |

## 正式 V3 使用流程

1. 正式 V3 评估产出 `aggregate_summary.csv`（五轴各一份，含 `ci95_low/high`）
2. `plot_v3_aggregate.py` 直接出图（A/B 布局已支持，无需改代码）
3. 显著性声明用 `utils/stats.py`（paired_bootstrap / mcnemar）基于 per-episode 结果
4. 论文图在选定后人工筛选替换本目录的 legacy 图
