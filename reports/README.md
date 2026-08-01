# Experiment reports index

`reports/` 保存支撑阶段结论的实验摘要。训练日志、模型、PID 和未筛选的中间产物
只在本地保留，不应提交到 Git。

> **有效性说明（2026-08-01）：** 当前目录中的 V1/V2/V3 结果均产生于最终
> 仿真环境对齐之前，统一视为 `legacy-unaligned` 历史诊断。它们可以用于验证
> 代码、分析失败模式和设计新实验，但不得作为正式 V3 验收或论文主结果。
> PX4 配置及环境对齐完成后，将建立独立的新 V3 报告目录，禁止与旧结果混合
> 汇总。

## Phase V1 / baseline

| 目录 | 内容 | Git 状态 |
| --- | --- | --- |
| `v1_baseline_s1_20260726/` | V1 基线摘要 | 跟踪摘要 |
| `high_entropy_baseline_3x500_20260726_191516/` | 高熵基线 | 跟踪 README；日志/模型仅本地 |
| `ablation_seed2_500ep/` | 输入消融 | 跟踪聚合 JSON |

## Phase V2 / degradation

| 目录 | 内容 |
| --- | --- |
| `phase_v2_real_20260726/` | 真实 GS 初始评估 |
| `phase_v2_smoke_5x5x10_20260727_082037/` | 五轴 smoke gate |
| `phase_v2_formal_5x5x50_20260727_082212/` | 五轴正式评估 |
| `phase_v2_structural_smoke_4x5x10_20260727_083219/` | 结构退化 smoke gate |
| `phase_v2_structural_formal_4x5x50_20260727_083353/` | 结构退化正式评估 |
| `phase_v2_depth_scale_confirm_5x200_20260727_100845/` | 深度尺度确认 |

## Phase V3 / robustness

| 主题 | 目录 |
| --- | --- |
| 尺度基线 | `v3_scale_eval_3x5x200_20260728_065821/` |
| 加权尺度 | `v3_scale_weighted_eval_3x5x200_20260728_084229/` |
| curriculum | `v3_scale_curriculum_gate_seed2_5x100_20260728_103900/`, `v3_scale_curriculum_eval_3x5x200_20260728_124659/` |
| checkpoint | `v3_checkpoint_selection_comparison_20260729/`, `v3_ckptfix_*` |
| clean recovery | `v3_clean_recovery_gate_seed0_5x100_20260729_053619/` |
| 几何对齐 | `v3_aligned_*`, `v3_gscloud_aligned_*` |
| 测地绕障 | `v3_geodesic_*`, `v3_hybrid_geodesic_*` |
| waypoint | `v3_waypoint_obs_gate_seed2_2x200_20260730/`, `v3_waypoint_oracle_200_20260730/` |
| 输入依赖诊断 | `v3b_input_gate_seed2_7x100_20260729/`, `v3b_v3c_stage_20260729/` |

## 本地目录

Git 未跟踪的训练目录通常包含模型、日志或可再生成的完整输出。它们可以继续位于
`reports/`，但不属于可合并内容。若需要跨机器保存，应上传到外部制品存储，并在对应
README 中记录制品地址、代码提交、随机种子和校验值。

## 新实验最低要求

进入 Git 的新实验目录至少应包含：

- 一份 README，说明目的、代码提交、配置、随机种子和结论；
- 一份聚合 CSV 或 JSON；
- 仅在被报告引用时加入关键图；
- 不包含 `.log`、`.pid`、`.pth` 或 `models/`。
