# Experiment reports index

`reports/` 保存支撑阶段结论的实验摘要。训练日志、模型、PID 和未筛选的中间产物
只在本地保留，不应提交到 Git。

> **有效性说明（2026-08-01）：** 当前目录中的 V1/V2/V3 结果均产生于最终
> 仿真环境对齐之前，统一视为 `legacy-unaligned` 历史诊断。它们可以用于验证
> 代码、分析失败模式和设计新实验，但不得作为正式 V3 验收或论文主结果。
> PX4 配置及环境对齐完成后，将建立独立的新 V3 报告目录，禁止与旧结果混合
> 汇总。

## Legacy-unaligned（已移入 past/）

> **2026-08-04**：V1/V2/V3 的 legacy-unaligned 报告（40 个目录）已移入
> `past/reports_legacy_unaligned/`。它们是 PX4 对齐前的历史诊断，不再作为当前
> 报告索引。`eval_results/` 的 mock 退化结果移入 `past/eval_results_mock/`。
> 详情见 `past/` 与 PROGRESS.md 的 `legacy-unaligned` 标记。

## Phase PX4 / 集成门禁

| 主题 | 目录 |
| --- | --- |
| PX4–3DGS 相机注册初检 | `px4_3dgs_camera_registration_20260801/` |
| 旧 checkpoint 策略接口冒烟 | `px4_policy_interface_smoke_20260801/` |
| 飞行体积碰撞净空 | `px4_flight_volume_clearance_20260802/` |
| 定点悬停观测回放 | `px4_hover_observation_test_20260802/` |
| 对齐 V3 尺度对照 + 配对统计 | `px4_v3_aligned_scale_comparison_20260803/` |
| 对齐 V3 输入消融 | `px4_v3_aligned_ablation_20260803/` |
| 视觉必要性（D3-lite 通过） | `px4_d3lite_visual_necessity_20260803/` |
| 集成测试套件（83 用例） | `integration_tests_20260802/` |
| 图表库（20 图，legacy 诊断） | `plot_gallery_20260802/` |

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
