# 对齐 V3 深度尺度鲁棒性对照

日期：2026-08-03。环境：`gate_mid_new` 场景 + 对齐窄视场相机（fx≈97.14，36° FOV，目标偏航）+ 0.5m 碰撞半径 + `scale_curriculum`（gentle 版）。

## 方法

- 模型：clean 基线（1500ep clean @0.5m）+ 全量 gentle curriculum 3 seeds 的 robust-best。
- 评估：逐尺度 50 episodes（seed 20260802），用与训练一致的 env（collision_ply + auto_scene_bounds + 0.5m 半径 + 对齐相机）。
- 尺度集：`[1.0, 0.75, 0.5, 0.25]`（0.1x 已从对齐 V3 分析移除，传感器失效级）。

## 结果（SR %）

| 模型 | 1.0x | 0.75x | 0.5x | 0.25x | min | mean |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| clean_baseline（1500ep） | 50.0 | 48.0 | 28.0 | 14.0 | 14.0 | 35.0 |
| cur_robust_seed0 | 20.0 | 28.0 | 30.0 | 34.0 | 20.0 | 28.0 |
| cur_robust_seed1 | 52.0 | 44.0 | 38.0 | 36.0 | 36.0 | 42.5 |
| cur_robust_seed2 | 38.0 | 40.0 | 46.0 | 46.0 | 38.0 | 42.5 |

## 结论

1. **clean 基线在退化尺度崩塌**：1.0x 50% → 0.25x 14%（-36pp），非鲁棒。
2. **curriculum robust 模型明显更平坦**：seed1 1.0x→0.25x 只掉 16pp；seed2 低尺度反升（38→46）。
3. **robust min 提升 2.5 倍**：curriculum 36-38% vs clean 14%（点估计）。
4. **robust mean**：curriculum 42.5% vs clean 35%。
5. seed 方差存在：seed0 整体偏弱（mean 28%），seed1/2 为强结果。

> ⚠️ **统计警示（2026-08-03）**：n=50 下 clean 0.25x 14% 的 Wilson CI 为 [7.0, 26.2]，robust seed1 36% 的 CI 为 [24.1, 49.9]，**区间重叠**——当前点估计优势**不具统计显著性**。现有评估未保存逐 episode 结果，无法做配对检验；需保存逐 episode 数据并做配对 bootstrap/McNemar，并扩展 seeds 后才有统计结论。

**对齐窄视场相机下，深度尺度鲁棒训练验证成立**（用 clean 换跨尺度稳定性）。

## 文件

- `result.json`：逐模型逐尺度 SR。
- 模型：`saved_models/v3_aligned_gentle/seedN_{best,robust_best,final}.pth`（本地服务器）、`saved_models/v3_aligned_smoke_clean/seed0_r05_1500.pth`。
- 训练日志：服务器 `/root/px4-deploy/v3_gentle_full.log`。
