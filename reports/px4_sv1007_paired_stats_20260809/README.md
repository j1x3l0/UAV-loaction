# sv_1007 输入消融配对检验（固定 episode，5 seeds）

日期：2026-08-09。针对用户评估指出的缺口（"尚无配对显著性检验"），用**固定
测试 seed**（20260809）重跑输入消融，保存逐 episode 数据，做配对 McNemar +
bootstrap CI + Cohen's g 效应量。

## 方法

- **模型**：sv_1007 clean 5 seeds（`v3_sv1007/seed*_3000_final.pth`）
- **评估**：`eval_v3_ablation.py --ablation all`，50ep/消融，**固定 seed=20260809**
  （所有消融/模型用相同 episode 序列 → 天然配对）
- **工具**：`scripts/paired_stats.py`（McNemar + 配对 bootstrap 95% CI + Cohen's g）
- **判定**：McNemar p<0.05 **且** bootstrap CI 不含 0 → 显著
- **GPU0 only**
- 原始数据：`/root/px4-deploy/paired_ablation/`（含 episodes_detail）

## 各 seed SR（%）+ 配对显著性

| seed | baseline | const_depth | no_velocity | no_target_dir |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 46 | 38 | 4 | 2 |
| 1 | 62 | 38 | 8 | 0 |
| 2 | 62 | 36 | 12 | 2 |
| 3 | 62 | 44 | 8 | 0 |
| 4 | 56 | 44 | 6 | 2 |
| **均值** | **57.6** | **40.0** | **7.6** | **1.2** |

## 配对检验结果（vs baseline）

### const_depth（去深度）——深度必要性

| seed | diff(pp) | McNemar p | bootstrap CI | Cohen's g | 显著 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | +8.0 | 0.289 | [-2, +20] | 0.50 | ❌ |
| 1 | +24.0 | 0.0015 | [+12, +36] | 1.00 | ✅ |
| 2 | +26.0 | 0.0009 | [+14, +38] | 1.00 | ✅ |
| 3 | +18.0 | 0.0159 | [+6, +30] | 0.82 | ✅ |
| 4 | +12.0 | 0.0412 | [+4, +22] | 1.00 | ✅ |

**4/5 seeds 显著**（seed0 不显著：p=0.29）。与 D3-lite 的 4/5 结论一致。

### no_velocity（去速度）——速度必要性

| seed | diff(pp) | McNemar p | bootstrap CI | Cohen's g | 显著 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | +42 | <0.0001 | [+28, +56] | 1.00 | ✅ |
| 1 | +54 | <0.0001 | [+40, +68] | 1.00 | ✅ |
| 2 | +50 | <0.0001 | [+36, +64] | 1.00 | ✅ |
| 3 | +54 | <0.0001 | [+38, +68] | 0.93 | ✅ |
| 4 | +50 | <0.0001 | [+34, +64] | 0.93 | ✅ |

**5/5 seeds 显著**（速度高度必要）。

### no_target_dir（去目标）——目标方向必要性

| seed | diff(pp) | McNemar p | bootstrap CI | Cohen's g | 显著 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | +44 | <0.0001 | [+28, +58] | 0.92 | ✅ |
| 1 | +62 | <0.0001 | [+48, +76] | 1.00 | ✅ |
| 2 | +60 | <0.0001 | [+46, +74] | 1.00 | ✅ |
| 3 | +62 | <0.0001 | [+48, +76] | 1.00 | ✅ |
| 4 | +54 | <0.0001 | [+38, +68] | 0.93 | ✅ |

**5/5 seeds 显著**（目标方向决定性输入）。

## 结论

1. **深度必要性配对显著（4/5 seeds）**：const_depth 在 4/5 seed 上 McNemar p<0.05
   且 CI 不含 0，效应量 Cohen's g 0.5-1.0。与 D3-lite 的 4/5 一致，支持
   "深度对避障必要"。
2. **速度/目标方向决定性（5/5 seeds）**：no_velocity/no_target_dir 全部显著，
   Cohen's g≈1.0（最大效应）。
3. **seed0 的深度必要性不显著**：seed0 的 const_depth 差异仅 8pp（p=0.29）——
   这是诚实报告的方差（种子差异），不掩盖整体 4/5 结论。

---

# 补充：PPO vs BC 与 clean vs curriculum 配对检验（2026-08-10）

## PPO vs BC（3 seeds，100ep，固定 seed）

| seed | PPO SR | BC SR | diff | McNemar p | CI | 显著 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 47% | 36% | +11pp | 0.063 | [+1, +21] | ❌（边缘）|
| 1 | 60% | 38% | +22pp | <0.0001 | [+13, +31] | ✅ |
| 2 | 63% | 29% | +34pp | <0.0001 | [+25, +44] | ✅ |

**RL 优于 BC 是强趋势，但仅 2/3 seeds 显著**（seed0 边缘 p=0.063）。
诚实结论：BC 配对检验支持 RL > BC，但未达到全 seed 显著。

## clean vs curriculum baseline（5 seeds，100ep，固定 seed）

| seed | clean | curriculum | diff | McNemar p | CI | 显著 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 47% | 44% | +3pp | 0.689 | [-7, +13] | ❌ |
| 1 | 60% | 36% | +24pp | <0.0001 | [+14, +35] | ✅ |
| 2 | 63% | 42% | +21pp | <0.0001 | [+12, +30] | ✅ |
| 3 | 60% | 35% | +25pp | <0.0001 | [+15, +35] | ✅ |
| 4 | 50% | 34% | +16pp | 0.0014 | [+8, +25] | ✅ |

**4/5 seeds 显著：clean 正常环境性能显著优于 curriculum**（McNemar p<0.01，
CI 不含 0，Cohen's g 0.7-0.8）。证实 curriculum 的"鲁棒优势"以牺牲正常性能为代价
——统计上成立。这印证了用户评估："curriculum 不能宣称总体更鲁棒"。

## 汇总（配对检验总览）

| 对比 | 显著 seed 数 | 结论 |
|------|:---:|------|
| baseline vs const_depth（深度）| 4/5 | 深度必要 |
| baseline vs no_velocity（速度）| 5/5 | 速度决定性 |
| baseline vs no_target_dir（目标）| 5/5 | 目标决定性 |
| PPO vs BC | 2/3 | RL 强趋势，未全显著 |
| clean vs curriculum（正常环境）| 4/5 | clean 显著优于 curriculum |

## 数据

- 输入消融：`/root/px4-deploy/paired_ablation/`（5 seeds）
- BC：`/root/px4-deploy/bc_paired3/`（3 seeds，100ep）
- curriculum：`/root/px4-deploy/cur_baseline_paired/`（5 seeds，100ep）
- 本地副本：`local_results/eval/`
4. **满足用户第 1 条要求**：固定 episode + 配对检验 + 95% CI + 效应量。

## 文件

- 工具：`scripts/paired_stats.py`
- 原始数据（服务器）：`/root/px4-deploy/paired_ablation/`（含 episodes_detail）
- 本地副本：`local_results/eval/paired_ablation/`
