# 两周 ICRA 2027 冲刺计划（2026-08-03 修订，按实际速度）

> 前提：实测 7/22→8/3 十二天完成 Phase 0-2 + 重大 pivot，1500ep 仅 ~1.2h/seed。
> 修正：原 21 天/3 GPU 预算过度保守；两周足以补到可投水平。

## 已确认现状（2026-08-03）

| 项目 | 状态 |
|------|------|
| PX4 对齐四门禁 + 相机统一 + 对齐 V3（gentle curriculum 3→5 seeds） | ✅ |
| V3 三方对照 + 配对统计（0.25x 4/5 seeds 显著，n=50） | ✅ |
| 视觉必要性 D1（碰撞半径硬化） | ❌ 失败 |
| 视觉必要性 D2（去目标向量） | ❌ 失败（任务不可解） |
| **视觉必要性 D3-lite（合成稠密走廊）** | ✅ **通过（baseline 54% / const_depth 0%）** |
| D3-lite 强化（5 seeds + 配对） | 🔄 运行中 |
| `formal_v3_ready` / `publication_ready` | `experiment_complete=true` / `publication_ready=false` |

## 门控决策（已确定）

- **8/10 门控的「视觉必要性成立」分支已达成**（D3-lite 通过）→ **ICRA 2027 主攻**。
- 剩余真正风险：D3-lite 5-seed 显著性、clean SR 攻关、退化轴补齐。

## 两周计划（D1-D14，GPU 估算按 ~1.2h/1500ep）

| 天 | 任务 | 门控 | GPU |
|:--:|------|------|:---:|
| D1 | D3-lite seeds 1-4 收尾 + 5-seed 必要性配对 | G: ≥4/5 seed baseline−const_depth 显著 | 1 |
| D2-3 | 退化轴补齐（稀疏化/分辨率/噪声/视角在对齐环境重做，纯评估） | 5 轴曲线完整 | 1.5 |
| D4 | clean SR 攻关（长训/reward 微调，夜间 GPU） | ≥70% | 2 晚 |
| D5-6 | V3c 跨场景（gate_left/right 对齐配置 + 训练） | 每场景鲁棒对照 | 2-3 |
| D7-8 | BC 对照基线（waypoint oracle 采数据训 BC） | BC SR 报告 | 1-2 |
| D9-10 | 消融补齐（RGB 替代/浅 CNN/无特权 Critic） | 3 组结果 | 1.5 |
| D11-12 | SHAC/GRaD-Nav 对照（timebox；失败降级理论分析） | 对照或 Discussion | ≤2 |
| D13-14 | 数据汇总 + 缺口检查 + 论文初稿（并行） | 完整实验矩阵 | — |

**不做（诚实边界）**：真机飞行、ScanNet/HM3D 大规模泛化、PX4 HITL（留 camera-ready/扩展版）。

## 投稿策略（修正）

```
D1 门控（D3-lite 5-seed 显著）→ ICRA 2027 主攻（审计 framing）
clean SR 攻关失败 → 仍 ICRA，绝对数字如实报告（对比 + CI 主张）
SHAC 对照失败 → Discussion 理论分析（预案 R4）
```

## 当前执行指针（2026-08-06 更新）

> **2026-08-04 主场景切换**：主场景定为 **sv_1007**（完整 gate，22×25×9m）。
> 障碍簇计数已更正：原始点云在 0.5m 体素有 16 个连通分量，但其中 13 个是低保真噪声斑点；
> `clean_sv1007.py` 清洗后为 **ground + 5 障碍簇**（详见 `experiments/visual_necessity/README.md`）。
> 对齐方式采用**合成 identity**（fx≈97.14 相机 + 目标偏航 + 无场景旋转，`experiments/visual_necessity/sv1007_alignment.json`）——相机对齐是视觉感知关键，已达成。gate_mid_new 结果保留作对比/V3c。

1. **D4 clean SR 攻关（sv_1007 新主场景）**：✅ 3000ep clean 5 seeds 完成，best_SR **43/60/55/64/44，均值 53.2%**（seed 方差大，44–64%）。
2. **curriculum 3000ep 公平对比（sv_1007）**：✅ 5 seeds × 3000ep 完成，best_SR **41/40/51/34/36，均值 40.4%**（训练用 scale_curriculum，模型 `v3_sv1007_curriculum/`）。
3. **退化轴评估（sv_1007，clean vs curriculum 5 seeds）**：✅ 7 轴 × 5 档 × 50ep 完成，见 `reports/px4_sv1007_degradation_20260805/`。
   **⚠️ 新结论（与旧场景相反）**：sv_1007 上 **clean 全面优于 curriculum**——无退化档 clean 52.4% vs curriculum 31.6%（**+20.8pp**），绝大多数轴每档 clean 领先 10–21pp；仅视角 45°（+1.6pp）、相机遮挡 50%（−3.2pp）curriculum 接近或略好。对比旧场景 gate_mid_new（curriculum 视角轴显著更鲁棒，clean 45° 掉 18%），**curriculum 鲁棒优势不泛化到 sv_1007**。待核对 curriculum 训练 robust eval 曲线确认是"优势不泛化"还是"场景失效"。
4. **资源约束**：后续训练**仅用 GPU0**（GPU1 保留给其他任务）。
5. **视觉必要性**：D3-lite 5-seed 4/5 显著（✅）；D1/D2 失败；sv_712 复杂度测试 / 相机统一 / 重构叙事**待决策**。

## 全项目优先级评估（含高保真加强项）

| 优先级 | 项目 | 必要性 | 状态 |
|:---:|------|--------|:---:|
| **P0（必须）** | V3c 跨场景泛化（left/right + sv_712）| 单场景无法过审 | 待做 |
| **P0（必须）** | 视觉必要性叙事定稿（重构 or 解决矛盾）| 致命矛盾不解决会打回 | 待决策 |
| **P0（必须）** | SHAC/DDRL 或域随机化基线 | 修复主张需硬对照 | 待做（timebox）|
| **P1（重要）** | BC 对照基线 | 补全 RL vs 行为克隆 | 待做 |
| **P1（重要）** | 消融补齐（RGB/浅CNN/无特权 Critic）| 方法完备性 | 待做 |
| **P2（可选加强）** | **高保真场景**（重采集/重建 1 个）| 退化基线更干净 + 绝对性能 + 「保真度→性能」对照 | 可选 |
| **P2（可选加强）** | 更多种子 / reward 调优 | 统计/性能加强 | 视时间 |

**高保真场景评估**：有意义的加强项（3 点：退化分析更干净、绝对性能提升、补 EmbodiedSplat 的「保真度→性能」对照），但**非核心主张的必要条件**（退化鲁棒结论在任何保真度成立）。成本中等（重采集 + GPU 重建）。**优先级 P2**：在 P0（V3c + 必要性 + 基线）完成后有余力再做。

## 投稿策略（修正）

```
D1 门控（D3-lite 5-seed 显著）→ ICRA 2027 主攻（审计 framing）
clean SR 攻关失败 → 仍 ICRA，绝对数字如实报告（对比 + CI 主张）
SHAC 对照失败 → Discussion 理论分析（预案 R4）
V3c/必要性不过 → 降级 IROS/workshop（不硬投 9 月）
```
