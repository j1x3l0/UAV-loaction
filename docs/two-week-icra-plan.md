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

## 当前执行指针

1. **D3-lite 强化**：正在自动跑（seeds 1-4 + 5-seed 评估 watcher），~2h 后出结果。
2. **下一步 GPU**（D3-lite 完成后）：退化轴补齐（对齐环境 5 轴）。
3. **并行（不占 GPU）**：论文骨架、审计 framing、门控跟踪。
