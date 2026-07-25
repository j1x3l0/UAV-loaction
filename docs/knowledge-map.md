# 知识地图 — UAV Visual RL 3D Path Planning

> 回答"系统长什么样？"。由 `@knowledge-map-maintainer` 维护。
> 任何代码改动后都应检查本文件是否需要更新。
> 更新原则：宁可新增一行，也不要让一段过期——过期比缺失更危险。

---

## 1. 职责矩阵

| 模块 | 一句话职责 | 不负责 | 关键依赖 | 理解度 |
|------|-----------|--------|----------|--------|
| `envs/visual_drone_env.py` | 3DGS渲染+深度图观测的RL环境 | 网络训练、3DGS训练、真机部署 | MockGSRenderer, degradation_utils | 85% |
| `envs/degradation_utils.py` | 5退化轴的参数化控制 | 环境物理、奖励函数 | numpy, PIL | 80% |
| `core/visual_ppo_agent.py` | CNN编码器+PPO视觉导航 | 环境交互、训练循环、日志 | PyTorch | 85% |
| `scripts/train_visual.py` | 视觉PPO训练管线 | 模型架构、环境实现、评估 | VisualDroneEnv, VisualPPO | 80% |
| `scripts/eval_degradation.py` | 批量退化评估 | 训练、退化工具实现、绘图 | VisualDroneEnv, degradation_utils | 80% |
| `scripts/plot_degradation_curves.py` | 衰减曲线可视化 | 数据采集、退化工具实现 | matplotlib | 75% |
| `envs/drone_env.py` | v1 向量环境 (纯向量观测) | 视觉渲染、退化评估 | gymnasium | 90% |
| `core/ppo_agent.py` | v1 向量PPO (MLP only) | 视觉编码、3DGS集成 | PyTorch | 90% |

## 2. 依赖图

```
┌──────────────────────┐
│  scripts/train_visual  │  ← Application层
└──────┬───────┬───────┘
       │       │
       ▼       ▼
┌──────────┐ ┌──────────────┐
│ VisualPPO │ │ VisualDroneEnv│  ← Foundation + Extension层
└─────┬────┘ └──┬───────┬───┘
      │          │       │
      ▼          ▼       ▼
┌──────────┐ ┌────────┐ ┌──────────────────┐
│VisualEnc │ │MockGS  │ │degradation_utils │  ← 内部组件
│(CNN)     │ │Renderer│ │(5退化轴)          │
└──────────┘ └────────┘ └──────────────────┘
```

关键约束:
- `eval_degradation.py` 不能直接调用 `train_visual.py`
- `degradation_utils` 不依赖 `VisualDroneEnv` (可独立测试)
- v1 和 v2 不互相依赖 (独立运行，仅共享物理参数)

## 3. 雾图（Fog Map）

| 区域 | 现状 | 风险等级 | 待澄清项 |
|------|------|----------|----------|
| 真实3DGS集成 | MockRenderer占位，未接入gsplat/Nerfstudio | 高 | gsplat API兼容性？渲染延迟是否可接受？ |
| Lighting退化轴 | Mock中RGB全为零，光照偏移无实际效果 | 中 | 接入真实3DGS后需重新标定EV水平值 |
| Viewpoint退化轴 | 简化深度不确定性模拟，非真实视角限制 | 中 | 真实视角限制需修改相机位姿采样逻辑 |
| 多环境并行训练 | 当前实现为串行for循环 | 中 | 真3DGS渲染需GPU并行 → 需重构为vec_env |
| 模型泛化到真实场景 | 仅在Mock障碍物上测试 | 高 | 需要L0/L1数据集验证 |

## 4. 认知快照（截至 2026-07-25）

- 团队最熟悉的模块: v1 PPO管线、v2 VisualPPO架构、degradation_utils
- 团队最不熟的模块: 真实3DGS渲染管线、gsplat API
- 最近一次大改动: 2026-07-25 — 创建degradation_utils.py，补全5退化轴集成
- 当前最大的认知债务来源: MockRenderer与真实3DGS之间的gap — 所有退化轴水平值可能需要重新标定

## 5. 关键概念速查表

| 概念 | 一句话定义 | 出现在哪里 | 易混淆项 |
|------|-----------|-----------|----------|
| PPO | Proximal Policy Optimization, 无模型RL算法 | core/visual_ppo_agent.py | SHAC (可微分RL, GRaD-Nav使用) |
| GAE | Generalized Advantage Estimation, 优势函数估计 | core/visual_ppo_agent.py:compute_gae | MC Returns |
| 3DGS | 3D Gaussian Splatting, 新视角合成方法 | MockGSRenderer (Phase 0) | NeRF (神经辐射场) |
| 退化轴 | 渲染质量下降的独立维度 | degradation_utils.py:DEGRADATION_AXES | 噪声 (退化轴之一，不是全部) |
| 知识包 | 代码改动的结构化理解交付物 | CLAUDE.md 定义 | Diff (只交付变更) |
| v1 向量基线 | 14D完美向量观测 + MLP PPO | envs/drone_env.py | v2 视觉 (64×64深度图 + CNN) |

## 6. 更新日志

| 日期 | 改动 | 由谁 |
|------|------|------|
| 2026-07-25 | 初始化知识地图 — 基于v2 Phase 0完成状态 | @knowledge-map-maintainer |

---

> 本文件是"活文档"。如果它和代码不一致，且没有更新日志说明——那是认知债务，不是版本问题。
