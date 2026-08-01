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
| `envs/gs_renderer.py` | gsplat/CPU 3DGS渲染器, 显式内参与c2w位姿 | 坐标对齐、PX4通信、3DGS训练 | gsplat, numpy | 80% |
| `integrations/px4_offboard.py` | 策略动作→PX4 NED setpoint 纯转换层 | 通信、状态机、场景坐标 | numpy (无外部依赖) | 85% |
| `integrations/mavlink_offboard.py` | pymavlink Offboard 客户端 (连接/arm/起飞/落地) | 坐标转换、策略推理 | pymavlink | 75% |
| `integrations/px4_scene_alignment.py` | PX4 LOCAL_NED↔3DGS场景坐标对齐 (唯一入口) | 渲染、MAVLink通信 | numpy, config json | 80% |
| `integrations/rl_policy_offboard.py` | 限幅策略冒烟门禁 (合成深度, 不声明对齐导航) | 真实视觉闭环、正式V3结果 | VisualPPO, mavlink_offboard | 75% |
| `scripts/validate_camera_registration.py` | 训练相机位姿注册验证 (OpenGL→OpenCV轴判别) | 在线观测、飞控交互 | GSplatRenderer, PIL | 80% |
| `configs/px4_gate_mid_alignment.json` | gate_mid_new 锚点/内参/坐标约定 (formal_v3_ready=false) | 代码逻辑 | — (配置) | 85% |

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

PX4↔3DGS 集成链 (2026-08-02 新增):

```
PX4 SIH/SITL ──MAVLink──► integrations/mavlink_offboard.py ─► integrations/px4_offboard.py (动作转换)
    │                                                              ▲
    ▼                                                              │
integrations/px4_scene_alignment.py ──► envs/gs_renderer.py ──► VisualPPO / VisualDroneEnv
    (NED+FRD→OpenCV c2w)   (显式内参+位姿)          │
    ▲                                                  ▼
configs/px4_gate_mid_alignment.json               scripts/validate_camera_registration.py
    (锚点/内参/约定)                                  (训练相机注册验证)
```

关键约束:
- `eval_degradation.py` 不能直接调用 `train_visual.py`
- `degradation_utils` 不依赖 `VisualDroneEnv` (可独立测试)
- v1 和 v2 不互相依赖 (独立运行，仅共享物理参数)
- `px4_scene_alignment.py` 是 NED↔scene 坐标转换唯一入口，不得用 ENU↔NED 交换代替场景标定旋转
- `rl_policy_offboard.py` 只做接口验证 (`aligned_v3_result=false`)，不产出正式 V3 结果

## 3. 雾图（Fog Map）

| 区域 | 现状 | 风险等级 | 待澄清项 |
|------|------|----------|----------|
| 真实3DGS集成 | 已接入 gsplat GPU + CPU回退，265k Gaussian 64×64 预热后 ~1.8ms | 中 | 绝对亮度相关仍低 (0.03–0.23)，需30位姿统计确认 |
| PX4坐标对齐 | 5帧GPU轴向注册临时通过，锚点/内参已固化 | 高 | 30位姿注册+飞行净空门禁未过；`formal_v3_ready=false` |
| Lighting退化轴 | Mock中RGB全为零，光照偏移无实际效果 | 中 | 接入真实3DGS后需重新标定EV水平值 |
| Viewpoint退化轴 | 简化深度不确定性模拟，非真实视角限制 | 中 | 真实视角限制需修改相机位姿采样逻辑 |
| 多环境并行训练 | 当前实现为串行for循环 | 中 | 真3DGS渲染需GPU并行 → 需重构为vec_env |
| 模型泛化到真实场景 | 仅在Mock障碍物上测试 | 高 | 需要L0/L1数据集验证 |

## 4. 认知快照（截至 2026-08-02）

- 团队最熟悉的模块: v1 PPO管线、v2 VisualPPO架构、degradation_utils、gsplat 渲染
- 团队最不熟的模块: MAVLink/PX4 状态机 (prime→offboard→arm→hover→land→disarm) 与 failsafe 语义
- 最近一次大改动: 2026-08-02 — PX4↔3DGS 集成层 (坐标对齐/MAVLink客户端/策略冒烟/相机注册)
- 当前最大的认知债务来源: PX4 坐标对齐 (scene_from_ned 旋转/平移仅5帧验证) 与旧 V3 `legacy-unaligned` 结果的方法学边界

## 5. 关键概念速查表

| 概念 | 一句话定义 | 出现在哪里 | 易混淆项 |
|------|-----------|-----------|----------|
| PPO | Proximal Policy Optimization, 无模型RL算法 | core/visual_ppo_agent.py | SHAC (可微分RL, GRaD-Nav使用) |
| GAE | Generalized Advantage Estimation, 优势函数估计 | core/visual_ppo_agent.py:compute_gae | MC Returns |
| 3DGS | 3D Gaussian Splatting, 新视角合成方法 | MockGSRenderer (Phase 0) | NeRF (神经辐射场) |
| 退化轴 | 渲染质量下降的独立维度 | degradation_utils.py:DEGRADATION_AXES | 噪声 (退化轴之一，不是全部) |
| 知识包 | 代码改动的结构化理解交付物 | CLAUDE.md 定义 | Diff (只交付变更) |
| v1 向量基线 | 14D完美向量观测 + MLP PPO | envs/drone_env.py | v2 视觉 (64×64深度图 + CNN) |
| LOCAL_NED | PX4 本地北东下坐标系 (LOCAL_POSITION_NED) | integrations/px4_scene_alignment.py | ENU (东北上, 策略内部) |
| FRD | 机体前右下坐标系 (PX4姿态用) | configs/px4_gate_mid_alignment.json | OpenCV光学轴 (右/下/前) |
| OpenGL→OpenCV | 相机轴转换 diag(1,-1,-1), Nerfstudio→gsplat | scripts/validate_camera_registration.py | 不做转换直接喂gsplat |
| c2w | camera-to-world 位姿矩阵 (OpenCV光学轴) | envs/gs_renderer.py:render | w2c / pos+quat 路径 |

## 6. 更新日志

| 日期 | 改动 | 由谁 |
|------|------|------|
| 2026-07-25 | 初始化知识地图 — 基于v2 Phase 0完成状态 | @knowledge-map-maintainer |
| 2026-08-02 | 新增 PX4↔3DGS 集成层 7 个模块、PX4 集成依赖链、坐标对齐高认知债务区、LOCAL_NED/FRD/c2w 概念 | @knowledge-pack-generator + 主对话 |

---

> 本文件是"活文档"。如果它和代码不一致，且没有更新日志说明——那是认知债务，不是版本问题。
