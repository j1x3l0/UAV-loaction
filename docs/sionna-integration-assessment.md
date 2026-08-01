# Sionna RT 集成评估

日期：2026-07-29

## 决策

Sionna RT 可作为“通信感知无人机导航”扩展，但不替代当前的 3DGS 视觉渲染、
点云碰撞几何或无人机动力学。当前主线仍是先证明场景对齐后策略确实依赖视觉；
该门控通过前不安装、不启动 Sionna 实验。

## 推荐系统边界

使用同一坐标系下的三种场景表示：

```text
scene mesh / calibrated coordinates
├── 3DGS + gsplat: optical depth/RGB observation
├── dense point cloud + KD-tree: collision and free-space geometry
└── Mitsuba XML + radio materials: Sionna RT propagation
```

Sionna 的适用输出包括 path gain、RSS、SINR、outage、CIR/CFR 和 Doppler。
它不负责飞行动力学、光学深度、碰撞检测或 Swift 式飞控。

## 集成方式

不在每个 RL step 中实时运行无线射线追踪。采用离线 Radio Map：

1. 将 gate_mid 或 Replica mesh 转为 Mitsuba XML。
2. 标定与 3DGS/碰撞点云一致的坐标。
3. 配置频率、基站、天线方向和墙体无线材料。
4. 为多个高度平面预计算 RSS/SINR/outage。
5. 保存为 NPZ/HDF5 三维网格。
6. PPO 环境按位置三线性插值，加入通信观测、奖励或约束。

候选目标：

```text
reward = navigation_reward + lambda_radio * normalized_sinr
constraint: outage_rate <= 5%
```

## 环境隔离

当前视觉训练环境固定为 Python 3.10、PyTorch 2.1.1+cu118、gsplat 1.5.3。
Sionna 2.x / Sionna RT 依赖较新的 Python/PyTorch 以及 Mitsuba/Dr.Jit，不应直接
安装进现有 `myconda`。未来使用两个独立环境：

```text
uav-gsplat-env  -> visual training/evaluation
uav-sionna-env  -> offline radio-map generation
```

两者只通过缓存的 Radio Map 文件交换数据。

## 启用门槛

只有同时满足以下条件才进入 Sionna：

1. 新场景闭环 baseline 能稳定学习；
2. 正常深度显著优于 const-depth；
3. gate_mid 的视觉、碰撞和 mesh 坐标完成一致性验证；
4. V3b 核心消融至少完成一个正式三种子实验。

若正常深度与 const-depth 仍接近，优先修复视觉任务，不用无线通信任务掩盖问题。

## 预计增量

| 工作 | 预计时间 |
|------|----------|
| 独立环境与官方 smoke | 0.5 天 |
| mesh/XML/材料与坐标标定 | 0.5–1 天 |
| 多高度 Radio Map | 0.5 天 |
| PPO 查询接口与奖励 | 0.5 天 |
| 三种子正式对比 | 1–2 天 |

总增量约 3–4 天。
