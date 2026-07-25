# GRaD-Nav 代码审查报告

- 审查时间: 2026-07-21 23:35-23:55
- 代码来源: `grad_nav-main/` (GitHub开源仓库)
- 审查版本: 仓库未打 tag，以 2026-07-21 拉取的 `main` 分支为准；建议后续锁定到某个 commit hash
- 审查深度: 全量关键文件精读（16个核心Python文件，下表列9个代表性文件；其余7个为配置/工具类文件）

---

## 一、审查范围

| 层级 | 文件 | 行数 | 审查重点 |
|------|------|------|---------|
| 算法 | `algorithms/gradnav.py` | 760 | SHAC核心训练循环、actor loss计算（梯度穿过仿真器） |
| 算法 | `algorithms/ppo.py` | 782 | PPO基线、GAE、collect_trajectories（与gradnav共享环境） |
| 环境 | `envs/drone_ppo.py` | 903 | 环境核心：GS渲染、dynamics、reward、obs构建 |
| 环境基类 | `envs/dflex_env.py` | 112 | 环境抽象基类（buffer管理、reset） |
| 动力学 | `envs/assets/quadrotor_dynamics_advanced.py` | 184 | 四元数+PD控制器+气动阻力+噪声 |
| 模型 | `models/actor.py` | 114 | Actor网络（Deterministic/Stochastic MLP） |
| 模型 | `models/vae.py` | 184 | VAE（观测历史→隐空间） |
| 模型 | `models/squeeze_net.py` | 25 | 视觉感知网络（冻结SqueezeNet+FC） |
| 渲染 | `utils/gs_local.py` | 250 | 3DGS渲染包装器（Nerfstudio集成） |

---

## 二、关键纠正（vs 之前的速览分析）

### 纠正1：视觉编码器是冻结的，不是端到端训练的 CNN

```
❌ 之前理解: CNN(512→256→128) 端到端训练
✅ 实际代码: SqueezeNet 1.0 (pretrained ImageNet, requires_grad=False)
             → AdaptiveAvgPool(1×1) → FC(512→16)
             整个视觉编码器不参与RL梯度更新
```

**影响**: 策略学的是"如何根据固定的视觉特征做决策"，不是"如何看+决策"。在退化GS下，这个冻结编码器的特征会退化，而网络无法适应——这是一个潜在弱点。

### 纠正2：观测不是图像，是57D向量

```
❌ 之前理解: RGB 64×64 → CNN → 动作
✅ 实际代码: RGB 640×360 → SqueezeNet → 16D视觉特征
             → 拼入57D向量 [z, vel, quat, action, prev_action, visual_info, latent]
             → Actor MLP → 4D动作
```

**影响**: GRaD-Nav的策略**不直接看图像**。16D视觉特征是一个极窄的信息瓶颈。我们计划端到端训练CNN从深度图学习——信息流更丰富。

### 纠正3：VAE是动力学上下文编码器，不是场景编码器

```
❌ 之前理解: CENet = 场景编码器（编码不同场景的布局）
✅ 实际代码: VAE输入 = 过去5步的57D观测历史
            VAE输出 = 24D latent（预测下一帧观测）
            功能: 隐式系统辨识——推测当前动力学参数（质量、惯性）
```

**影响**: VAE只在域随机化有意义（需要推测变化的物理参数）。我们不计划做动力学域随机化，所以不需要这个组件。

### 纠正4：GRaD-Nav代码库自带PPO基线

```
❌ 之前理解: GRaD-Nav只有DDRL
✅ 实际代码: algorithms/ppo.py 使用完全相同的DronePPOEnv
            唯一差异: no_grad=True（PPO）vs no_grad=False（GradNav）
            共享: 3DGS渲染器、动力学、reward函数
```

**影响**: 我们可以直接在GRaD-Nav代码上做PPO→DDRL的退化对比，不需要自己写环境。

### 纠正5：Actor MLP维度可配置

```
yaml中配置: actor_mlp.units = [512, 256, 128]（可调）
            activation = 'elu'
            每层后有LayerNorm
```

---

## 三、代码质量评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 可读性 | 6/10 | 大量硬编码、注释不足、变量名有时混淆（obs_buf vs privilege_obs_buf vs vae_obs_buf） |
| 鲁棒性 | 5/10 | 大量`nan_to_num`调用说明数值稳定性是已知问题；`autograd.set_detect_anomaly(True)`在生产训练中是性能杀手 |
| 模块化 | 6/10 | 算法/环境/模型分离清晰，但环境内部耦合度高（obs构建、reward、dynamics在一个文件） |
| 可复现性 | 7/10 | 依赖Nerfstudio生态、需要预下载GS数据和点云数据；路径处理有跨平台问题 |
| 文档 | 4/10 | README只有训练命令；无API文档；大量magic numbers无注释 |

### 具体问题

1. **NaN处理泛滥**: 代码中大量 `torch.nan_to_num(..., nan=0.0, posinf=1e3, neginf=-1e3)` ——说明训练过程中NaN/Inf频繁出现，但没有追根因。**建议**：在关键位置（actor loss 计算前、reward 计算后）加入 `torch.isnan().any()` 断言 + 日志，出现NaN时保存完整tensor快照以便定位数值不稳定源
2. **3DGS渲染逐batch循环**: `gs_local.py:124` 的 `for i in range(batch_size)` ——非批量化渲染，可能是速度瓶颈
3. **每个reset重建QuadrotorSimulator**: `drone_ppo.py:480` ——域随机化时的必要开销，但不做域随机化时可以缓存
4. **坐标变换复杂**: 无人机坐标系→ROS坐标系→Nerfstudio坐标系的多次变换，容易出错
5. **无测试**: 整个仓库没有单元测试

---

## 四、对 v2 项目的可复用评估

| 组件 | 可复用量 | 难度 | 说明 |
|------|---------|------|------|
| `utils/gs_local.py` | 🟢 高 | 低 | 3DGS渲染包装器——可直接用Nerfstudio渲染我们的场景 |
| `envs/drone_ppo.py` | 🟡 中 | 中 | 环境框架可参考，但需要大幅简化（去掉四元数、PD控制器） |
| `algorithms/ppo.py` | 🟢 高 | 低 | PPO+GAE实现可直接对比，验证我们的PPO实现 |
| `utils/point_cloud_util.py` | 🟡 中 | 低 | 点云碰撞检测——如果我们用点云做碰撞检测 |
| `models/squeeze_net.py` | 🔴 低 | 低 | 冻结SqueezeNet不适用于我们的端到端训练方案 |
| `models/vae.py` | 🔴 低 | 中 | VAE只在域随机化有意义，我们不需要 |
| `quadrotor_dynamics_advanced.py` | 🔴 低 | 高 | 全四元数动力学过于复杂，与我们的简化需求不匹配 |

---

## 五、更新后的差异化优势

基于代码审查，我们的v2方案在以下方面**明确优于**GRaD-Nav：

1. **端到端可训练视觉编码器** — GRaD-Nav冻结SqueezeNet，视觉特征无法适应退化GS；我们端到端训练CNN，可能更鲁棒
2. **无需预规划路径** — GRaD-Nav依赖参考轨迹+waypoints（通过全局路径规划器）；我们只给目标方向，更通用
3. **不需要VAE上下文编码器** — 简化架构，减少一个可能的故障源
4. **3D推力vs体角速度** — 我们的动作空间更简单，更适合导航鲁棒性分析

---

*审查完成时间：2026-07-21*
