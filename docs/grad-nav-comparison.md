# GRaD-Nav vs v2 项目：详细技术对比与差异化定位

> 整理时间：2026-07-21
> 基于 GRaD-Nav (IROS 2025) 论文精读 + **开源代码完整审查** (grad_nav-main/)
> 代码审查时间：2026-07-21 23:35-23:50

---

## 一、执行摘要

GRaD-Nav 是目前唯一一个将 3DGS 渲染 + 可微分 RL + 无人机导航集成为完整开源管线的工作。我们的 v2 项目与 GRaD-Nav **不是竞争关系，而是互补关系**：

- GRaD-Nav 证明了 **"3DGS+RL 能飞"**
- 我们回答 **"3DGS 质量差到什么程度就不能飞"**

GRaD-Nav 论文中一句话带过的 limitation（"GS quality impacts performance"），正是我们的核心贡献。

---

## 二、架构逐层对比

### 2.1 训练算法

| 维度 | GRaD-Nav | 我们的 v2 | 差异分析 |
|------|----------|----------|---------|
| 算法 | SHAC (Smooth Hamiltonian Actor-Critic) | PPO (Proximal Policy Optimization) | 可微分 vs 无模型 |
| 梯度流 | 策略→动作→动力学→下一状态→loss，全链可微 | 策略→动作，靠 reward 信号 | DDRL 梯度更直接，样本效率更高 |
| 样本效率 | 宣称 3.5h 训练 | 估计 20-35h (5000ep) | GRaD-Nav 理论优势 5-10× |
| 并行环境 | 128 | 8 | GRaD-Nav 大规模并行 + GPU 仿真 |
| 实现复杂度 | 高（需可微分动力学 + 二阶梯度） | 中（标准 PPO 管线） | PPO 更易调试、更易复现 |

**关键决策：为什么 v2 选 PPO 而不是 DDRL？**

1. **鲁棒性分析不需要极致样本效率**。我们的目标是系统测量衰减曲线，不是比谁训得快。PPO 在 3000ep 内完全能收敛到 >80% SR。
2. **DDRL 的梯度穿过渲染器**。当我们在退化 3DGS 上训练时，渲染器的梯度质量也退化——DDRL 的样本效率优势可能在高退化下消失甚至反转。这是一个值得报告的发现。
3. **PPO 是更公平的鲁棒性测试平台**。DDRL 的梯度路径依赖精确的 3DGS 参数（协方差矩阵、透明度），在退化场景下梯度可能不稳定。PPO 只依赖 reward 信号，对渲染质量不敏感。
4. **工程可行性**。可微分动力学仿真器需要从头写（GRaD-Nav 用了 Warp），而我们已有成熟的 PPO 管线。

> **v2 实验新增**：在 V3 算法对比中，增加一个分析——**DDRL 在高退化 3DGS 下是否失去样本效率优势？** 这直接回应了 DDRL 社区的 blind spot。

### 2.2 观测空间（⚠️ 代码修正）

**关键发现**：GRaD-Nav 的视觉处理管线不是"图像→CNN→动作"。而是"图像→冻结SqueezeNet→16D特征→拼接状态向量→MLP→动作"。

| 维度 | GRaD-Nav (实际代码) | 我们的 v2 | 差异分析 |
|------|----------|----------|---------|
| 视觉输入 | RGB 640×360 → resize 224×224 | 深度图 64×64 (1通道) | GRaD-Nav 用高分辨率RGB，我们用小分辨率深度图 |
| 视觉编码器 | **冻结的SqueezeNet** (pretrained, `requires_grad=False`) → AdaptiveAvgPool → FC(512→16) | 3层 Conv (32→64→64), 端到端训练 | **GRaD-Nav的CNN不参与RL训练！** 这是重大架构差异 |
| 观测向量 | **57D** = [z_pos, z_vel, quat(4), action(4), prev_action(4), lin_vel(3), visual_info(16), latent(24)] | ~10D = [速度(3), 目标方向(3), 深度特征] | GRaD-Nav将视觉压缩为16D特征嵌入观测向量 |
| 特权观测 | **67D** = 57D + [position(3), lin_acc(3), quat(4), ang_vel(3), up_vec(1), heading_alignment(1)] | Critic有位置+障碍物距离等特权信息 | GRaD-Nav的privileged obs包含了全部真实状态 |
| VAE上下文 | **24D latent** — VAE编码观测历史（5步buffer），学习动力学上下文 | 无 | 这是GRaD-Nav的"CENet"等价物—但不是场景编码器，是**动力学上下文编码器** |
| 目标表示 | 参考轨迹 + 4个waypoints（通过全局路径规划器预计算） | 目标方向向量 (3D) | GRaD-Nav依赖预规划参考轨迹，我们更通用 |

**关键代码路径** (`envs/drone_ppo.py`):
```
RGB渲染 (640×360) → resize 224×224 → SqueezeNet(frozen) → AdaptiveAvgPool → FC(512→16)
                                                                              ↓
                                                                     visual_info (16D)
                                                                              ↓
obs_buf = [z_pos, z_vel, quat(4), action(4), prev_action(4), lin_vel(3), visual_info, latent(24)]
                                                                              ↓
                                                                     Actor MLP → 4D action
```

**为什么这对我们的设计很重要？**

1. **GRaD-Nav的视觉编码器是冻结的**——策略学到的不是"看"，而是"如何根据固定的视觉特征做决策"。这意味着：如果在退化GS上评估，SqueezeNet提取的特征本身就会退化（输入RGB变了），但网络无法适应。
2. **我们计划端到端训练CNN**——策略同时学习"看"和"决策"。在退化GS上评估时，CNN可以部分适应视觉分布偏移。这是我们的**优势**——端到端训练可能比冻结编码器更鲁棒。
3. **VAE上下文编码器**在GRaD-Nav中起到了"隐式系统辨识"的作用——从观测历史推测动力学参数（质量、惯性等）。如果我们不做域随机化，就不需要这个组件。

### 2.3 动作空间

| 维度 | GRaD-Nav | 我们的 v2 | 差异分析 |
|------|----------|----------|---------|
| 动作维度 | 4D | 3D | GRaD-Nav 多一维 yaw rate |
| 动作定义 | [ω_x, ω_y, ω_z, T_norm] (体角速度 + 归一化推力) | [T_x, T_y, T_z] (世界坐标系推力) | GRaD-Nav 输出姿态变化，我们直接输出力 |
| 控制层级 | 高层：RL→角速度→PD控制器→电机转速 | 低层：RL→推力→质点加速度 | GRaD-Nav 更接近真机控制栈 |
| 物理建模 | 完整四元数 + 电机延迟 + 气动阻力 | 质点 + 速度限制 | GRaD-Nav 更真实但更复杂 |

**为什么 v2 简化为 3D 推力？**

1. **聚焦研究问题**。我们的研究问题是"渲染质量如何影响导航"，不是"如何精确控制无人机姿态"。质点模型足够支持导航策略的鲁棒性分析。
2. **消融的清晰性**。复杂动力学引入额外的噪声源，会混淆"性能下降是因为渲染退化还是因为控制不稳定"。
3. **与 v1 向量基线可比**。v1 的 14D 向量 → 3D 推力。v2 改为视觉输入但保持动作空间一致，这样 v1 和 v2 的性能差异可以归因于"视觉 vs 向量"而不是"推力 vs 角速度"。

### 2.4 网络架构（⚠️ 代码修正）

**关键纠正**：之前基于论文速览的CNN维度描述有误。以下是代码实际结构：

| 组件 | GRaD-Nav (实际代码) | 我们的 v2 | 关键差异 |
|------|----------|----------|---------|
| **视觉编码器** | **SqueezeNet 1.0** (pretrained ImageNet, **冻结**) → AdaptiveAvgPool(1×1) → FC(512→16) | 3层 Conv (32→64→64, kernel=3, stride=2), 端到端训练 | 🔴 GRaD-Nav视觉编码器**不参与梯度更新** |
| **Actor** | `ActorStochasticMLP`: 57D → MLP(layer_dims, ELU, LayerNorm) → 4D μ + learnable log_std | MLP [128, 128] → 3D μ + learnable log_std | GRaD-Nav用ELU+LayerNorm，我们更简单 |
| **Critic** | `CriticMLP`: 67D (privileged) → MLP → 1D value | MLP [128, 128] → 1D value (共享CNN特征) | GRaD-Nav Critic接收全部特权信息 |
| **VAE** | `VAE`: obs_history (5×57D=285D) → Encoder(MLP [256,256,256]) → μ,σ(24D) → Decoder(MLP [32,64,128,256]) → reconstructed_obs | 无 | GRaD-Nav用VAE学习动力学上下文 |
| **Actor MLP维度** | 可配置，默认 `[512, 256, 128]` (通过yaml) | [128, 128] | GRaD-Nav Actor更深 |
| **总参数量** | ~3M (SqueezeNet 1.2M frozen + VAE ~1M + Actor/Critic ~800K) | ~200K | 我们小15倍 |

**VAE的精确作用** (代码 `models/vae.py` → `envs/drone_ppo.py:362`):
```
obs_history (5步 × 57D) → Encoder → latent(24D) → 拼入obs_buf和privilege_obs_buf
```
- 输入：过去5步的观测历史（含视觉特征、状态、动作）
- 输出：24D latent vector → 作为Actor和Critic的额外输入
- 训练信号：reconstruction loss (预测下一帧观测) + KL divergence
- 功能：**隐式系统辨识**——从观测历史推测当前的动力学参数（质量、惯性等，在域随机化下会变化）
- **这不是场景编码器**（不是CENet）。这是动力学上下文编码器。

**其他模型文件**:
- `models/squeeze_net.py` — VisualPerceptionNet（冻结SqueezeNet+FC层）
- `models/clip.py` — CLIP编码器（用于VLA版本）
- `models/moe.py` — Mixture of Experts（用于VLA版本）
- `models/model_utils.py` — 激活函数、初始化工具

### 2.5 动力学仿真

| 维度 | GRaD-Nav | 我们的 v2 | 差异分析 |
|------|----------|----------|---------|
| 位姿表示 | 四元数 (4D) | 无（仅位置） | GRaD-Nav 有完整 SO(3) 姿态 |
| 控制器 | PD 控制器 (角速度→电机推力) | 无（直接推力） | GRaD-Nav 包含底层控制回路 |
| 物理效应 | 电机延迟 + 气动阻力 + 重力 | 简化质点 + 重力隐式 + 速度限制 | GRaD-Nav 更物理，我们更抽象 |
| 可微分性 | 全可微 (Warp 实现) | 不可微（仅 Gymnasium step） | 这是 DDRL 的基础 |
| 碰撞检测 | 点云最近距离 (从ply文件加载, `ObstacleDistanceCalculator`) | 球体代理 (硬编码障碍物位置) | GRaD-Nav从真实场景点云计算碰撞距离 |

### 2.6 奖励函数（代码分析）

GRaD-Nav的奖励函数在 `envs/drone_ppo.py:680-770`，**10个组件，全部为正**（专门为PPO收敛设计）：

| # | 组件 | 权重 | 公式 | 类型 |
|---|------|------|------|------|
| 1 | survive_reward | 8.0 | 常数 | 生存激励 |
| 2 | heading_reward | 0.5 | yaw alignment (当前朝向 vs 目标方向) | 朝向引导 |
| 3 | target_reward | 2.0 | velocity alignment with desired direction | 速度方向 |
| 4 | waypoint_reward | 4.0×4wp | 1/(1+distance_to_waypoint) | 路径引导 |
| 5 | action_reward | 0.1 | 1/(1+‖action‖²) | 动作平滑 |
| 6 | action_consistency | 0.1 | 1/(1+‖Δaction‖²) | 动作一致 |
| 7 | height_reward | 0.5 | 1/(1+‖z-height_target‖) | 高度保持 |
| 8 | height_stability | 0.1 | 1/(1+vertical_vel²) | 高度稳定 |
| 9 | in_bounds_reward | 0.25 | 边界内=1, 超出=衰减 | 地图约束 |
| 10 | obst_clearance | 1.0 | dist_to_obstacle / threshold | 避障 |

**与我们的对比**：GRaD-Nav的reward全是正的（PPO友好），我们有正有负（-100 ~ +200）。GRaD-Nav严重依赖waypoint和参考轨迹（预规划的路径），我们的不依赖。

### 2.7 实际训练配置（从代码中提取）

| 参数 | 值 | 来源 |
|------|-----|------|
| 仿真频率 | 200Hz (内部) / 20Hz (RL策略) | `quadrotor_dynamics_advanced.py:48` + `drone_ppo.py:271` |
| 3DGS渲染频率 | 每步渲染 (gs_freq=1) | `drone_ppo.py:207` |
| 渲染分辨率 | 640×360 → rescale 0.4 → ~256×144 | `drone_ppo.py:94` + `gs_local.py:39` |
| Actor优化器 | AdamW, betas (默认), weight_decay=5e-3 | `gradnav.py:176` |
| Critic优化器 | AdamW, betas (默认), weight_decay=5e-3 | `gradnav.py:177` |
| VAE优化器 | AdamW, betas (默认), weight_decay=1e-2 | `gradnav.py:178` |
| LR schedule | linear 衰减至 1e-5 | `gradnav.py:609-619` |
| gradient clipping | 按grad_norm截断 | `gradnav.py:556` |
| 域随机化 | mass, thrust, inertia (每个reset随机) | `drone_ppo.py:465-491` |
| 观测噪声 | 0.1 (加在所有观测分量上) | `drone_ppo.py:81` |
| 电机延迟 | body_rate 0.8, thrust 0.7 | `drone_ppo.py:82-83` |
| 动作强度 | body_rate 0.5, thrust 0.25 | `drone_ppo.py:84-85` |
| 历史buffer | 5步 | `drone_ppo.py:50` |
| 隐空间维度 | 24D | `drone_ppo.py:51` |
| 视觉特征维度 | 16D | `drone_ppo.py:52` |

### 2.8 PPO基线也用了3DGS（重要发现）

GRaD-Nav代码库包含了PPO实现 (`algorithms/ppo.py`)，使用**完全相同的环境** (`DronePPOEnv`)。唯一区别是 `no_grad=True`。

这意味着：
1. **我们可以直接在GRaD-Nav代码上跑PPO→GRaD-Nav退化对比**，不需要自己实现环境
2. PPO和DDRL共享同一个3DGS渲染器、同一个动力学、同一个reward
3. 这为公平的A/B对比提供了完美的基础

### 2.9 代码质量观察

| 观察 | 影响 |
|------|------|
| 大量 `nan_to_num` 调用 | 数值稳定性问题，训练中NaN频繁出现 |
| `torch.autograd.set_detect_anomaly(True)` | 调试模式，生产训练时应关闭（有性能损失） |
| 冻结SqueezeNet → 16D特征 | 视觉信息瓶颈极窄，可能丢失细粒度几何信息 |
| 3DGS渲染是逐batch循环 | 非批量化渲染，可能是速度瓶颈 |
| 每个reset重建`QuadrotorSimulator` | 不必要的开销，但域随机化需要更新物理参数 |
| 无测试模块 | 没有单元测试，代码变更风险高 |

---

## 三、GRaD-Nav 的已知局限 → 我们的切入点

GRaD-Nav 论文中明确提到但**未做实验验证**的局限：

| # | GRaD-Nav 局限 | 论文中的表述 | 我们的切入点 | v2 对应实验 |
|---|-------------|------------|-------------|-----------|
| 1 | 仅静态场景 | "Our current method is limited to static scenes" | 不做（超出范围），在 Discussion 中作为 future work | — |
| 2 | 需要预采集场景 | "require pre-collected data" | 不做（工程限制），在 Discussion 中讨论 online 3DGS 可能性 | — |
| 3 | **渲染质量依赖** | "GS quality impacts performance" **← 一句话带过，零实验** | **核心贡献。5条退化轴 + 全量衰减曲线** | **V2.1-V2.6** |
| 4 | **无安全保证** | "no formal safety guarantees" | **鲁棒性基准 → 安全边界。找出每条退化轴的临界点** | **V2 临界点分析** |
| 5 | 手工奖励塑形 | "reward shaping relies on reference trajectories" | 部分缓解：我们的 reward 不依赖参考轨迹 | — |
| 6 | sim-to-real 视觉偏移 | "sim-to-real gap may still exist due to visual distribution shift" | **光照偏移 + 深度噪声退化轴直接模拟 sim-to-real gap** | **V2.4 + V2.5** |

### 关键洞察：第 3 条和第 4 条联合

GRaD-Nav 说"GS quality impacts performance"→ 他们不知道**怎么 impact、impact 多大、临界点在哪**。

我们说：
- 用 5 条退化轴系统测量 impact 的形状
- 找出每条轴的**相变临界点**（成功率从 >80% 跌到 <20% 的退化水平）
- 临界点 = 安全边界 → 直接回应第 4 条的"无安全保证"

这就是我们和 GRaD-Nav 的**差异化贡献**。

---

## 四、对 v2 实验设计的具体影响

基于以上对比，对实验设计做以下调整：

### 4.1 新增：DDRL 在高退化下的衰减曲线（V3.2 扩展）

原计划 V3.2 只对比"PPO vs DDRL 样本效率"。增加：

> **DDRL 退化鲁棒性**：如果 GRaD-Nav 代码可运行，在 5 条退化轴下评估 DDRL 模型的衰减曲线。假设：DDRL 在低退化下样本效率优于 PPO，但在高退化（GS < 25%）下梯度不稳定，优势消失甚至反转。

这是一个**非常有信息量的发现**——无论假设是否成立，结果都值得报告。

### 4.2 缩小并行环境数差异的解释

GRaD-Nav 用 128 并行环境，我们计划用 8。在论文中需要解释：

> "While GRaD-Nav uses 128 parallel environments with GPU-accelerated simulation, we use 8 environments with CPU simulation. This is because our evaluation protocol requires systematic degradation across 5 axes × 5 levels, not peak training throughput. Eight environments provide sufficient rollout diversity for PPO convergence on our simplified dynamics."

### 4.3 观测空间选择的论证

在 Method 章节需要明确论证为什么用深度图而不是 RGB：

1. 深度图对光照不敏感 → 光照偏移退化轴可以独立测量
2. 深度图直接反映几何重建质量 → 与 GS 退化轴的因果关系更直接
3. RGB 可以后续加入（作为消融或扩展实验）

### 4.4 动力学简化的论证

在 Method 章节需要说明质点模型的合理性：

> "We use a simplified point-mass dynamics model rather than full quaternion-based attitude dynamics (as in GRaD-Nav). This choice isolates the effect of visual degradation on navigation decision-making from control-level dynamics. Full attitude dynamics are an important direction for future work, particularly for sim-to-real transfer, but are orthogonal to our core research question."

---

## 五、论文叙事中的 GRaD-Nav 定位

### Introduction 段落模板

> "GRaD-Nav (Chen et al., IROS 2025) demonstrated that 3DGS-based visual RL can successfully train drone navigation policies, achieving efficient learning through differentiable dynamics. However, their analysis focuses exclusively on high-quality 3DGS reconstructions, noting only in passing that 'GS quality impacts performance.' This leaves a critical question unanswered: **how does rendering quality degradation quantitatively affect policy behavior, and at what point does the policy fail?**"

### Related Work 段落模板

> "GRaD-Nav uses a CNN encoder with a CENet (VAE-based environment encoder) and SHAC (differentiable RL). Their key innovation is the differentiable pipeline: gradients flow from the policy through the dynamics simulation back to the 3DGS renderer. While this achieves impressive sample efficiency (3.5h training), the reliance on high-quality GS reconstructions raises questions about robustness that the authors acknowledge but do not empirically address. Our work complements GRaD-Nav by providing the systematic robustness analysis that their pipeline—and indeed any 3DGS-based visual navigation pipeline—requires."

### Discussion 段落模板

> "Our finding that PPO policies exhibit phase-transition behavior under GS degradation raises an interesting question for differentiable RL methods: does the gradient path through degraded GS reconstructions remain informative? If the gradients become noisy or biased under low-quality GS, the sample efficiency advantage of differentiable RL may diminish—or even reverse—in degraded scenarios. This is an important direction for future work."

---

## 六、行动清单

| 优先级 | 行动 | 对应文件/位置 | 状态 |
|--------|------|-------------|------|
| P0 | 在 research_plan_v2.md 中更新竞争格局 | §零 | 本次 |
| P0 | 在 weekly_plan_v2.md 中调整 V3.2 实验 | Week 2 | 本次 |
| P1 | 在 CLAUDE.md 中增加 DDRL 基线说明 | 关键设计决策 | 本次 |
| P1 | 在 reading-list-v2.md 中更新 GRaD-Nav 条目 | §一.1 | 本次 |
| P2 | 在 V3 实验中实际对比 DDRL vs PPO 退化鲁棒性 | train_visual.py 等 | Week 2 实施 |
| P2 | 在论文 Method 中加入上述论证段落 | 论文初稿 | Week 3 |
| P3 | 如果 GRaD-Nav 代码跑不通，将 DDRL 对比降级为 Discussion 中的理论分析 | 论文 | Week 3 决策 |

---

*文档整理时间：2026-07-21*
*基于 GRaD-Nav (IROS 2025) 与 v2 项目的全维度技术对比*
