# 精读笔记 — GRaD-Nav（🔴 #1）

> 阅读日期：2026-08-02 | 来源：`docs/2503.03984v3.pdf`（本地 8 页全文，v3 2025-07-30）
> 笔记类型：🔴 必读精读 | 状态：✅ 完成（含实验数据核对）

## 一句话概括

3DGS 场景 + 可微分四旋翼动力学 + SHAC 算法端到端训练视觉导航策略，
用 CENet(β-VAE) 上下文编码器实现运行时环境自适应，128 并行环境 3.5h 训练，
真机零样本迁移穿过不同位置的门。

## 方法要点（从 PDF 实读确认）

- **算法**：SHAC（Short-Horizon Actor-Critic），沿论文 Eq.(2) 在 h=32 子窗口内
  反向传播梯度穿过动力学；Critic 用 Eq.(3) MSE 目标。BPTT 整轨迹 600 步作对照。
- **动力学**：可微分四旋翼，体角速度 ω_d∈R³ + 归一化推力 c_t∈[0,1] 控制输入；
  状态 16D = (p, q, v, ω, a)。PD 姿态控制器 Eq.(7-8)、电机延迟 + 阻力（Table III）。
- **视觉**：冻结 SqueezeNet → FC 512→24 视觉嵌入 e_t（不参与 RL 梯度，仅特征）。
- **策略/价值**：3 层 MLP 512/256/128（ELU + LayerNorm），价值网络额外访问特权
  观测 s_t = o_t + 位置 p_t + 深度先验 d_t∈R²⁴（3DGS 深度图平均池化下采样）。
- **CENet**：β-VAE 编码器-解码器，输入最近 5 步历史观测 o^H_t，输出 latent z_t∈R¹⁶；
  损失 = 重建 MSE + β·KL。消融显示去掉 CENet 真机成功率大幅下降。
- **观测** o_t = [高度, 四元数, 线速度, 当前动作, 上一动作]（Eq.11）—
  无需 x-y 位置估计，全部来自机载传感器，支撑 0-shot 迁移。
- **奖励**：10 组件（Table II），全部带符号：survival +8.0、waypoint exp(-d_wp) +2.0、
  避障 d_obst +1.0（阈值 0.5m）、yaw 对齐 +0.25、速度/姿态/高度/动作平滑惩罚。
- **域随机化**（Table III）：质量 [1.0,1.5]kg、推力 [22,30]N、惯量、电机延迟 [0.5,0.8]、
  阻力系数 [0.4,0.6]。
- **课程训练**：3 个不同门位置的 3DGS 环境滚动训练（每环境 100 epoch × 5 次），
  单一策略适应未见过的门位置 + 干扰物。
- **仿真性能**：128 envs、~100m² 房间、3DGS 1.5GB，0.05s 步长(20Hz)，
  i9-13900K + RTX 4090 上 0.07s/步（动力学 33.5%、渲染 55.7%、碰撞 10.8%）。
- **硬件**：Pixracer 低层控制器 + Jetson Orin Nano + RealSense D435，策略推理 30Hz。

## 关键实验数据（可直接引用）

### 训练效率（Fig.2）

| 算法 | 时间 | 样本 |
|------|------|------|
| GRaD-Nav (SHAC) | **3.5h** | 1e6 内收敛 |
| BPTT | 15h | 全轨迹反传，内存受限 |
| PPO | 22h | **1e7 样本内训不出满意策略** |

> ⚠️ 这直接支撑我们选 PPO 的理由，但也是审稿人会质疑的点：GRaD-Nav 论文实测
> PPO 在同任务上 22h 仍差。我们需在论文中说明：我们的目标不是样本效率而是
> **鲁棒性测试平台**，PPO 的 reward 不依赖 GS 精度（CLAUDE.md 决策 #6）。

### 消融（Table IV，10 次 rollout 成功率，long traj.）

| 消融 | middle gate | right gate |
|------|:-----------:|:----------:|
| w/o visual obs | 0/10 | 0/10 |
| **w/o RGB; w/ depth** | **0/10** | **0/10** |
| w/o velocity | 0/10 | 0/10 |
| w/o CENet | 4/10 | 5/10 |
| Proposed | 8/10 | 7/10 |

> 🔴 **重要**：GRaD-Nav 实测「深度图代替 RGB」在仿真 0/10、真机 1-2/10（Table V）。
> 这与我们的深度图方案正面冲突（CLAUDE.md 决策 #7）。差异可能在于：
> 他们用 SqueezeNet 预训练特征（为 RGB 设计）+ 光流缺失；我们端到端训练 CNN 于
> 深度图。**论文写作必须正面回应**：可在 Discussion 中比较「预训练 RGB 特征 vs
> 端到端深度学习」的差异，或补一个 RGB 消融对照实验。

### 真机迁移（Table V，sim | real）

| 方法 | Left | Middle | Right |
|------|:----:|:------:|:-----:|
| w/o RGB; w/ depth | 4/10 \| 1/10 | 7/10 \| 1/10 | 5/10 \| 0/10 |
| w/o CENet | 9/10 \| 0/10 | 10/10 \| 2/10 | 7/10 \| 1/10 |
| Proposed | 10/10 \| 7/10 | 10/10 \| 7/10 | 9/10 \| 6/10 |

- 成功判据：不提前终止 + 依次到达每个 waypoint（≤0.3m）+ 全程距障碍 ≥0.2m。
- CENet latent PCA（Fig.6）：穿过门时 latent 分布比接近/离开阶段更紧凑—
  上下文编码器在"关键避障时刻"压缩了环境信息。可借鉴的评估可视化方法。

## 与我们项目的关系

| 维度 | GRaD-Nav | 我们 |
|------|----------|------|
| 算法 | SHAC（可微分 RL） | PPO（无模型，鲁棒性测试平台） |
| 视觉 | 冻结 SqueezeNet + RGB | 端到端 CNN + 深度图 |
| 上下文 | CENet β-VAE（真机部署时运行） | 无（未来工作：退化感知条件） |
| 场景 | 3 环境课程训练、100m² | 单场景 gate_mid 窄走廊 |
| 动力学 | 四元数 + PD + 电机延迟 | 质点模型（隔离视觉效应） |
| 渲染质量鲁棒性 | **一句话带过，零实验** | **我们的核心贡献** |

**切入点**（GRaD-Nav 局限 → 我们的机会）：
1. "GS quality impacts performance" 无实验验证 → 我们 5 轴退化系统实验
2. 深度图 vs RGB 结论矛盾（他们深度失败）→ 我们端到端深度学习的对照证据
3. 无对抗/退化条件测试 → 我们的临界退化点分析
4. 手工奖励塑形依赖参考轨迹 → 我们的 7 组件奖励不依赖参考轨迹

## 可直接引用的句子（草稿）

> "GRaD-Nav [ref] integrates 3D Gaussian Splatting with differentiable
> dynamics (SHAC) for sample-efficient visual drone navigation, but its
> rendering-quality robustness is not evaluated: the effect of scene
> degradation on policy performance is asserted rather than measured."

## 相关任务

- [ ] 下载 GRaD-Nav 官方代码（github.com/Qianzhong-Chen/grad_nav）核对 Table IV 复现设置
- [ ] 论证深度图路线时引用本笔记的消融矛盾点，决定是否加 RGB 消融
- [ ] 域随机化参数（Table III）作为我们未来 DR 扩展的参考基线
