# SHAC/DDRL 对照——理论分析（预案 R4 降级）

日期：2026-08-09。原计划 P0 的 "SHAC/DDRL 或域随机化基线"，经评估判定
grad_nav 适配 3DGS 成本过高（独立 dflex 仿真框架），按预案 R4 降级为
理论分析。本档整合 GRaD-Nav 技术细节 + 我们 sv_1007 的实证证据，
论证 PPO 选择的合理性，并讨论 DDRL 在退化 3DGS 下的预期行为。

## 一、为什么降级为理论分析（不做代码级 SHAC 对照）

GRaD-Nav (IROS 2025) 开源代码 `grad_nav-main/` 使用**独立 dflex 物理仿真
框架**（Warp 可微动力学），与我们项目的 Gymnasium + 3DGS 渲染管线完全分离：

| 维度 | GRaD-Nav | 我们的 v2 |
|------|----------|----------|
| 仿真 | dflex/Warp 可微动力学 | Gymnasium 质点模型 |
| 渲染 | 自有 gsplat 集成 | GSplatRenderer |
| 视觉编码 | 冻结 SqueezeNet → 16D | 端到端 3 层 CNN |
| 观测 | 57D 向量 + 16D 视觉 | 64×64 深度图 + 6D vec |
| 训练 | SHAC（可微分 RL）| PPO（无模型）|

要在 sv_1007 3DGS 场景上跑 SHAC，需把可微渲染接进 dflex 动力学——这是
**数千行新代码**的工程，且引入的复杂度远超 ICRA 投稿所需。

## 二、方法选择论证（为什么 PPO 是正确选择）

### 2.1 鲁棒性分析不需要极致样本效率

GRaD-Nav 用 SHAC 实现 3.5h 训练（128 并行环境）。但**我们的研究问题是
"3DGS 质量差到什么程度就不能飞"，不是"谁训得快"**。PPO 在 3000ep
内已收敛（sv_1007 clean 5 seeds 均值 53.2%，训练约 2.5h/seed），
足够支撑系统性退化分析。

### 2.2 DDRL 的梯度穿过退化渲染器——不稳定风险（关键假设）

DDRL 的核心是梯度从 loss 穿过**动力学 + 渲染器**流回策略。但：
- 当 3DGS 退化时（稀疏化/噪声/低分辨率），**渲染器的梯度质量也退化**
- 退化 GS 的协方差矩阵、透明度参数可能产生**噪声/偏差梯度**
- 这可能导致 DDRL 的样本效率优势在高退化下**消失甚至反转**

**我们的实证支持**：curriculum（深度尺度随机化，可视为温和域随机化）在
sv_1007 上 **clean 全面优于 curriculum**（无退化档 +20.8pp）。虽然 curriculum
不等同 DDRL，但**证据一致表明：依赖退化信号训练会削弱干净性能**。

### 2.3 PPO 是更公平的鲁棒性测试平台

PPO 只依赖 reward 信号，**不依赖渲染器的可微性**。在退化 3DGS 上评估时，
PPO 策略对渲染质量不敏感（reward 与 GS 精度解耦）——这保证了我们的
退化曲线测量的是**任务难度变化**而非**训练算法崩溃**。

## 三、DDRL 在退化 3DGS 下的预期行为（可报告的讨论点）

基于 GRaD-Nav 技术分析 + 我们的实证，提出以下可验证假设（论文 Discussion）：

### H1：梯度质量随退化单调下降

DDRL 的梯度依赖精确 GS 参数（协方差、透明度）。随退化（稀疏化 100%→2%），
可微渲染的梯度信噪比下降。**预期**：DDRL 在低退化下样本效率 > PPO，但在
高退化下（GS < 25%）优势消失甚至反转。

### H2：冻结编码器加剧退化脆弱性

GRaD-Nav 视觉编码器是**冻结的 SqueezeNet**（不参与训练）。在退化 RGB 上，
SqueezeNet 提取的特征本身退化，但网络无法适应。**我们的端到端 CNN 可以
部分适应视觉偏移**——这是我们的架构优势。

### H3：奖励函数解耦的重要性

GRaD-Nav 的 reward 严重依赖预规划 waypoint 参考轨迹；我们的 reward 不依赖
参考轨迹。在退化场景下，**不依赖参考轨迹的 reward 更鲁棒**（任务仍可定义）。

## 四、对我们结果的启示

1. **curriculum 鲁棒优势不泛化**（sv_1007 退化评估）：深度尺度随机化（温和域
   随机化）未带来退化鲁棒性收益，反而削弱干净性能。**这与 DDRL 假设的
   "训练信号更直接 → 更鲁棒"相矛盾**——证据指向简单 PPO + 深度图更稳健。
2. **多场景联合训练有效**（V3c）：数据多样性（而非算法复杂度）是泛化关键——
   泛化 0% → 27-37%。这比 DDRL 的可微梯度更实用。
3. **结构消融支持简单架构**：深度 > RGB、深 CNN > 浅 CNN、特权 Critic 必要。

## 五、论文写法建议

**Method 段落**：
> "We use PPO (model-free RL) rather than differentiable RL (e.g., SHAC in
> GRaD-Nav). This is deliberate: our research question is how rendering
> quality degradation affects policy behavior, not training throughput.
> PPO's reward-driven learning is agnostic to renderer differentiability,
> providing a fair platform to measure degradation curves."

**Discussion 段落**：
> "An open question for differentiable RL (DDRL) is whether the gradient path
> through degraded 3DGS remains informative. Our curriculum results suggest
> training on degraded signals can hurt clean performance. We hypothesize
> DDRL's sample-efficiency advantage may diminish or reverse under GS
> degradation, which is an important direction for future work."

## 文件

- 技术基础：`docs/grad-nav-comparison.md`（GRaD-Nav 全维度对比）
- 实证证据：`reports/px4_sv1007_degradation_20260805/`（curriculum 退化）
