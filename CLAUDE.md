# CLAUDE.md — UAV Visual RL 3D Path Planning Project

## 项目身份

基于深度强化学习的无人机 3D 视觉导航研究项目。
当前阶段: 路线 v2 — 3DGS 重建真实场景 + 视觉 PPO + 渲染质量鲁棒性分析 (2026年7-8月)。
目标会议: ICRA 2027 (~2026年9月截止)。

## 核心架构

```
手机视频 → 3DGS重建 → 深度图渲染 → CNN编码器 → 共享MLP → 动作(3D推力)
                                              ↑
                              PPO Update (GAE + Clip)
```

**对比架构 (GRaD-Nav, Stanford IROS 2025)**:
```
手机视频 → 3DGS重建 → RGB渲染 → CNN(512→256→128) + CENet(VAE) → Actor-Critic(512×256×128)
                                              ↑                        ↓
                              SHAC Update (梯度穿过动力学+渲染器) → 动作(体角速度4D)
```
- GRaD-Nav: SHAC可微分RL, 128并行环境, 3.5h训练, 全四元数+PD控制器+电机延迟
- 我们: PPO无模型RL, 8并行环境, 20-35h训练, 简化质点模型
- **为什么选PPO?** 鲁棒性分析不需要极致样本效率；DDRL梯度穿过退化GS可能不稳定（这是我们要验证的假设）；PPO的reward信号不依赖GS精度→更公平的鲁棒性测试平台
- 详细对比: `docs/grad-nav-comparison.md`

**两个代码版本**:
- `rlproject -jxl-rlib/` — 旧版 (CNN+MLP, 0%成功率, 已弃用, 保留作基线对比)
- `rlproject-swift-improved/` — v1 向量PPO基线 (98%成功率, 保留作对照实验)

**v2 新增文件** (待建):
- `visual_drone_env.py` — 3DGS渲染环境
- `visual_ppo_agent.py` — CNN编码器 + PPO
- `train_visual.py` — 视觉训练管线

## 核心文档

| 文件 | 职责 |
|------|------|
| `research_plan_v2.md` | **当前研究方案** — 3DGS视觉RL + 渲染质量鲁棒性 |
| `weekly_plan_v2.md` | **当前周计划** — 3周逐日计划 |
| `docs/reading-list-v2.md` | 论文阅读清单 (15篇, 三级优先级) |
| `docs/plan-v1-to-v2-migration.md` | v1→v2 迁移记录 (决策理由、变化对比) |
| `past/` | v1 历史归档 (旧方案、旧实验、旧代码) |

## 关键数值常量 (v1 基线, v2 沿用物理参数)

- 空间: 20×20×10m ([-10,10], [-10,10], [0,10])
- 无人机: 质量1.0, 最大推力10.0, 最大速度5.0m/s, dt=0.05s (20Hz)
- Episode: 500步上限, 成功判定: 距目标 ≤ 0.5m
- 碰撞判定: 距障碍物 ≤ 1.5m (radius + threshold)

## v1 训练超参数 (向量PPO基线, 视觉版参考调整)

- rollout_steps=2048 (8环境×256步), gamma=0.99, gae_lambda=0.95
- PPO clip_eps=0.2, epochs=10, minibatch=64
- lr=3e-4 线性衰减至0, hidden_dim=128
- 自适应熵: target_entropy=-3, initial_coeff=0.01
- 梯度裁剪: max_grad_norm=0.5

## v1 训练命令 (基线对照)

```bash
cd rlproject-swift-improved
python train.py                           # 默认3000 episodes
python train.py --episodes 500 --lr 1e-4  # 快速测试
python eval_baseline.py                   # 评估已保存模型
```

---

## AI 协作原则 (Cognitive Debt Defense V2)

> 约束在前，代码在后。一个 Prompt = 一个模块。每增加一行代码，就增加一行理解。

### 工程约束

1. 遵守项目规则文件中定义的技术栈、目录结构、命名规范和安全红线
2. 不要一次生成整个功能。每次只交付一个模块（≤200 行代码）
3. 不要在没有确认的情况下修改已存在的核心文件
4. 不要使用 eval()、不要硬编码密钥、不要忽略输入校验

### 认知交付（代码 + 理解）

每次生成代码，附带"知识包":
- 位置（在架构中的哪里）
- WHY（为什么这个方案而不是其他方案）
- 数据流（输入→变换→输出）
- 边界（明确不负责什么）
- 风险点（最可能的故障模式）

生成 3 个"理解自测"问题（测试理解，不测试正确性）。

检查 `docs/knowledge-map.md` 是否需要更新，给出具体建议（如不存在则标注缺失）。

### 质量保证

代码完成后自我检查:
- 输入为空 → 会怎样？
- 并发调用 → 会怎样？
- 错误信息 → 调用方能定位问题吗？
- 有没有硬编码值可以提取？

用"3 个月后的同事"视角审视代码可读性:
- 去掉对话上下文后，代码本身是否自解释？
- 关键设计决策的 WHY 是否体现在代码注释中？

永远不要只交付代码。交付: 代码 + 理解 + 质量保证。

### 可用 Agent（@ 调用）

| Agent | 用途 | 何时调用 |
|-------|------|---------|
| `@spec-writer` | 模糊需求 → 结构化 Spec | 需求不明确时 |
| `@modular-architect` | 大需求拆分为 3-5 个模块 | 预计 >200 行代码 |
| `@fast-debugger` | 粘贴错误 → 定位 → 最小修复 | 遇到报错时 |
| `@knowledge-pack-generator` | 生成结构化知识包 | 每次代码改动后 |
| `@knowledge-map-maintainer` | 更新认知地图 | 一轮开发结束后 |
| `@microworld-builder` | 生成交互式训练场脚本 | 复杂模块需要深度理解 |
| `@understanding-reviewer` | 4 层理解审查 | PR Review 时 |
| `@understanding-gate` | 合并前理解门槛检查 | 合并前最后一步 |

### 提示词长度原则

- 日常高频使用: 精简版（≤150 字）— 只保留核心约束
- 首次使用 / 复杂场景: 完整版 — 包含完整上下文
- 结构化约束语法: `[需求] [输入约束] [输出约束] [禁止] [质量门槛]`
- 遇到错误: 贴错误 > 描述错误

---

## 关键设计决策 (v1 经验, v2 继承)

1. **GAE per-environment**: 并行环境下必须按环境分别计算GAE, 否则跨环境边界优势传播错误
2. **共享特征提取器**: Actor和Critic共享, 减少参数, 加速训练 (Swift实践)
3. **7组件密集奖励**: 每个时间步都有非零信号, 比稀疏奖励信号强得多
4. **tanh(mean)**: 将动作均值限制在(-1,1), 匹配归一化动作空间
5. **v1 纯向量 → v2 视觉**: v1的14D完美向量在真机上不可用。v2升级为3DGS深度图, 从"上帝视角"变为"机载视角"
6. **v2: PPO vs DDRL (SHAC)**: GRaD-Nav用可微分RL (SHAC) 实现3.5h训练。我们选PPO的理由:
   - 鲁棒性分析不需要极致样本效率 (3000ep PPO足够收敛到>80% SR)
   - DDRL梯度穿过退化3DGS渲染器 → 梯度可能不稳定 (待验证假设)
   - PPO的reward信号不依赖GS精度 → 更公平的鲁棒性测试平台
   - 已有成熟PPO管线, 降低工程风险
7. **v2: 深度图 vs RGB**: GRaD-Nav用RGB。我们选深度图:
   - 光照不变性 → 光照偏移退化轴可独立测量
   - 几何重建质量的直接反映 → 与GS退化轴的因果链更直接
   - RGB可后续加入作为消融/扩展
8. **v2: 简化动力学 vs 全四元数**: GRaD-Nav用完整四元数姿态+PD控制器+电机延迟+气动阻力。我们用质点模型:
   - 隔离视觉退化效应与控制动力学 → 实验结论更干净
   - 与v1向量基线保持动作空间一致 → v1↔v2差异可归因于"视觉vs向量"

## 过去归档

v1 子课题A (向量噪声鲁棒性) 已完成并作为 v2 论文的 preliminary study。
v1 子课题B (SAC环境策略) 已放弃, 留作 future work。
所有 v1 文件在 `past/` 中: 旧方案、旧实验脚本、评估结果、训练日志、SAC代码。
