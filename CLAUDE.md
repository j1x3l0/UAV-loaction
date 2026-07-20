# CLAUDE.md — UAV RL 3D Path Planning Project

## 项目身份

基于深度强化学习 (PPO) 的无人机 3D 路径规划研究项目。
目标: 训练智能体在 20×20×10m 空间中从起点飞向目标点, 避开 3 个静态障碍物。
当前阶段: 8周研究冲刺 (2026年7-9月), 目标发表级成果。

## 核心架构

```
状态 (14D向量) → ActorCritic网络 → 动作 (3D推力) → 环境仿真 → 奖励 (7组件)
         ↑                                                      |
         └────────────── PPO Update (GAE + Clip) ←──────────────┘
```

**两个代码版本**:
- `rlproject -jxl-框架-rlib/` — 旧版 (CNN+MLP, 0%成功率, 已弃用, 保留作基线对比)
- `rlproject-swift-improved/` — **当前版本** (Swift风格MLP, 98%成功率)

## 关键文件

| 文件 | 职责 | 最常修改 |
|------|------|----------|
| `drone_env.py` | 14D状态 + 7组件奖励 + 物理仿真 | 奖励函数权重, 状态特征 |
| `ppo_agent.py` | ActorCritic网络 + GAE + PPO更新 | 网络架构, 超参数 |
| `train.py` | 8环境并行训练 + 评估 + TensorBoard | 训练流程, 评估指标 |
| `config.py` | 超参数配置 + CLI参数解析 | 超参数默认值 |

## 关键数值常量

- 空间: 20×20×10m ([-10,10], [-10,10], [0,10])
- 障碍物: 3个静态球体, 半径1m, 位于 [2,2,3], [6,3,5], [3,7,4]
- 无人机: 质量1.0, 最大推力10.0, 最大速度5.0m/s, dt=0.05s (20Hz)
- 目标: 随机生成在 [5,8]×[5,8]×[2,8]
- Episode: 500步上限 (25s真实时间)
- 成功判定: 距目标 ≤ 0.5m
- 碰撞判定: 距障碍物 ≤ 1.5m (radius + threshold)

## 训练超参数 (Swift版)

- rollout_steps=2048 (8环境×256步), gamma=0.99, gae_lambda=0.95
- PPO clip_eps=0.2, epochs=10, minibatch=64
- lr=3e-4 线性衰减至0, hidden_dim=128
- 自适应熵: target_entropy=-3 (=-action_dim), initial_coeff=0.01
- 梯度裁剪: max_grad_norm=0.5

## 关键设计决策 (为什么这样设计)

1. **14D纯向量状态 (无图像)**: 旧版16×16深度图像信息量极低, 移除后性能大幅提升
2. **共享第一层**: Actor和Critic共享特征提取器, 减少参数, 加速训练 (Swift实践)
3. **GAE per-environment**: 并行环境下必须按环境分别计算GAE, 否则跨环境边界优势传播错误(曾导致0%成功率的关键bug)
4. **7组件密集奖励**: 旧版3组件稀疏奖励信号不足。新奖励每个时间步都有非零信号
5. **tanh(mean)**: 将动作均值限制在(-1,1), 匹配归一化动作空间

## 训练命令

```bash
cd rlproject-swift-improved
python train.py                           # 默认3000 episodes
python train.py --episodes 500 --lr 1e-4  # 快速测试
python eval_baseline.py                   # 评估已保存模型
python smoke_test.py                      # 验证GAE修复
```

## 当前状态

- 最佳模型: `saved_models/ppo_swift_3000ep_20260712_115059` (98%成功率 @ Ep201)
- TensorBoard: `logs/ppo_swift_3000ep_20260712_115059/`
- 当前训练轮数: ~231/3000

## 文档索引

- `improvement_report.md`: 改进前后完整对比 (必读)
- `research_plan.md`: 8周研究计划 + 论文阅读清单
- `weekly_plan.md`: 逐日实验计划
- `swift_improvement_directions.md`: 未来创新方向 (A-F类)
- `docs/knowledge-map.md`: 系统认知地图 + 雾图
- `docs/understanding-gate.md`: PR合并前理解审查
- `docs/cognitive-debt-guide.md`: 认知债务防御实践指南
