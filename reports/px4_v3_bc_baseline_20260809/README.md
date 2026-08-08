# V3 BC 对照基线（行为克隆 vs RL）

日期：2026-08-09。sv_1007 主场景上的行为克隆（BC）对照，用 waypoint oracle
作为 expert，对比 RL（PPO）性能。补全论文的 RL vs 行为克隆对比。

## 方法

1. **Expert 数据采集**（`scripts/collect_oracle_data.py`）：用最短路径 waypoint
   控制器（`eval_waypoint_oracle.py` 的 WaypointController）在 sv_1007 采样 expert
   轨迹，记录 (depth, vec) → action 对。**200 episodes，oracle 成功率 98%**，
   采集 **16,871 个样本**。
2. **BC 训练**（`scripts/train_bc.py`）：CNN(depth) + MLP(vec) → action，监督回归
   （MSE），架构与 VisualPPO 编码器一致。50 epochs，batch 256，lr 1e-3。
   **train_mse=0.0008，val_mse=0.0111**。
3. **评估**（`scripts/eval_bc.py`）：50 episodes，sv_1007，与 RL baseline 相同环境。

## 结果（sv_1007）

| 方法 | SR | CR | 说明 |
|------|:---:|:---:|------|
| **BC（行为克隆）** | **34.0%** | 64.0% | 监督学习复现 oracle |
| **RL（PPO clean 5 seeds）** | **53.2%** | ~57% | 端到端强化学习 |

**BC 34% < RL 53.2%（−19.2pp）**。

## 结论

1. **RL 优于 BC（−19.2pp）**：强化学习的闭环适应（探索 + 奖励信号）优于行为克隆
   的静态监督复现。BC 从 oracle 复现"应该怎么做"，但无法像 RL 一样在交互中学习
   对观测噪声/初始状态扰动的鲁棒应对。
2. **BC 基线有意义**：34% 证明任务可被监督学习部分解决，但达不到 RL 水平——
   支持"RL 是视觉导航正确方法"的论文主张。
3. **oracle 98% vs RL 53.2%**：oracle 有全局路径信息（上帝视角），远超学习策略；
   但 BC（从 oracle 学）只达 34%，说明 oracle 动作无法被端到端复现（视觉歧义、
   分布漂移）。

## 论文含义

- **RL vs BC 对照**：RL 显著优于 BC，补齐"为什么用 RL 而非监督学习"的论证。
- **oracle 上界**：98% 是任务可解上界（上帝视角路径），RL 53.2% 是合理下界，
  BC 34% 说明监督复现的局限。

## 文件

- 工具：`scripts/collect_oracle_data.py`、`scripts/train_bc.py`、`scripts/eval_bc.py`
- 数据：`/root/px4-deploy/bc_oracle_data.npz`（16,871 样本）
- 模型：`saved_models/v3_bc_baseline/bc_model.pth`
- 评估：`/root/px4-deploy/bc_eval.json`（SR=34%）
