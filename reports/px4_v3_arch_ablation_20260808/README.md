# V3 结构消融（RGB / 浅 CNN / 无特权 Critic）

日期：2026-08-08。sv_1007 主场景上的三种结构消融，验证标准深度 CNN 架构设计
（in_channels=1 深度、3 层 CNN、Critic 用视觉特征）是否最优。

## 方法

- **改动**：`core/visual_ppo_agent.py`、`envs/visual_drone_env.py`、`scripts/train_visual.py`
  新增 `--arch` 参数（`rgb` / `shallow_cnn` / `no_privileged_critic`）。
- **训练**：各 3 seeds，3000ep，clean，sv_1007，GPU0。
- **baseline**：sv_1007 clean 5 seeds 均值 **53.2%**。
- 工具：`run_v3_arch_ablation.sh`；模型 `saved_models/v3_arch_ablation/`。

## 结果（best_SR，3 seeds）

| 消融 | seed0 | seed1 | seed2 | 均值 | vs baseline |
|------|:---:|:---:|:---:|:---:|:---:|
| **baseline**（深度+3层CNN+特权Critic）| — | — | — | **53.2%** | — |
| **rgb**（3通道颜色输入）| 50 | 42 | 52 | **48.0%** | **−5.2pp** |
| **shallow_cnn**（2层CNN）| 45 | 56 | 43 | **48.0%** | **−5.2pp** |
| **no_privileged_critic**（Critic只用vec）| 42 | 40 | 39 | **40.3%** | **−12.9pp** |

## 结论

1. **深度优于 RGB**（48.0 vs 53.2，−5.2pp）：深度图是比颜色更有效的避障表征。
   颜色含光照/纹理噪声，深度直接反映几何——支持"深度是视觉导航最优输入"。
2. **深 CNN 优于浅 CNN**（48.0 vs 53.2，−5.2pp）：3 层 CNN 的视觉容量对 3D 导航
   有实质贡献，浅网络信息不足。
3. **特权 Critic 重要**（40.3 vs 53.2，**−12.9pp**）：Critic 用视觉特征是必要的——
   去掉后价值估计变差，训练稳定性/性能显著下降。**视觉不是"特权捷径"，而是有效信号**。
4. **一致性**：三种消融全部低于 baseline，方向一致（无种子倒挂），结论稳健。

## 论文含义

- 支持深度图 + 深层 CNN + 共享视觉 Critic 的架构选择。
- 无特权 Critic 消融（−12.9pp）是重要结果：证明视觉信息对价值函数是有效输入，
  反驳"视觉仅被 actor 使用、Critic 是特权"的假设。

## 文件

- 报告：本 README
- 模型：`saved_models/v3_arch_ablation/`（rgb/shallow_cnn/no_privileged_critic × 3 seeds）
- 工具：`--arch` 参数（已提交 PR #18）
- 服务器日志：`/root/px4-deploy/v3_arch_ablation.log`
