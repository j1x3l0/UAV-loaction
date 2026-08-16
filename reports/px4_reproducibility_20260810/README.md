# 可复现性报告（sv_1007 视觉 RL 全实验）

日期：2026-08-10。整理全部实验的配置、种子、统计摘要与 checkpoint 哈希，
满足可复现、可提交的正式报告要求（评审优先级第 5 条）。

## 环境

- **服务器**：root@172.16.30.53（内网），NVIDIA RTX 3090 ×2，仅 GPU0 用于本项目训练
- **Python**：conda env `myconda`（Python 3.10）
- **核心依赖**：torch 2.1.1+cu118, numpy 1.25.2, scipy 1.11.2, plyfile, gsplat
- **代码**：本仓库（rlproject-swift-improved/）

## 主场景 sv_1007

- **场景**：sv_1007_gate_mid（完整 gate，22×25×9m，265,631 高斯）
- **对齐**：合成 identity（fx≈97.14 相机，无场景旋转），`sv1007_alignment.json`
- **动力学**：质点模型，mass=1.0，max_thrust=10.0，max_speed=5.0，dt=0.05s
- **任务**：500 步上限，成功判定距目标 ≤0.5m，碰撞半径 0.5m

## 训练配置（统一）

| 参数 | 值 |
|------|-----|
| episodes | 3000 |
| envs | 2（多场景 3）|
| rollout_steps | 256 |
| gamma / gae_lambda | 0.99 / 0.95 |
| clip_eps | 0.2 |
| epochs / minibatch | 5 / 32 |
| lr | 3e-4 线性衰减至 0 |
| hidden_dim | 128 |
| 自适应熵 | target 2.5 |

## 实验清单（配置 + 种子）

### 1. clean 基线（5 seeds）
- 命令：`train_visual.py --episodes 3000 --envs 2 --degradation clean --renderer gsplat`
- 种子：0, 1, 2, 3, 4
- best_SR：43/60/55/64/44，均值 **53.2%**
- 模型：`v3_sv1007/seed{0-4}_3000_final.pth`

### 2. curriculum（scale_curriculum，5 seeds）
- 命令：`--degradation scale_curriculum`
- 种子：0-4
- best_SR：41/40/51/34/36，均值 **40.4%**
- 模型：`v3_sv1007_curriculum/seed{0-4}_robust_best.pth`
- curriculum 概率：foundation[.6,.25,.15,0] → transition[.4,.25,.25,.1] → robustness[.3,.2,.3,.2]

### 3. V3c 单场景（left/right，各 3 seeds）
- left best_SR：53/52/55（均值 53.3%）
- right best_SR：43/45/45（均值 44.3%）
- 模型：`v3c_sv917/`

### 4. V3c 多场景联合（3 seeds）
- `--multiscene sv1007,left,right --envs 3`
- best_SR：82/83/85（均值 **83.3%**）
- 跨场景泛化：sv_1007 28.7% / left 26.7% / right 37.3%
- 模型：`v3c_multiscene/seed{0-2}_final.pth`

### 5. 结构消融（3 seeds）
- rgb / shallow_cnn / no_privileged_critic，各 3 seeds
- 均值：48.0% / 48.0% / 40.3%（vs baseline 53.2%）
- 模型：`v3_arch_ablation/`

### 6. BC 基线（3 seeds）
- oracle 数据：200ep，98% 成功率，16,871 样本
- BC best_SR：34/30/32（均值 32.0%）
- 模型：`v3_bc_baseline/bc_seed{0-2}.pth`

## 配对统计检验（评审优先级第 1 条）

| 对比 | 显著 seed 数 | 说明 |
|------|:---:|------|
| baseline vs const_depth | 4/5 | 深度必要 |
| baseline vs no_velocity | 5/5 | 速度决定性 |
| baseline vs no_target_dir | 5/5 | 目标决定性 |
| PPO vs BC | 2/3 | RL 强趋势 |
| clean vs curriculum | 4/5 | clean 优于 curriculum |

工具：`scripts/paired_stats.py`（McNemar + bootstrap CI + Cohen's g）

## 评估协议

- 输入消融：`eval_v3_ablation.py --ablation all --episodes 50 --seed 20260809`
- 退化：`eval_degradation.py --axis all --episodes 50 --seed 20260805`
- 跨场景：`eval_v3c_cross_scene.py --episodes 50 --seed 20260806`
- BC：`eval_bc.py --episodes 50/100 --seed 20260809`

## Checkpoint 哈希

`model_sha256.txt`：全部 70 个 .pth 的 SHA256，用于追溯/验证模型完整性。

## 数据位置

- 模型：`local_results/models/`（70 个 .pth）
- 评估 JSON：`local_results/eval/`（68 个文件，含 episodes_detail）
- 服务器原始：`/root/rlproject-swift-improved/saved_models/`、`/root/px4-deploy/`
- 报告：`reports/` 各实验目录

## 诚实声明（评审结论映射）

1. ✅ 深度必要性成立（配对检验 4/5）
2. ⚠️ curriculum 不满足"总体更鲁棒"（clean 显著优于，4/5）
3. ⚠️ 跨场景泛化弱（27-37%），多场景联合部分缓解
4. ⚠️ 架构消融趋势小，需更多验证
5. ⚠️ PX4 对齐仅相机一致化，非真实注册
