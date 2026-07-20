#!/usr/bin/env python3
"""
🕹️ PPO 训练 Microworld — 交互式理解训练场

这个脚本让你在一个最小化的、可控的环境中亲手运行 PPO 的每个步骤。
不是看代码——而是玩代码。

用法:
    python microworld/ppo_playground.py

功能:
    1. 单步模式: 手动执行一次 env.step()，观察 14D 状态和奖励分解
    2. 网络探查: 输入一个 state，观察 ActorCritic 每层的输出
    3. GAE 演示: 输入一个假轨迹，逐步观察 GAE 的计算过程
    4. PPO Update 演示: 在假数据上运行一次完整的 PPO 更新
    5. 超参数调参: 修改超参数并立即看到对训练的影响

设计原则:
    - 每次只展示一个概念
    - 每一步都打印"正在发生什么"和"为什么"
    - 所有中间值可见、可修改
"""

import numpy as np
import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rlproject-swift-improved'))

from drone_env import DroneEnv
from ppo_agent import PPO, ActorCritic, AdaptiveEntropyCoeff


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_step(step_num: int, description: str):
    """打印步骤说明"""
    print(f"\n  📍 Step {step_num}: {description}")


def print_value(label: str, value, detail: str = ""):
    """打印值和解释"""
    if isinstance(value, np.ndarray):
        if value.ndim == 1 and len(value) <= 20:
            val_str = np.array2string(value, precision=3, suppress_small=True)
        else:
            val_str = f"shape={value.shape}, mean={value.mean():.4f}, std={value.std():.4f}"
    elif isinstance(value, float):
        val_str = f"{value:.4f}"
    else:
        val_str = str(value)

    detail_str = f"  ← {detail}" if detail else ""
    print(f"     {label}: {val_str}{detail_str}")


# ═══════════════════════════════════════════════════════════════════════
# 模块 1: 环境探索 — 理解 14D 状态和 7 组件奖励
# ═══════════════════════════════════════════════════════════════════════

def module_1_env_exploration():
    """
    🎮 模块 1: 亲手操作环境

    目标: 理解 DroneEnv 的 14D 状态向量和 7 组件奖励函数
    """
    print_section("模块 1: 环境探索 — 14D 状态 & 7组件奖励")

    env = DroneEnv()
    state, info = env.reset()

    print_step(1, "环境初始化 — 了解状态空间")
    print(f"""
    🏗️ 环境配置:
       - 空间范围: x∈[-10,10], y∈[-10,10], z∈[0,10]
       - 3个静态障碍物 (半径1m):
         · [2.0, 2.0, 3.0]
         · [6.0, 3.0, 5.0]
         · [3.0, 7.0, 4.0]
       - 无人机: 质量1.0, 最大推力10.0, dt=0.05s
       - 最大速度: 5.0 m/s

    📍 目标位置: {info['target_pos']}
    """)

    print_step(2, "14D 状态向量拆解")
    state = state  # shape (14,)
    print(f"""
    ┌─────────────── 14维状态向量 ───────────────┐
    │ 索引 0-2:  位置 [x, y, z]       = {state[0:3]}        │
    │ 索引 3-5:  速度 [vx, vy, vz]    = {state[3:6]}         │
    │ 索引 6-8:  目标方向 [dx, dy, dz] = {state[6:9]}      │
    │ 索引 9:    目标距离              = {state[9]:.3f}               │
    │ 索引 10-12: 障碍物方向 [ox,oy,oz] = {state[10:13]}    │
    │ 索引 13:   障碍物距离            = {state[13]:.3f}              │
    └─────────────────────────────────────────────┘

    💡 设计意图:
       - 位置+速度 = 无人机自身状态（基础运动学）
       - 目标方向+距离 = 导航信号（往哪飞、还有多远）
       - 障碍物方向+距离 = 避障信号（哪里有危险、多危险）
       - 14D 纯向量（无图像）= 信息密集、训练快
    """)

    print_step(3, "执行一个动作并观察奖励分解")

    # 尝试几个不同的动作
    test_actions = [
        ("朝目标方向加速", np.array([0.8, 0.8, 0.5])),
        ("悬停/减速", np.array([0.0, 0.0, 0.0])),
        ("远离目标", np.array([-0.5, -0.5, -0.5])),
    ]

    for action_desc, action in test_actions:
        env_copy = DroneEnv()
        env_copy.reset()
        env_copy.target_pos = info['target_pos']  # 使用相同目标

        next_state, reward, terminated, truncated, step_info = env_copy.step(action)
        comps = step_info['reward_components']

        print(f"\n  🎯 动作: {action_desc} {action}")
        print(f"     总奖励: {reward:.3f}")
        print(f"     ├─ r_dist    (距离引导):   {comps['r_dist']:+.3f}  ← 指数衰减, 越近越积极")
        print(f"     ├─ r_heading (速度方向):   {comps['r_heading']:+.3f}  ← 鼓励朝目标飞")
        print(f"     ├─ r_obs     (障碍物惩罚): {comps['r_obs']:+.3f}  ← 势场式, 越近越负")
        print(f"     ├─ r_smooth  (动作平滑):   {comps['r_smooth']:+.3f}  ← 避免剧烈抖动")
        print(f"     ├─ r_goal    (到达奖励):   {comps['r_goal']:+.3f}  ← 100 + 时间效率加成")
        print(f"     ├─ r_collision(碰撞惩罚):  {comps['r_collision']:+.3f}  ← -50 一次性惩罚")
        print(f"     └─ r_timeout (超时惩罚):   {comps['r_timeout']:+.3f}  ← -10 兜底惩罚")

    print_step(4, "亲自输入动作试试（交互模式）")
    print("   (在实际使用中，这里会进入交互循环)")

    return env, info


# ═══════════════════════════════════════════════════════════════════════
# 模块 2: 网络探查 — 理解 ActorCritic 的每一层
# ═══════════════════════════════════════════════════════════════════════

def module_2_network_exploration():
    """
    🧠 模块 2: 探查神经网络

    目标: 理解 ActorCritic 网络的数据流——从 14D 输入到 3D 动作
    """
    print_section("模块 2: 网络探查 — ActorCritic 数据流")

    # 创建网络
    model = ActorCritic(state_dim=14, action_dim=3, hidden_dim=128)
    model.eval()

    print_step(1, "了解网络结构")
    print(f"""
    ┌─────────────────── ActorCritic 网络 ───────────────────┐
    │                                                        │
    │  输入: state (14,)                                     │
    │    │                                                   │
    │    ▼                                                   │
    │  ┌──────────────────────────┐                          │
    │  │ shared_layer             │                          │
    │  │ Linear(14 → 128) + ReLU  │  ← Actor和Critic共享     │
    │  └──────────────────────────┘                          │
    │    │                                                   │
    │    ├──────────────┬────────────────┐                   │
    │    ▼              ▼                │                   │
    │  ┌──────────┐ ┌──────────┐        │                   │
    │  │Actor 分支│ │Critic 分支│       │                   │
    │  │128→128   │ │128→128   │        │                   │
    │  │+ ReLU    │ │+ ReLU    │        │                   │
    │  │128→3     │ │128→1     │        │                   │
    │  └──────────┘ └──────────┘        │                   │
    │    │              │               │                   │
    │    ▼              ▼               ▼                    │
    │  mean (3,)     value (1,)     log_std (3,)            │
    │  [推力x,y,z]   [状态价值]      [可学习参数]            │
    │    │                                                   │
    │    ▼                                                   │
    │  action ~ Normal(mean, exp(log_std))                   │
    │                                                        │
    └────────────────────────────────────────────────────────┘

    💡 关键设计决策:
       - 共享第一层: 让 Actor 和 Critic 使用相同的特征表示
       - tanh(mean): 将动作均值限制在 (-1, 1)，匹配动作空间
       - 可学习 log_std: 让网络自己学探索程度（而非硬编码）
       - Orthogonal init: 保持梯度范数稳定（Swift 论文实践）
    """)

    print_step(2, "传入一个样本状态，观察每层输出")

    # 创建样本状态（无人机在起点，目标在右上方）
    sample_state = torch.tensor([
        -8.0, -8.0, 1.0,  # 位置: 左下角
        0.0, 0.0, 0.0,    # 速度: 静止
        14.0, 14.0, 6.0,  # 目标方向: 右上
        20.0,              # 目标距离: 20m
        0.5, 0.5, 0.3,    # 障碍物方向
        5.0                # 障碍物距离: 5m
    ], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        # 手动逐步前向传播
        features = model.shared_layer(sample_state)
        actor_hidden = model.actor_layers[0](features)
        actor_hidden_relu = model.actor_layers[1](actor_hidden)
        mean = torch.tanh(model.actor_layers[2](actor_hidden_relu))

        critic_hidden = model.critic_layers[0](features)
        critic_hidden_relu = model.critic_layers[1](critic_hidden)
        value = model.critic_layers[2](critic_hidden_relu)

        std = model.log_std.exp().clamp(min=1e-3, max=1.0)

    print(f"""
    输入 state (14D): 无人机在[-8,-8,1], 目标在[6,6,7]

    ──────────── 前向传播 trace ────────────

    shared_layer: Linear(14→128)+ReLU
      输出 shape: {features.shape}
      输出范围: [{features.min().item():.3f}, {features.max().item():.3f}]
      → 128维共享特征向量

    Actor分支:
      Linear(128→128): 输出范围 [{actor_hidden.min().item():.3f}, {actor_hidden.max().item():.3f}]
      ReLU:            输出范围 [{actor_hidden_relu.min().item():.3f}, {actor_hidden_relu.max().item():.3f}]
      Linear(128→3):   mean = {mean.squeeze().numpy()}  ← 推力 [x, y, z]
      tanh 后范围固定在 (-1, 1)

    Critic分支:
      Linear(128→128): 输出范围 [{critic_hidden.min().item():.3f}, {critic_hidden.max().item():.3f}]
      ReLU:            输出范围 [{critic_hidden_relu.min().item():.3f}, {critic_hidden_relu.max().item():.3f}]
      Linear(128→1):   value = {value.item():.4f}  ← 当前状态的"好坏"估计

    log_std (可学习参数): {model.log_std.data.numpy()}
    std = exp(log_std):   {std.squeeze().numpy()}

    💡 解释:
       - mean 表示网络认为"最优的动作方向"（未经训练时接近随机）
       - value 表示网络对当前状态的估值（正向=好位置，负向=差位置）
       - std 表示探索噪声的大小（每个动作维度独立）
       - 实际动作从 Normal(mean, std) 采样，所以有随机性
    """)

    return model


# ═══════════════════════════════════════════════════════════════════════
# 模块 3: GAE 演示 — 逐行跟踪优势估计
# ═══════════════════════════════════════════════════════════════════════

def module_3_gae_demo():
    """
    📐 模块 3: GAE 逐步演示

    目标: 用一个小轨迹，手工跟踪 GAE 的计算过程
    """
    print_section("模块 3: GAE 优势估计 — 逐行跟踪")

    ppo = PPO(state_dim=14, action_dim=3, num_envs=1)

    print_step(1, "GAE 公式回顾")
    print("""
    GAE (Generalized Advantage Estimation):
      δ_t  = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
      A_t  = δ_t + γ · λ · (1 - done_t) · A_{t+1}

    参数: γ=0.99 (折扣因子), λ=0.95 (偏差-方差权衡)

    💡 直觉:
      - δ_t (TD误差): "这一步的奖励 + 未来价值 — 当前价值"
                      正数 = 比预期好, 负数 = 比预期差
      - A_t (优势):   综合考虑当前和未来所有TD误差
                      λ→0: 只看一步 (低方差, 高偏差)
                      λ→1: 看无限步 (高方差, 低偏差, 等同于MC)
      - λ=0.95:       几乎看全部, 但给远期误差打折
    """)

    print_step(2, "构造一个 4 步的假轨迹, 逐步计算 GAE")

    # 构造假轨迹: 智能体从远处逐渐接近目标
    T = 4
    rewards = np.array([-1.0, -0.5, 0.5, 2.0], dtype=np.float32)
    values = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
    dones = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    next_value = 3.0  # 最终状态的价值估计 (接近目标, 所以估值高)

    print(f"""
    轨迹 (T={T}步):
    ┌──────┬────────┬────────┬──────┐
    │ 步 t │ r_t    │ V(s_t) │ done │
    ├──────┼────────┼────────┼──────┤
    │  0   │ {rewards[0]:5.1f}  │ {values[0]:5.1f}  │  0   │
    │  1   │ {rewards[1]:5.1f}  │ {values[1]:5.1f}  │  0   │
    │  2   │ {rewards[2]:5.1f}  │ {values[2]:5.1f}  │  0   │
    │  3   │ {rewards[3]:5.1f}  │ {values[3]:5.1f}  │  0   │
    └──────┴────────┴────────┴──────┘
    V(s_4) = {next_value} (最终状态估值, 接近目标所以为正)
    """)

    # 手动计算 GAE 并显示每一步
    advantages_manual = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    gamma, lam = 0.99, 0.95

    print("  GAE 反向计算 (从 t=3 到 t=0):\n")

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages_manual[t] = last_gae

        print(f"  t={t}:")
        print(f"    δ_{t} = r_{t} + γ·V(s_{t+1}) - V(s_{t})")
        print(f"         = {rewards[t]:.1f} + {gamma}·{next_val:.1f} - {values[t]:.1f}")
        print(f"         = {delta:.4f}")
        print(f"    A_{t} = δ_{t} + γ·λ·A_{t+1}")
        if t < T - 1:
            print(f"         = {delta:.4f} + {gamma}·{lam}·{advantages_manual[t+1]:.4f}")
        else:
            print(f"         = {delta:.4f} + 0 (最后一步, 无 A_{t+1})")
        print(f"         = {last_gae:.4f}")
        print()

    # 标准化
    adv_mean = advantages_manual.mean()
    adv_std = advantages_manual.std()
    advantages_normalized = (advantages_manual - adv_mean) / (adv_std + 1e-8)

    print(f"  标准化前 advantages: {advantages_manual}")
    print(f"  均值={adv_mean:.4f}, 标准差={adv_std:.4f}")
    print(f"  标准化后 advantages: {advantages_normalized}")
    print(f"\n  💡 标准化使 advantages 均值为0、标准差为1, 降低训练方差")

    # 验证与 compute_gae 一致
    adv_auto, ret_auto = ppo.compute_gae(rewards, values, dones, next_value)
    print(f"\n  ✅ 验证: compute_gae() 输出 = {adv_auto}")
    print(f"     returns = advantages + values = {ret_auto}")


# ═══════════════════════════════════════════════════════════════════════
# 模块 4: PPO Update 演示
# ═══════════════════════════════════════════════════════════════════════

def module_4_ppo_update_demo():
    """
    🔄 模块 4: PPO 更新 — 在假数据上完整运行

    目标: 理解 PPO-Clip 损失函数和完整的更新循环
    """
    print_section("模块 4: PPO 更新 — 完整更新循环")

    ppo = PPO(state_dim=14, action_dim=3, num_envs=1, minibatch_size=4)

    print_step(1, "构造假训练数据 (模拟一次 rollout)")

    # 生成假数据
    T = 8
    states = np.random.randn(T, 14).astype(np.float32) * 2
    actions = np.random.randn(T, 3).astype(np.float32) * 0.5
    rewards = np.random.randn(T).astype(np.float32) * 0.1
    dones = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)

    # 填充 memory
    for i in range(T):
        ppo.store_transition(
            state=states[i], action=actions[i], reward=rewards[i],
            next_state=states[i], done=bool(dones[i]),
            log_prob=0.0, value=0.0, entropy=0.5
        )

    print(f"  采样 {T} 步, 包含 1 个 done 事件")

    print_step(2, "执行 PPO.update() — 观察损失计算")

    print("""
    PPO-Clip 损失函数:
      ratio = exp(log_prob_new - log_prob_old)    ← 新旧策略的概率比
      L_clip = -min(ratio · A, clip(ratio, 0.8, 1.2) · A)

    💡 直觉:
      - ratio > 1: 新策略更倾向这个动作 → 如果A>0(好), 增加概率; 但 clip 限制增长
      - ratio < 1: 新策略更不倾向这个动作 → 如果A<0(差), 减少概率; 但 clip 限制减少
      - clip(0.8, 1.2): 信任域——不允许策略变化太大
      - 取 min: 悲观估计——当 ratio 超出范围时, 不给梯度(被 clip 了)
    """)

    result = ppo.update()

    print_step(3, "更新结果")
    print(f"""
    训练统计:
      total_loss:   {result['total_loss']:.4f}   ← actor + 0.5·critic - α·entropy
      actor_loss:   {result['actor_loss']:.4f}   ← PPO-Clip 损失
      critic_loss:  {result['critic_loss']:.4f}  ← MSE(value, returns)
      entropy:      {result['entropy']:.4f}       ← 策略随机性 (高=探索, 低=利用)
      entropy_coeff:{result['entropy_coeff']:.4f} ← 自适应熵权重

    💡 诊断指南:
      - actor_loss 应该在 0.01-10 范围 (太小 = 没学到, 太大 = 不稳定)
      - critic_loss 应该在 0.1-10 范围 (持续下降 = 价值估计在改善)
      - entropy 应该在 0.5-2.0 范围 (<0.1 = 策略太确定, 可能过早收敛)
      - entropy_coeff: 如果持续上升, 说明策略在"忘记"探索
    """)


# ═══════════════════════════════════════════════════════════════════════
# 模块 5: 超参数调参沙盒
# ═══════════════════════════════════════════════════════════════════════

def module_5_hyperparameter_sandbox():
    """
    🎛️ 模块 5: 超参数调参沙盒

    目标: 修改超参数, 观察对训练动力学的影响
    """
    print_section("模块 5: 超参数调参沙盒")

    print("""
    你可以修改以下超参数, 然后立即看到对训练的影响:

    ┌─────────────────────┬──────────┬──────────────────────────────────┐
    │ 超参数               │ 当前值    │ 调节方向 & 效果                    │
    ├─────────────────────┼──────────┼──────────────────────────────────┤
    │ gamma (折扣因子)      │ 0.99     │ ↑ 更重视远期奖励, 适合长轨迹       │
    │                      │          │ ↓ 更重视近期奖励, 适合短轨迹       │
    │ gae_lambda (GAE λ)   │ 0.95     │ ↑ → 1: 低偏差高方差 (更像MC)      │
    │                      │          │ ↓ → 0: 低方差高偏差 (更像TD(0))   │
    │ clip_eps (PPO裁剪)   │ 0.2      │ ↑ 允许更大的策略更新              │
    │                      │          │ ↓ 更保守, 训练更稳定              │
    │ lr (学习率)           │ 3e-4     │ ↑ 更快学习, 可能不稳定            │
    │                      │          │ ↓ 更稳定, 可能收敛慢              │
    │ epochs (更新轮数)     │ 10       │ ↑ 更充分利用数据, 可能过拟合       │
    │                      │          │ ↓ 更保守, 可能欠拟合              │
    │ minibatch_size       │ 64       │ ↑ 更稳定梯度, 更慢               │
    │                      │          │ ↓ 更快, 更噪声的梯度              │
    │ hidden_dim           │ 128      │ ↑ 更强表达能力, 更多参数          │
    │                      │          │ ↓ 更轻量, 可能欠拟合              │
    │ entropy_coeff        │ 0.01     │ ↑ 更多探索, 更随机               │
    │                      │          │ ↓ 更多利用, 更确定               │
    └─────────────────────┴──────────┴──────────────────────────────────┘

    💡 调参建议:
      - 先固定大部分参数, 一次只改一个
      - 观察 critic_loss: 如果很大(>100), 降低 lr 或增加 minibatch_size
      - 观察 entropy: 如果趋近0, 增加 entropy_coeff 或降低 clip_eps
      - 观察 reward: 如果不增长, 检查奖励函数设计 (回到模块1)
    """)

    # 交互式调参
    print("\n  在实际使用中，这里会进入交互循环，让你:")
    print("    1. 修改超参数")
    print("    2. 在小规模上训练 10 个 episode")
    print("    3. 观察 loss/reward/entropy 曲线")
    print("    4. 对比不同超参数的效果\n")


# ═══════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   🕹️  PPO 训练 Microworld — 交互式理解训练场                              ║
║                                                                          ║
║   这不是一个训练脚本。这是一个"理解加速器"。                                ║
║   每个模块拆开 PPO 的一个组件, 让你亲手运行、修改、观察。                    ║
║                                                                          ║
║   选择模块:                                                                ║
║     1. 环境探索 — 14D状态 & 7组件奖励                                      ║
║     2. 网络探查 — ActorCritic 数据流                                       ║
║     3. GAE演示  — 逐步跟踪优势估计                                          ║
║     4. PPO更新  — 完整更新循环 + 损失函数                                    ║
║     5. 超参数沙盒 — 调参并观察影响                                          ║
║     0. 运行所有模块                                                         ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        choice = input("  请输入模块编号 (0-5): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  退出 Microworld。记住: 理解是累计的, 不是一次性的。👋")
        return

    if choice == '0':
        module_1_env_exploration()
        module_2_network_exploration()
        module_3_gae_demo()
        module_4_ppo_update_demo()
        module_5_hyperparameter_sandbox()
    elif choice == '1':
        module_1_env_exploration()
    elif choice == '2':
        module_2_network_exploration()
    elif choice == '3':
        module_3_gae_demo()
    elif choice == '4':
        module_4_ppo_update_demo()
    elif choice == '5':
        module_5_hyperparameter_sandbox()
    else:
        print(f"  未知选项: {choice}")

    print(f"\n{'='*70}")
    print("  ✅ Microworld 完成")
    print(f"{'='*70}")
    print("""
  📚 下一步:
    - 阅读 docs/knowledge-map.md 建立系统级理解
    - 在真实训练中观察 TensorBoard 指标
    - 修改代码并运行 smoke_test.py 验证理解
    - 向团队讲解你今天学到的概念 (最好的学习方式是教别人)
    """)


if __name__ == "__main__":
    main()
