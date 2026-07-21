#!/usr/bin/env python3
"""
🔍 个人理解审计 — 找出你认知地图上的雾区

这不是测试。没有分数。没有通过/失败。
这是一个"理解探测器"——帮你找到:
  - 哪些概念你真正掌握了（能预测、能解释、能修改）
  - 哪些概念你只是"熟悉"（能看懂，但不能独立复现）
  - 哪些概念是雾区（依赖代码/文档/AI 才能回答）

运行方式:
  python microworld/personal_audit.py

设计原理:
  每道题测试不同的理解层次:
  L1 (语法层): 知道代码在做什么
  L2 (设计层): 知道为什么这样设计
  L3 (系统层): 知道各部分如何相互影响

  你不需要全对。你需要知道哪里不对。
"""

import numpy as np
import torch
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rlproject-swift-improved'))

from drone_env import DroneEnv
from ppo_agent import PPO, ActorCritic, AdaptiveEntropyCoeff, DEVICE


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

LEVEL_COLORS = {
    "L1": "🟡",
    "L2": "🟠",
    "L3": "🔴",
}

RESULTS = {
    "correct": [],
    "partial": [],
    "missed": [],
}

def section(title, level="L1"):
    """打印测试章节"""
    color = LEVEL_COLORS.get(level, "⚪")
    print(f"\n{'─'*65}")
    print(f"  {color} [{level}] {title}")
    print(f"{'─'*65}")

def ask(question, level="L1"):
    """提问并等待用户回答"""
    print(f"\n  ❓ {question}")
    print(f"     (思考后按 Enter 查看答案)")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  ⏹️  审计中断。")
        sys.exit(0)

def reveal(answer_text, is_correct=None):
    """揭示答案"""
    print(f"\n  ✅ 答案:")
    for line in answer_text.strip().split('\n'):
        print(f"     {line}")

    if is_correct is True:
        print(f"\n  🟢 自信度标记: 我完全理解，无需参考代码")
        RESULTS["correct"].append(1)
    elif is_correct is False:
        print(f"\n  🟡 自信度标记: 我部分理解，需要偶尔参考代码")
        RESULTS["partial"].append(1)
    elif is_correct is None:
        print(f"\n  🔴 自信度标记: 我需要加深理解，经常需要查代码")
        RESULTS["missed"].append(1)

    print()
    print(f"  [c] 确认理解  [p] 部分理解  [m] 需要加深")
    try:
        choice = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = 'm'

    if choice == 'c':
        RESULTS["correct"].append(1)
        print("     ✅ 已标记为: 完全理解")
    elif choice == 'p':
        RESULTS["partial"].append(1)
        print("     🟡 已标记为: 部分理解")
    else:
        RESULTS["missed"].append(1)
        print("     🔴 已标记为: 需要加深")

    print()


def verify_with_code(label, code_str):
    """询问用户是否要运行验证代码"""
    print(f"  🧪 可选验证: {label}")
    try:
        choice = input("     运行验证代码? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if choice == 'y':
        print(f"     执行中...")
        try:
            exec(code_str)
        except Exception as e:
            print(f"     ⚠️ 验证代码出错: {e}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# 第一轮: 环境理解 — 14D 状态 & 奖励函数
# ═══════════════════════════════════════════════════════════════════════

def round_1_state_and_reward():
    """测试你对环境的核心理解"""
    print("\n" + "="*65)
    print("  第一轮: 环境理解 — 状态空间 & 奖励函数")
    print("="*65)
    print("  目标: 测试你是否真正理解 14D 状态的每一维和 7 个奖励组件的设计理由")

    # ── Q1: 状态空间 ──
    section("Q1: 14D 状态向量的每一维", "L1")

    ask("""
    不看代码，写出 14D 状态向量的完整结构。
    格式: "索引 0-2: [名称], 索引 3-5: [名称], ..."

    然后回答: 为什么障碍物方向用单位向量而不是原始位置差？
    """)

    reveal("""
    14D 状态向量结构:
      索引 0-2:  位置 [x, y, z]           — 无人机当前位置
      索引 3-5:  速度 [vx, vy, vz]        — 当前速度矢量
      索引 6-8:  目标相对位置 [dx, dy, dz] — target_pos - drone_pos
      索引 9:    目标距离 (标量)            — ||target_pos - drone_pos||
      索引 10-12: 最近障碍物方向 (单位向量) — (closest_obs - drone_pos) / distance
      索引 13:   最近障碍物距离 (标量)      — ||drone_pos - closest_obs|| - radius

    为什么障碍物方向用单位向量？
      → 单位向量只编码方向信息，与距离解耦。
        如果直接用原始位置差，距离信息会和索引9/13重复，
        且数值范围不稳定（近处很小，远处很大）。
        方向+距离分开编码 = 信息解耦，网络更容易学习。

    旧版为什么失败？
      → 旧版用 16x16 深度图像 + 10D 向量。深度图像分辨率极低，
        信息量远不如直接给出障碍物方向和距离。
        外加一个无用的四元数 [1,0,0,0] 占 4 个维度。
    """)

    # ── Q2: 奖励函数设计 ──
    section("Q2: 7 组件奖励函数的设计理由", "L2")

    ask("""
    奖励函数有 7 个组件。对于以下每个组件，回答"为什么这样设计":

    1. r_dist = -5.0 * (1 - exp(-0.3 * d))
       为什么用指数衰减而不是线性 r = -d？

    2. r_obs = -2.0 / (min_dist + 0.5)
       为什么分母加 0.5？不加会怎样？

    3. r_goal = 100 + 50 * remaining_ratio
       为什么加时间效率项 50 * remaining_ratio？
    """)

    reveal("""
    1. 指数衰减 vs 线性:
       线性 r=-d: 所有距离的梯度相同。在远处(如 d=20)梯度也是1，
       但在远处微小的方向调整对到达目标几乎没有影响。
       指数衰减: 梯度 = 5*0.3*exp(-0.3d) = 1.5*exp(-0.3d)
       - d=0  (到达): 梯度≈1.5 (大梯度, 精确引导)
       - d=10 (中途): 梯度≈0.075 (小梯度, 不干扰避障)
       - d=20 (远处): 梯度≈0.004 (几乎无梯度, 让智能体自由探索)
       ✅ 自适应梯度: 越近越精确，越远越自由。

    2. 分母 +0.5 的作用:
       不加 0.5: r_obs = -2.0/min_dist
       - min_dist→0: r_obs→-∞ (梯度爆炸)
       - min_dist→∞: r_obs→0 (正确)
       加 0.5: r_obs = -2.0/(min_dist + 0.5)
       - min_dist→0: r_obs→-4.0 (有界惩罚，不会梯度爆炸)
       - min_dist=0.5: r_obs=-2.0 (安全距离时惩罚适中)
       ✅ 这是势场法的标准技巧——防止靠近障碍物时梯度爆炸。

    3. 时间效率项:
       到达奖励 = 100 + 50 * (剩余步数/500)
       - 第10步到达: r_goal = 100 + 50*(490/500) = 149
       - 第490步到达: r_goal = 100 + 50*(10/500) = 101
       ✅ 鼓励更快到达目标 → 路径更短、更高效。
       如果没有这一项, 智能体可能在目标附近盘旋很久才降落。
    """)

    # ── Q3: 奖励组件权重 ──
    section("Q3: 奖励组件的相对重要性", "L3")

    ask("""
    假设你要把碰撞惩罚从 -50 改成 -10。预测会发生什么:

    1. 训练初期的碰撞率会怎么变化？
    2. 最终的避障行为会变差还是变好？
    3. 这会影响到达目标的速度吗？（间接影响）
    """)

    reveal("""
    预测: 降低碰撞惩罚 → 智能体不再"害怕"障碍物

    1. 训练初期: 碰撞率会上升。因为 r_collision=-10 比 r_obs 的日常惩罚
       (约 -2 到 -4) 大不了太多, 智能体可能觉得"撞一下也无所谓"。

    2. 最终避障行为: 取决于其他信号的强度。
       - 如果 r_obs (障碍物势场) 足够引导避障 → 最终还是会避开
       - 但学习速度会慢很多, 因为坏行为的"学费"便宜了
       - 可能出现: 训练早期频繁碰撞, 但因为碰撞后 episode 终止,
         智能体没机会学会"接近但避开"的精细操作

    3. 间接影响到达速度: 会。
       - 智能体会飞得更"大胆"（更直的路径）
       - 但碰撞率上升 → 更多 episode 提前终止 → 有效训练步数减少
       - 净效果: 可能更快到达（如果运气好），但方差大、不稳定

    系统层洞察:
      奖励函数是一个整体。改一个组件的权重, 会改变所有组件之间的相对重要性。
      r_obs + r_collision 共同构成"安全预算"——智能体在这两者之间权衡。
      降低 r_collision = 安全预算降低 = 更多冒险行为。
    """)


# ═══════════════════════════════════════════════════════════════════════
# 第二轮: PPO 算法理解 — GAE & PPO-Clip
# ═══════════════════════════════════════════════════════════════════════

def round_2_ppo_algorithm():
    """测试你对 PPO 算法核心机制的理解"""
    print("\n" + "="*65)
    print("  第二轮: PPO 算法理解 — GAE & PPO-Clip")
    print("="*65)
    print("  目标: 测试你是否真正理解 GAE 的计算过程和 PPO-Clip 的意图")

    # ── Q4: GAE 手动计算 ──
    section("Q4: GAE 手动计算", "L1")

    ask("""
    给定以下 3 步轨迹:
      r = [0.5, -0.2, 1.0]
      V = [2.0, 1.8, 2.2]
      done = [0, 0, 0]
      V(s_3) = 2.5

    参数: gamma=0.99, lambda=0.95

    请手动计算 δ_2, δ_1, δ_0 和 A_2, A_1, A_0。
    （不需要精确到小数点，能说出计算过程即可）
    """)

    reveal("""
    从后往前计算 (t=2, t=1, t=0):

    t=2:
      δ_2 = r_2 + γ·V(s_3)·(1-done_2) - V_2
          = 1.0 + 0.99×2.5×1 - 2.2
          = 1.0 + 2.475 - 2.2
          = 1.275
      A_2 = δ_2 + γ·λ·(1-done_2)·A_3
          = 1.275 + 0           (最后一步, 无 A_{t+1})
          = 1.275

    t=1:
      δ_1 = r_1 + γ·V_2·(1-done_1) - V_1
          = -0.2 + 0.99×2.2×1 - 1.8
          = -0.2 + 2.178 - 1.8
          = 0.178
      A_1 = δ_1 + γ·λ·(1-done_1)·A_2
          = 0.178 + 0.99×0.95×1×1.275
          = 0.178 + 1.199
          = 1.377

    t=0:
      δ_0 = r_0 + γ·V_1·(1-done_0) - V_0
          = 0.5 + 0.99×1.8×1 - 2.0
          = 0.5 + 1.782 - 2.0
          = 0.282
      A_0 = δ_0 + γ·λ·(1-done_0)·A_1
          = 0.282 + 0.99×0.95×1×1.377
          = 0.282 + 1.295
          = 1.577

    标准化:
      mean(A) = (1.275+1.377+1.577)/3 = 1.410
      std(A)  ≈ 0.125
      A_norm  ≈ [-1.08, -0.26, +1.34]

    关键洞察:
      - 虽然 t=1 的奖励是负的(-0.2), 但因为后续 t=2 有大的正奖励,
        GAE 给 t=1 也分配了正的优势(1.377)
      - 这就是 GAE 的"信用分配"能力——当前步的好处可能在未来才体现
      - λ=0.95 意味着远期信号打折很少(0.95^3≈0.86), 几乎是 MC 估计
    """)

    verify_with_code("用 ppo_agent.py 的 compute_gae 验证", """
r = np.array([0.5, -0.2, 1.0], dtype=np.float32)
v = np.array([2.0, 1.8, 2.2], dtype=np.float32)
d = np.array([0.0, 0.0, 0.0], dtype=np.float32)
nv = 2.5
ppo = PPO(state_dim=14, action_dim=3, num_envs=1)
adv, ret = ppo.compute_gae(r, v, d, nv)
print(f"GAE advantages: {adv}")
print(f"GAE returns:    {ret}")
print(f"Expected adv:   [1.577, 1.377, 1.275] (before normalization)")
    """)

    # ── Q5: GAE 的 per-environment 陷阱 ──
    section("Q5: GAE 的 per-environment 计算 — 最著名的 bug", "L2")

    ask("""
    这个项目曾有一个关键 bug: GAE 在 8 个并行环境下跨环境边界计算。

    1. 为什么在并行环境下 GAE 必须按环境分别计算？
    2. 如果错误地将所有 transition 当作单一连续轨迹,
       优势估计会偏高还是偏低？为什么？
    """)

    reveal("""
    1. 为什么必须 per-environment 计算:

       8 个并行环境的 transition 存储顺序:
         [env0_s0, env1_s0, ..., env7_s0, env0_s1, env1_s1, ...]

       如果当作单一轨迹:
         GAE 会在 env7_s0 → env0_s1 之间传播优势
         但 env7_s0 和 env0_s1 是完全独立的两个环境！
         env7_s0 的下一个状态是 env7_s1，不是 env0_s1。

       正确做法:
         reshape 为 (rollout_steps, num_envs) → 转置为 (num_envs, rollout_steps)
         → 每个环境独立计算 GAE → 再 flatten 回来

    2. 跨环境 GAE 会偏高还是偏低？

       偏高（更乐观）。
       原因: 不同环境的轨迹不相关。环境 A 的奖励和环境 B 的价值估计
       之间没有因果关系。但 GAE 公式会把它们的 TD 误差串联起来。
       因为 GAE 是累加的（每个 δ 加上 γλ·上一个 A），
       无关的误差会随机叠加 → 方差极大 → 可能偏高也可能偏低。
       在这个项目中观察到的是 Critic loss 飙升到 4000-13000
       （正常应该 300-450），说明价值估计完全错误。

    这个 bug 的代价: 5000 episodes, 0% 成功率。
    修复后: 100 episodes, 94% 成功率。

    教训: 并行计算的正确性取决于对数据布局的深刻理解。
         代码可以"看起来对"且不报错，但数学上完全错误。
    """)

    # ── Q6: PPO-Clip 的直觉 ──
    section("Q6: PPO-Clip 的直觉理解", "L2")

    ask("""
    PPO-Clip 损失函数:
      ratio = exp(log_prob_new - log_prob_old)
      L = -min(ratio·A, clip(ratio, 0.8, 1.2)·A)

    1. 当 A > 0（这个动作比预期好），ratio 从 1.0 升到 1.5 时，
       PPO-Clip 会怎么做？为什么？

    2. 为什么取 min 而不是 max？
       （提示: 想想"悲观估计"）

    3. 如果 clip_eps=0（即 clip 到 [1.0, 1.0]），会发生什么？
    """)

    reveal("""
    1. A>0, ratio=1.0→1.5:
       ratio·A = 1.5A (因为 ratio 增大)
       clip(ratio)·A = 1.2A (被 clip 限制了)
       min(1.5A, 1.2A) = 1.2A
       → PPO 取 1.2A，不给 ratio=1.5 的额外"奖励"
       → 防止策略因为一个好的动作而剧烈改变

       直觉: "你做得好, 我给你加分。但加分的上限是 1.2 倍。
              不要因为一次成功就彻底改变自己。"

    2. 为什么取 min 而不是 max:
       取 max 会是"乐观估计"——总是取 ratio 更大的一方。
       这会导致:
       - A>0 时, ratio 越大越好 → 鼓励策略剧烈改变
       - A<0 时, ratio 越小越好 → 也鼓励策略剧烈改变
       → 信任域约束失效, 策略更新过大, 训练崩溃

       取 min 是"悲观估计":
       - A>0 时, ratio 被上限 clip → 不奖励过度自信
       - A<0 时, ratio 被下限 clip → 不惩罚过度保守
       → 保守更新, 训练稳定

    3. clip_eps=0 (clip 到 [1.0, 1.0]):
       ratio 被强制为 1.0 → log_prob_new = log_prob_old
       → 策略完全不变
       → 训练什么都不学
       这是"最安全"但"最无用"的设置。

    clip_eps 的权衡:
      - 太小 (0.05): 训练极其稳定，但学得极慢
      - 太大 (0.5): 训练快，但可能不稳定/崩溃
      - 0.2: 经验上的 sweet spot（来自原始 PPO 论文）
    """)


# ═══════════════════════════════════════════════════════════════════════
# 第三轮: 系统级理解 — 各部分如何相互影响
# ═══════════════════════════════════════════════════════════════════════

def round_3_system_thinking():
    """测试系统级理解"""
    print("\n" + "="*65)
    print("  第三轮: 系统级理解 — 模块间的相互影响")
    print("="*65)
    print("  目标: 测试你是否理解一个改动会如何传播到整个系统")

    # ── Q7: 网络架构变更的连锁反应 ──
    section("Q7: 如果把 hidden_dim 从 128 改成 256", "L3")

    ask("""
    假设你把 ActorCritic 的 hidden_dim 从 128 改为 256。

    不要只说"模型变大了"。请预测:
    1. 训练速度会怎么变化？（具体数字: 目前 ~800 步/秒）
    2. 需要调整哪些其他超参数？为什么？
    3. rollout_steps=2048 还合适吗？
    4. 最终性能（成功率）一定会提升吗？
    """)

    reveal("""
    1. 训练速度:
       参数量: 128→256, 约 4 倍参数。
       目前 128 版本 ~3.5万参数; 256 版本 ~10万参数。
       训练速度从 ~800 步/秒可能降到 ~500-600 步/秒 (非线性的原因
       是 batch 计算中矩阵乘法的 GPU 利用率更高)。
       但如果 GPU 显存充足, 降幅可能更小 (~650 步/秒)。

    2. 需要调整的超参数:
       - 学习率 lr: 更大网络 → 更复杂损失面 → 可能需要更小的 lr (如 1e-4)
       - minibatch_size: 更大网络需要更稳定的梯度 → 增大到 128 或 256
       - epochs: 更大网络容易过拟合 → 可能需要减少到 5-8
       - entropy_coeff: 更大网络表达能力更强 → 可能更早收敛 → 需要更多探索

    3. rollout_steps=2048 合适吗？
       更大网络需要更多数据来训练（更多参数 = 需要更多样本来估计梯度）。
       2048 步可能不够 → 考虑增加到 4096。
       但这又会让每次更新变慢。需要在"数据量"和"更新频率"之间权衡。

    4. 最终性能一定会提升吗？
       不一定。甚至可能下降。
       - 这个任务是 14D→3D 的连续控制, 相对简单
       - 128 网络已经达到 98% 成功率
       - 256 网络可能过拟合（记住训练环境而非学到泛化策略）
       - 过拟合表现: 训练成功率很高但评估时遇到新目标位置就失败
       - 更大的网络还更容易陷入局部最优

       系统层洞察:
       "更大的模型"不等于"更好的模型"。
       网络容量需要与任务复杂度匹配。当前 128 网络对 3 个静态障碍物
       的任务来说可能已经足够甚至偏大。
    """)

    # ── Q8: 自适应熵的动力学 ──
    section("Q8: 自适应熵系数的训练动力学", "L2")

    ask("""
    自适应熵系数 (AdaptiveEntropyCoeff) 的 target_entropy 设为 -3 (= -action_dim)。

    1. 为什么 target_entropy 是 -action_dim，而不是其他值？

    2. 如果训练中 entropy 始终 > 0.5 且 entropy_coeff 持续上升，
       这说明什么？应该怎么调整？

    3. 如果 entropy 在训练 50 轮后就降到 0.01 且不再上升，
       这说明什么？后果是什么？
    """)

    reveal("""
    1. 为什么 target_entropy = -action_dim = -3:
       这是 SAC (Soft Actor-Critic) 论文中的启发式。
       对于 3 维连续动作空间 (每个动作独立, 服从高斯分布):
       - 高斯分布的熵 = 0.5 * log(2πe * σ²) 每维度
       - 3 维总熵 ≈ 3 * 0.5 * log(2πe * σ²)
       - 当 σ=1 时, 单维熵 ≈ 1.42, 三维 ≈ 4.2
       - 我们不想要太大 (纯随机) 也不想要太小 (无探索)
       - target_entropy = -3 给出一个合理的探索下限
       - 负值作为"下限"是合理的, 因为策略可以学得很确定 (低熵)
       - 这是经验规则, 不是严格的数学推导

    2. entropy > 0.5 且 entropy_coeff 持续上升:
       说明: 策略的随机性高于目标 (target_entropy=-3对应的熵水平),
       自适应机制在增加熵系数 → 试图"压制"策略的随机性。
       但这可能不是坏事: 如果 reward 还在增长, 说明探索是有效的。
       如果 reward 停滞且 entropy 高 → 策略没有收敛, 可能需要:
       - 降低 entropy_coeff 初始值
       - 或接受高 entropy 意味着任务需要持续探索

    3. entropy 降到 0.01 后不再上升:
       说明: 策略几乎变成确定性策略 (std≈0)。
       后果:
       - 好的情况: 策略已经学好了, 确定性执行是最优的
       - 坏的情况: 策略过早收敛到次优解, 不再探索
       判断方法: 看 success_rate
       - 如果 success_rate 很高且稳定 → 好的收敛
       - 如果 success_rate 中等且不再提升 → 过早收敛 (需要更多探索)
       修复方法:
       - 增大 entropy_coeff 初始值 (如 0.05)
       - 增大 target_entropy (如 -1 或 0)
       - 在训练后期手动注入噪声

    实践中: 当前项目的 entropy_coeff 从 0.01 缓慢增长到 0.0106,
    说明策略在"可控地探索"——这是健康的训练信号。
    """)

    # ── Q9: 数据流完整追踪 ──
    section("Q9: 完整数据流 — 从 env.reset() 到 ppo.update()", "L3")

    ask("""
    在不看代码的情况下, 描述一次完整的训练迭代的数据流:

    从 8 个环境同时 reset() 开始,
    到 ppo.update() 返回 loss 字典结束。

    列出中间每一步的数据形状变化（tensor shapes）。
    特别标注: 在哪一步发生了 reshape/transpose, 以及为什么。
    """)

    reveal("""
    完整数据流 (一个 rollout 循环):

    ┌─ 阶段 1: 采样 ─────────────────────────────────────┐

    1. states = env.reset() → (8, 14)
       8 个环境, 每个返回 14D 状态

    2. for step in range(256):  # rollout_steps / num_envs
          for i in range(8):
              action, log_prob, value, entropy = ppo.get_action(states[i])
              # states[i]=(14,) → mean,std,value → action=(3,)
          actions = np.array([...])  → (8, 3)
          next_states, rewards, dones, infos = env.step(actions)
              # next_states=(8,14), rewards=(8,), dones=(8,)
          for i in range(8):
              ppo.store_transition(state, action, reward, ...)
              # 存入 ppo.memory 列表

      循环结束后: ppo.memory 有 8×256=2048 条记录

    ┌─ 阶段 2: GAE 计算 ─────────────────────────────────┐

    3. 从 memory 提取数组:
       states = (2048, 14)
       actions = (2048, 3)
       rewards = (2048,)      ← 关键: 1D 展平
       values = (2048,)       ← 关键: 1D 展平
       dones = (2048,)        ← 关键: 1D 展平

    4. reshape + transpose:
       rewards.reshape(256, 8).T   → (8, 256)
                ↑          ↑
          rollout_steps  num_envs
       目的: 将按 [env0_s0, env1_s0, ..., env7_s0, env0_s1, ...]
       排列的数据, 转置为每个环境独立的轨迹

    5. for env_idx in range(8):
           compute_gae(rewards[env_idx], values[env_idx],
                       dones[env_idx], next_value)
           # 每个环境独立计算: (256,) → advantages(256,), returns(256,)

    6. flatten 回来:
       advantages = all_advantages.T.flatten()  → (2048,)
       returns    = all_returns.T.flatten()     → (2048,)

    7. 标准化 advantages:
       advantages = (adv - mean) / (std + 1e-8)  → (2048,)

    ┌─ 阶段 3: PPO 更新 ─────────────────────────────────┐

    8. for epoch in range(10):
           indices = randperm(2048)  # shuffle
           for i in range(32):  # 2048/64 = 32 minibatches
               batch = indices[i*64 : (i+1)*64]  # (64,)

               mean, std, value = model(batch_states)  # (64,14)→(64,3),(64,3),(64,)

               ratio = exp(log_prob_new - log_prob_old)  # (64,)
               L_clip = -min(ratio·A, clip(ratio)·A).mean()
               L_critic = MSE(value, returns)
               L = L_clip + 0.5*L_critic - α*entropy.mean()

               backward() → clip_grad_norm(0.5) → step()

    9. AdaptiveEntropyCoeff.update(entropy)
       根据当前平均熵调整 α

    ┌─ 关键形状变化总结 ─────────────────────────────────┐

    (8,14) → [(3,) × 8] → store → (2048,14)
    (2048,) → reshape → (256,8) → .T → (8,256)
    → per-env GAE → (8,256) → .T.flatten() → (2048,)
    → minibatch(64,) → model → loss → backward

    最关键的 reshape/transpose: 步骤 4
    这是 bug 的发源地。错误做法: 不 transpose,
    直接将 (2048,) 当作单一轨迹输入 compute_gae。
    """)


# ═══════════════════════════════════════════════════════════════════════
# 第四轮: 预测实验 — 先猜后验证
# ═══════════════════════════════════════════════════════════════════════

def round_4_prediction_experiments():
    """预测实验结果"""
    print("\n" + "="*65)
    print("  第四轮: 预测实验 — 先猜后验证")
    print("="*65)
    print("  目标: 先做出预测，再运行代码验证。差距 = 你的理解盲区")

    # ── 实验 1: 网络输出范围 ──
    section("实验 1: 预测 ActorCritic 的输出范围", "L2")

    ask("""
    创建一个未训练的 ActorCritic(state_dim=14, action_dim=3, hidden_dim=128)。

    给定 state = [0,0,0, 0,0,0, 0,0,0, 0, 0,0,0, 0] (全零向量),
    预测:
    1. mean (动作均值) 的每个元素大概在什么范围？
       A) [-1, 1]   B) [-0.5, 0.5]   C) [-0.1, 0.1]   D) 接近 0

    2. value (状态价值) 大概在什么范围？
       A) 接近 0   B) [-1, 1]   C) [-10, 10]   D) 不确定, 取决于初始化

    3. std (标准差) 的每个元素大概是多少？
       A) 接近 0   B) 接近 1   C) 接近 0.1   D) 不确定
    """)

    reveal("""
    1. mean 的范围: D) 接近 0

       原因: tanh(actor_output), 而 actor_output 最后一层的
       orthogonal init gain=0.01, 所以 actor_output ≈ 很小的值,
       tanh(小值) ≈ 小值 ≈ 接近 0。
       全零输入时, 所有层的 bias 初始化为 0, 所以输出也接近 0。

    2. value 的范围: A) 接近 0

       原因: critic 最后一层的 orthogonal init gain=1.0,
       但 bias=0 + 全零输入 → 所有层的加权和为 0 → value≈0。

    3. std 的范围: B) 接近 1

       原因: log_std 初始化为 torch.zeros(3),
       std = exp(0) = 1.0。
       clamp(min=1e-3, max=1.0) 不影响, 因为 1.0 在范围内。

    结论: 未训练网络对全零输入返回:
      mean≈[0,0,0], std=[1,1,1], value≈0
    即"我不知道该做什么, 所以随机探索, 不认为当前位置好坏"。
    这是合理的初始行为。
    """)

    verify_with_code("创建网络并验证预测", """
model = ActorCritic(14, 3, 128)
model.eval()
x = torch.zeros(1, 14)
with torch.no_grad():
    m, s, v = model(x)
print(f"mean  = {m.squeeze().numpy()}")
print(f"std   = {s.squeeze().numpy()}")
print(f"value = {v.item():.4f}")
    """)

    # ── 实验 2: 奖励函数对动作的敏感性 ──
    section("实验 2: 奖励函数对动作方向的敏感性", "L2")

    ask("""
    无人机从 [-8,-8,1] 出发, 目标是 [6,6,7] (右上方约 20m 外)。

    预测以下 3 个动作的奖励大小顺序 (哪个最大, 哪个最小):
    A) action = [0.8, 0.8, 0.5]  (朝目标方向加速)
    B) action = [0.0, 0.0, 0.0]  (悬停不动)
    C) action = [-0.5, -0.5, -0.5] (反方向)

    注意: 只比较第一步的奖励 (r_dist + r_heading + r_obs + r_smooth),
    不考虑 r_goal/r_collision/r_timeout (第一步不会触发)。
    """)

    reveal("""
    预测顺序: A > B > C

    分析:
    A) 朝目标加速: r_heading 为正 (速度方向与目标方向一致)
       r_dist 取决于有没有接近目标
    B) 悬停: r_heading=0 (速度为0), r_smooth 正常
       r_dist 不变 (位置没变)
    C) 反方向: r_heading 为负 (速度方向与目标方向相反) → 总奖励最差

    具体数值 (近似):
    A: r_dist≈-4.9 + r_heading≈+0.8 + r_obs≈-0.4 + r_smooth=0 ≈ -4.5
    B: r_dist≈-4.9 + r_heading=0    + r_obs≈-0.4 + r_smooth=0 ≈ -5.3
    C: r_dist≈-5.0 + r_heading≈-0.8 + r_obs≈-0.4 + r_smooth=-0.4 ≈ -6.6

    但实际数字需要运行才知道, 因为:
    - r_heading = speed * cos(angle) * 2.0
    - 第一步 speed 取决于 thrust/mass * dt
    - obstacle distance 会影响 r_obs

    重要: 我们只需要预测顺序, 不需要精确数值。
    这就是"心智模型"的作用——定性理解 > 定量记忆。
    """)

    verify_with_code("运行环境验证奖励", """
env = DroneEnv()
state, info = env.reset()
# 手动设置起点和目标
env.state = np.array([-8.0, -8.0, 1.0, 0, 0, 0], dtype=np.float32)
env.target_pos = np.array([6.0, 6.0, 7.0])
env._prev_action = None

for desc, act in [("朝目标 [0.8,0.8,0.5]", np.array([0.8,0.8,0.5])),
                   ("悬停 [0,0,0]", np.array([0.0,0.0,0.0])),
                   ("反方向 [-0.5,-0.5,-0.5]", np.array([-0.5,-0.5,-0.5]))]:
    env2 = DroneEnv()
    env2.state = env.state.copy()
    env2.target_pos = env.target_pos.copy()
    env2._prev_action = None
    ns, r, term, trunc, info2 = env2.step(act)
    c = info2['reward_components']
    print(f"{desc}: total={r:.3f} | dist={c['r_dist']:+.3f} heading={c['r_heading']:+.3f} obs={c['r_obs']:+.3f} smooth={c['r_smooth']:+.3f}")
    """)

    # ── 实验 3: 预测 GAE bug 的影响 ──
    section("实验 3: 模拟 GAE 跨环境 bug", "L3")

    ask("""
    假设我们有 2 个环境, 各自 2 步:

    Env 0: r=[1.0, 1.0], V=[10.0, 10.0], done=[0, 0], V_next=10.0
    Env 1: r=[-1.0, -1.0], V=[-10.0, -10.0], done=[0, 0], V_next=-10.0

    存储顺序: [env0_s0, env1_s0, env0_s1, env1_s1]

    1. 正确做法 (per-env): Env0 的 A_1 和 Env1 的 A_1 分别是什么？
    2. 错误做法 (单轨迹): 将 4 步当作一个轨迹, A_3 (env1_s1) 是什么？
    3. 两种做法的结果一样吗？如果不一样, 差在哪？
    """)

    reveal("""
    1. 正确做法 (per-env):

    Env 0 (r=[1,1], V=[10,10], V_next=10):
       δ_1 = 1.0 + 0.99×10 - 10 = 1.0 + 9.9 - 10 = 0.9
       A_1 = 0.9  (最后一步)
       δ_0 = 1.0 + 0.99×10 - 10 = 0.9
       A_0 = 0.9 + 0.99×0.95×0.9 = 0.9 + 0.847 = 1.747

    总 advantages: [1.747, 0.9]

    Env 1 (r=[-1,-1], V=[-10,-10], V_next=-10):
       δ_1 = -1.0 + 0.99×(-10) - (-10) = -1.0 - 9.9 + 10 = -0.9
       A_1 = -0.9
       δ_0 = -1.0 + 0.99×(-10) - (-10) = -0.9
       A_0 = -0.9 + 0.99×0.95×(-0.9) = -0.9 - 0.847 = -1.747

    总 advantages: [-1.747, -0.9]

    2. 错误做法 (4步单轨迹):
       轨迹: r=[1, -1, 1, -1], V=[10, -10, 10, -10], V_next=-10

       t=3: δ_3 = -1 + 0.99×(-10) - (-10) = -1 - 9.9 + 10 = -0.9
            A_3 = -0.9 + 0.99×0.95×0 = -0.9

       t=2: δ_2 = 1 + 0.99×(-10) - 10 = 1 - 9.9 - 10 = -18.9
            A_2 = -18.9 + 0.99×0.95×(-0.9) = -18.9 - 0.847 = -19.747

       t=1: δ_1 = -1 + 0.99×10 - (-10) = -1 + 9.9 + 10 = 18.9
            A_1 = 18.9 + 0.99×0.95×(-19.747) = 18.9 - 18.57 = 0.33

       t=0: δ_0 = 1 + 0.99×(-10) - 10 = 1 - 9.9 - 10 = -18.9
            A_0 = -18.9 + 0.99×0.95×0.33 = -18.9 + 0.31 = -18.59

    3. 对比:

    正确:   Env0=[1.747, 0.9],  Env1=[-1.747, -0.9]
    错误:   [A0=-18.59, A1=0.33, A2=-19.747, A3=-0.9]

    结果完全不一样。
    - 正确的 Env0 A_0=+1.747 (这个动作很好)
    - 错误的 A_0=-18.59 (这个动作极差)

    优势估计的符号都反了！

    为什么: 在错误做法中, t=2 (env0_s1) 的 V=-10 被用来计算 δ_2,
    但 env0_s1 的正确价值应该是 +10。环境 1 的价值 (-10) "污染"了
    环境 0 的 GAE 计算, 导致整个链的符号错乱。

    这就是为什么 Critic loss 会从 300 飙升到 4000-13000。
    Critic 被迫去拟合完全错误的 returns 估计。
    """)


# ═══════════════════════════════════════════════════════════════════════
# 第五轮: 代码中的"为什么"——设计决策审计
# ═══════════════════════════════════════════════════════════════════════

def round_5_design_rationale():
    """测试对设计决策的理解"""
    print("\n" + "="*65)
    print("  第五轮: 设计决策审计 — 你能解释每一个 WHY 吗")
    print("="*65)
    print("  目标: 检查你能否独立解释关键设计决策的理由")

    decisions = [
        ("为什么 Actor 和 Critic 共享第一层？",
         "两个原因: (1) 特征复用——状态的良好表示对策略和价值估计都有用;"
         "(2) 参数效率——共享层减少了约30%参数。Swift论文的经验是共享层"
         "不会损害性能，反而通过特征共享加速了学习。但也有风险: 如果"
         "Actor和Critic需要不同特征, 共享会限制表达能力。对这个简单任务, 共享是净收益。"),

        ("为什么用 Orthogonal 初始化而不是 Kaiming/Xavier？",
         "Orthogonal初始化在深度RL中更受欢迎(特别是PPO系), 因为:\n"
         "1. 正交矩阵保持梯度范数——在多层MLP中, 前向和反向传播的\n"
         "   范数不会爆炸或消失\n"
         "2. Swift论文使用它(遵循了RL社区的实践)\n"
         "3. gain=√2 给ReLU激活函数提供适当的缩放\n"
         "4. actor输出gain=0.01——确保初始策略接近零动作(安全默认)\n"
         "5. critic输出gain=1.0——让价值估计有足够的动态范围\n\n"
         "Xavier/Kaiming在监督学习中更常见, 它们的假设(批量归一化、"
         "短网络)在RL的小MLP中不一定成立。"),

        ("为什么动作输出要经过 tanh？",
         "动作空间是[-1, 1]^3。tanh的自然输出范围就是(-1,1)。\n\n"
         "不使用tanh的替代方案:\n"
         "- 直接输出无界值 + clip: 梯度在边界处为0(被clip截断)\n"
         "- sigmoid*2-1: 饱和问题比tanh更严重\n\n"
         "tanh的优势:\n"
         "- 中心对称(输出均值自然为0)\n"
         "- 在0附近近似线性, 远离0时平滑饱和\n"
         "- 饱和时梯度小但不为0(不像clip完全截断)\n\n"
         "一个微妙的点: tanh后.mean()≈0是好的初始策略——"
         "'不做什么'比'乱动'更安全。"),

        ("为什么 log_std 是可学习参数而不是网络输出？",
         "两种方式各有优劣:\n\n"
         "可学习参数(当前):\n"
         "+ 状态无关的探索——所有状态下探索程度相同\n"
         "+ 简单, 参数少(只有action_dim个)\n"
         "+ 训练稳定——探索程度平滑变化\n"
         "- 不能根据状态调整探索(危险区域不更小心)\n\n"
         "网络输出:\n"
         "+ 状态相关的探索——不确定时多探索\n"
         "- 更复杂, 可能不稳定\n"
         "- 在某些RL任务中, 网络学会把std推向0来'作弊'\n"
         "(确定性策略短期reward更高,但长期卡住)\n\n"
         "对于这个任务, 状态无关的std足够, 因为环境没有\n"
         "'需要更谨慎'的区域和'可以更大胆'的区域之分。"),

        ("为什么 PPO update 循环中每一步都 shuffle？",
         "打破数据的时间相关性。\n\n"
         "rollout数据是按时间顺序收集的:\n"
         "  第1步→第2步→...→第256步\n"
         "相邻步骤的状态高度相关(位置只差一点点)。\n\n"
         "如果按顺序训练:\n"
         "- minibatch 1: 全是第1-64步 → 过拟合早期状态\n"
         "- minibatch 32: 全是第1985-2048步 → 过拟合晚期状态\n"
         "- 10个epoch下来, 网络在32种不同状态分布上来回摇摆\n\n"
         "shuffle每个epoch打乱顺序, 确保每个minibatch包含\n"
         "来自不同时间步的样本 → 训练更稳定。"),
    ]

    for question, answer in decisions:
        section(question, "L2")
        ask("先自己回答, 再看答案。")
        reveal(answer)


# ═══════════════════════════════════════════════════════════════════════
# 审计报告
# ═══════════════════════════════════════════════════════════════════════

def generate_report():
    """生成个人理解审计报告"""
    print("\n\n")
    print("="*65)
    print("  📊 个人理解审计报告")
    print("="*65)

    total = len(RESULTS["correct"]) + len(RESULTS["partial"]) + len(RESULTS["missed"])
    if total == 0:
        print("\n  没有数据。请先完成审计。")
        return

    pct_correct = len(RESULTS["correct"]) / total * 100
    pct_partial = len(RESULTS["partial"]) / total * 100
    pct_missed = len(RESULTS["missed"]) / total * 100

    print(f"""
  总题数: {total}

  🟢 完全理解: {len(RESULTS['correct'])} 题 ({pct_correct:.0f}%)
     → 这些概念你已经内化了，可以教给别人。

  🟡 部分理解: {len(RESULTS['partial'])} 题 ({pct_partial:.0f}%)
     → 你理解大致方向，但细节需要查代码确认。
     → 建议: 在 Microworld 中亲手运行验证这些概念。

  🔴 需要加深: {len(RESULTS['missed'])} 题 ({pct_missed:.0f}%)
     → 这是你的"雾区"。优先投入时间在这里。
     → 建议: 手写一遍关键算法的伪代码, 不看源码。
""")

    # 生成雾图
    print("  ┌─────────────────────────────────────────────┐")
    print("  │          你的个人理解雾图                      │")

    bar_width = 40
    correct_bar = "🟢" * int(pct_correct / 100 * bar_width)
    partial_bar = "🟡" * int(pct_partial / 100 * bar_width)
    missed_bar = "🔴" * int(pct_missed / 100 * bar_width)
    rest = bar_width - len(correct_bar) - len(partial_bar) - len(missed_bar)

    print(f"  │  {correct_bar}{partial_bar}{missed_bar}{' ' * max(0, rest)} │")
    print(f"  │  理解: {pct_correct:.0f}%  熟悉: {pct_partial:.0f}%  雾区: {pct_missed:.0f}%         │")
    print("  └─────────────────────────────────────────────┘")

    # 建议
    print(f"""
  📋 下一步建议:

  {'1. 优先消除红色区域' if pct_missed > 20 else '1. 巩固绿色区域'}
     → {'在 microworld/ppo_playground.py 中运行相关模块' if pct_missed > 20 else '尝试教给一个假想的同事'}
     → {'重新阅读 improvement_report.md 中对应章节' if pct_missed > 20 else '挑战自己: 能否不看代码画架构图？'}

  2. 加深系统级理解 (L3)
     → 运行 python microworld/ppo_playground.py
     → 选择你不熟悉的模块, 修改参数看效果

  3. 验证你的心智模型
     → 做一个小实验: 改一个超参数, 预测结果, 跑10个episode验证
     → 预测准确率 = 你真正的理解度

  4. 教给别人
     → 向一个假想的同事解释 PPO 的完整数据流
     → 如果中间卡住了——那里就是你的理解盲区
""")


# ═══════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🔍 个人理解审计 — Personal Understanding Audit              ║
║                                                              ║
║  这不是测试。没有分数。                                       ║
║  这是一个"雾区探测器"——帮你找到认知地图上的盲点。              ║
║                                                              ║
║  5 轮审计:                                                    ║
║    第一轮: 环境理解 (14D状态 + 7组件奖励)       [L1-L3]      ║
║    第二轮: PPO算法理解 (GAE + PPO-Clip)         [L1-L2]      ║
║    第三轮: 系统级理解 (模块间相互影响)           [L3]         ║
║    第四轮: 预测实验 (先猜后验证)                 [L2-L3]      ║
║    第五轮: 设计决策审计 (每个 WHY)               [L2]         ║
║                                                              ║
║  每道题: 先自己思考 → 按 Enter 看答案 → 标记你的自信度        ║
║           🟢 完全理解  🟡 部分理解  🔴 需要加深               ║
║                                                              ║
║  诚实回答。唯一会骗你的人是你自己。                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    try:
        input("  按 Enter 开始审计...")
    except (EOFError, KeyboardInterrupt):
        print("\n\n  退出。")
        return

    # 执行各轮
    round_1_state_and_reward()
    round_2_ppo_algorithm()
    round_3_system_thinking()
    round_4_prediction_experiments()
    round_5_design_rationale()

    # 生成报告
    generate_report()

    print("="*65)
    print("  ✅ 审计完成")
    print("="*65)
    print("""
  理解不是一个二进制状态（懂/不懂）。
  理解是一个光谱:
    完全陌生 → 能看懂 → 能修改 → 能预测 → 能重构 → 能教别人

  你现在在哪一级？
  针对 🔴 区域，建议:
    1. 在 Microworld 中亲手运行对应模块
    2. 手写伪代码, 不看源码
    3. 尝试向假想的同事解释

  一周后再运行一次审计, 观察雾图的变化。
    """)


if __name__ == "__main__":
    main()
