"""
Week 1 Benchmark & Oracle — UAV RL 3D Path Planning
=====================================================

Benchmarks:
  1. Baseline 复现成功率 (96-100%) + 碰撞率 (≤4%)
  2. 噪声环境训练管线完整性 (100轮, σ_pos=0.5, 无崩溃)
  3. 噪声环境观测差异量化 (3 timesteps clean vs noisy)

Oracles:
  4. 完美信息 Oracle (无噪声baseline表现)
  5. 随机策略 Oracle (下界)
  6. A* Oracle (理论最优路径下界)

Usage:
  python week1_benchmark.py                     # 全部运行
  python week1_benchmark.py --skip-training     # 跳过耗时训练
  python week1_benchmark.py --only-oracles      # 仅运行Oracle
"""

import numpy as np
import torch
import os
import sys
import json
import time
import heapq
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone_env import DroneEnv
from drone_env_noisy import NoisyDroneEnv, NOISE_PATTERN_DIMS
from ppo_agent import PPO, DEVICE

# ============================================================
# 路径常量
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'ppo_swift_3000ep_20260712_115059')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'eval_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 环境参数 (与 drone_env.py 保持一致)
SPACE_BOUNDS = (np.array([-10.0, -10.0, 0.0]), np.array([10.0, 10.0, 10.0]))
TARGET_MIN = np.array([5.0, 5.0, 2.0])
TARGET_MAX = np.array([8.0, 8.0, 8.0])
OBSTACLES = np.array([[2.0, 2.0, 3.0], [6.0, 3.0, 5.0], [3.0, 7.0, 4.0]])
OBSTACLE_RADIUS = 1.0
COLLISION_THRESHOLD = 0.5  # 碰撞判定: ≤ radius + threshold = 1.5m
EFFECTIVE_OBSTACLE_RADIUS = OBSTACLE_RADIUS + COLLISION_THRESHOLD


# ============================================================
# 工具函数
# ============================================================
def print_section(title: str, width: int = 70):
    """打印章节标题"""
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def print_subsection(title: str):
    """打印子章节"""
    print(f"\n  --- {title} ---")


def build_ppo_agent() -> PPO:
    """构建与训练时一致的 PPO agent"""
    return PPO(
        state_dim=14, action_dim=3, action_max=1.0,
        lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
        epochs=10, minibatch_size=64, hidden_dim=128,
        use_adaptive_entropy=True, num_envs=1
    )


def run_eval_episode(env, agent: PPO, deterministic: bool = True) -> dict:
    """运行单个评估episode，返回详细指标"""
    state, info = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    path_length = 0.0
    prev_pos = state[:3].copy()
    min_obs_dist = float('inf')
    trajectory = [tuple(state[:3])]

    while True:
        action = agent.select_action(state, deterministic=deterministic)
        next_state, reward, terminated, truncated, info = env.step(action)

        ep_reward += reward
        ep_steps += 1
        current_pos = next_state[:3]
        path_length += np.linalg.norm(current_pos - prev_pos)
        prev_pos = current_pos.copy()
        trajectory.append(tuple(current_pos))

        if 'current_pos' in info:
            obs_dist = env._get_min_obstacle_distance(info['current_pos'])
            min_obs_dist = min(min_obs_dist, obs_dist)

        if terminated or truncated:
            break

        state = next_state

    return {
        'reward': float(ep_reward),
        'steps': ep_steps,
        'path_length': float(path_length),
        'min_obs_dist': float(min_obs_dist),
        'success': bool(info.get('reached_target', False)),
        'collision': bool(info.get('collision', False)),
        'timeout': (not info.get('reached_target', False) and not info.get('collision', False)),
        'target_pos': tuple(info.get('target_pos', (0, 0, 0))),
        'trajectory': trajectory,
    }


# ============================================================
# Benchmark 1: Baseline 复现
# ============================================================
def benchmark_baseline(model_path: str = MODEL_PATH, num_episodes: int = 50, seed: int = 42) -> dict:
    """
    Baseline 复现验证
    目标: 成功率 96-100%, 碰撞率 ≤ 4%
    """
    print_section("Benchmark 1: Baseline 复现验证")
    print(f"  模型: {model_path}")
    print(f"  评估轮数: {num_episodes} | 种子: {seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = DroneEnv()
    env.reset(seed=seed)

    agent = build_ppo_agent()
    agent.load_model(model_path)

    results_list = []
    for ep in range(num_episodes):
        result = run_eval_episode(env, agent, deterministic=True)
        results_list.append(result)

    successes = sum(1 for r in results_list if r['success'])
    collisions = sum(1 for r in results_list if r['collision'])
    timeouts = sum(1 for r in results_list if r['timeout'])
    success_rate = successes / num_episodes * 100
    collision_rate = collisions / num_episodes * 100
    timeout_rate = timeouts / num_episodes * 100

    rewards = [r['reward'] for r in results_list]
    success_steps = [r['steps'] for r in results_list if r['success']]
    path_lengths = [r['path_length'] for r in results_list]
    min_obs_dists = [r['min_obs_dist'] for r in results_list]

    # 打印结果
    print(f"\n  成功率:    {successes}/{num_episodes} = {success_rate:.1f}%")
    print(f"  碰撞率:    {collisions}/{num_episodes} = {collision_rate:.1f}%")
    print(f"  超时率:    {timeouts}/{num_episodes} = {timeout_rate:.1f}%")
    print(f"  平均奖励:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    if success_steps:
        print(f"  成功步数:  mean={np.mean(success_steps):.1f}, min={np.min(success_steps)}, max={np.max(success_steps)}")
    print(f"  路径长度:  mean={np.mean(path_lengths):.2f}m, min={np.min(path_lengths):.2f}m")
    print(f"  最近障碍:  mean={np.mean(min_obs_dists):.2f}m, min={np.min(min_obs_dists):.2f}m")

    # 通过判定
    pass_success = 96.0 <= success_rate <= 100.0
    pass_collision = collision_rate <= 4.0
    overall_pass = pass_success and pass_collision

    print(f"\n  成功率目标 96-100%: [{'PASS' if pass_success else 'FAIL'}] ({success_rate:.1f}%)")
    print(f"  碰撞率目标 ≤4%:    [{'PASS' if pass_collision else 'FAIL'}] ({collision_rate:.1f}%)")
    print(f"  综合判定:           [{'PASS' if overall_pass else 'FAIL'}]")

    return {
        'name': 'baseline_reproduction',
        'model_path': model_path,
        'num_episodes': num_episodes,
        'seed': seed,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'avg_reward': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'avg_success_steps': float(np.mean(success_steps)) if success_steps else None,
        'avg_path_length': float(np.mean(path_lengths)),
        'avg_min_obs_dist': float(np.mean(min_obs_dists)),
        'pass_success': pass_success,
        'pass_collision': pass_collision,
        'overall_pass': overall_pass,
    }


# ============================================================
# Benchmark 2: 噪声环境训练管线完整性
# ============================================================
def benchmark_noisy_training(num_episodes: int = 100, sigma: float = 0.5, seed: int = 42) -> dict:
    """
    噪声环境训练管线完整性验证
    目标: 位置噪声 σ=0.5, 训练 100 轮无崩溃
    """
    print_section("Benchmark 2: 噪声环境训练管线完整性")
    print(f"  噪声模式: pos | σ: {sigma} | 训练轮数: {num_episodes}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # 使用 SyncVectorEnv 单环境训练
    from gymnasium.vector import SyncVectorEnv

    def make_env():
        return NoisyDroneEnv.from_pattern('pos', sigma=sigma)

    env = SyncVectorEnv([make_env])
    eval_env = make_env()

    agent = build_ppo_agent()
    agent.num_envs = 1  # 单环境训练

    episode_rewards = []
    episode_lengths = []
    crashes = 0

    print(f"\n  开始训练...")
    start_time = time.time()

    try:
        states = env.reset()[0]

        for episode in range(num_episodes):
            ep_reward = 0.0
            ep_length = 0

            for step in range(256):  # 单环境 rollout = 256 steps
                action, log_prob, value, entropy = agent.get_action(states[0], deterministic=False)
                actions_np = np.array([action])
                next_states, rewards, terminateds, truncateds, infos = env.step(actions_np)
                dones = np.logical_or(terminateds, truncateds)

                agent.store_transition(
                    state=states[0], action=actions_np[0],
                    reward=float(rewards[0]), next_state=next_states[0],
                    done=bool(dones[0]), log_prob=log_prob, value=value, entropy=entropy
                )

                ep_reward += float(rewards[0])
                ep_length += 1
                states = next_states

            # PPO update
            lr = 3e-4 * (1 - episode / num_episodes)
            agent.set_lr(lr)
            update_result = agent.update()

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

            if (episode + 1) % 10 == 0 or episode == 0:
                elapsed = time.time() - start_time
                print(f"    Ep {episode+1:4d}/{num_episodes} | "
                      f"reward={ep_reward:8.2f} | steps={ep_length:4d} | "
                      f"loss={update_result['total_loss']:.4f} | "
                      f"elapsed={elapsed:.1f}s")

    except Exception as e:
        print(f"\n  [FAIL] 训练崩溃于 Ep {episode+1}: {e}")
        crashes += 1
        return {
            'name': 'noisy_training_integrity',
            'noise_pattern': 'pos',
            'sigma': sigma,
            'num_episodes': num_episodes,
            'completed_episodes': episode + 1,
            'crashes': crashes,
            'pipeline_intact': False,
            'avg_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            'avg_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        }

    total_time = time.time() - start_time
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)

    print(f"\n  训练完成! 总耗时: {total_time:.1f}s")
    print(f"  平均每轮奖励: {avg_reward:.2f}")
    print(f"  平均每轮步数: {avg_length:.1f}")
    print(f"  崩溃次数: {crashes}")
    print(f"  训练管线完整性: [{'PASS' if crashes == 0 else 'FAIL'}]")

    env.close()
    eval_env.close()

    return {
        'name': 'noisy_training_integrity',
        'noise_pattern': 'pos',
        'sigma': sigma,
        'num_episodes': num_episodes,
        'completed_episodes': num_episodes,
        'crashes': crashes,
        'pipeline_intact': crashes == 0,
        'avg_reward': float(avg_reward),
        'avg_length': float(avg_length),
        'total_time_seconds': total_time,
    }


# ============================================================
# Benchmark 3: 噪声环境观测差异量化
# ============================================================
def benchmark_noisy_obs_comparison(sigma: float = 0.5, num_timesteps: int = 3) -> dict:
    """
    噪声环境: 观测差异可量化
    打印 N 个时间步的 clean obs vs noisy obs 对比
    """
    print_section("Benchmark 3: 噪声环境观测差异量化")
    print(f"  噪声模式: pos | σ: {sigma} | 打印 {num_timesteps} 个时间步")

    DIM_NAMES = NoisyDroneEnv.DIM_NAMES
    env = NoisyDroneEnv.from_pattern('pos', sigma=sigma)
    env.reset(seed=42)

    comparisons = []

    for t in range(num_timesteps):
        clean = env.get_clean_observation()
        noisy = env._get_observation()
        noise = noisy - clean

        active_dims = NOISE_PATTERN_DIMS['pos']  # [0, 1, 2]
        inactive_dims = [d for d in range(14) if d not in active_dims]

        print(f"\n  ┌─ 时间步 {t+1} ─────────────────────────────────────────────┐")
        print(f"  │ {'维度':<6} {'名称':<12} {'Clean':>10} {'Noisy':>10} {'噪声':>10} {'差异':>10} │")
        print(f"  │ {'-'*58} │")

        step_comparison = {'timestep': t+1, 'dims': {}}
        for dim in range(14):
            name = DIM_NAMES[dim]
            diff = abs(noise[dim])
            marker = " <--噪声" if dim in active_dims else ""
            print(f"  │ {dim:<6} {name:<12} {clean[dim]:10.4f} {noisy[dim]:10.4f} {noise[dim]:+10.4f} {diff:10.4f}{marker}")
            step_comparison['dims'][dim] = {
                'name': name,
                'clean': float(clean[dim]),
                'noisy': float(noisy[dim]),
                'noise': float(noise[dim]),
                'abs_diff': float(diff),
                'is_active': dim in active_dims,
            }

        # 验证
        active_diffs = [abs(noise[d]) for d in active_dims]
        inactive_diffs = [abs(noise[d]) for d in inactive_dims]
        print(f"  │ {'-'*58} │")
        print(f"  │ 活跃维度(0-2) 平均差异: {np.mean(active_diffs):.4f} (期望 ~σ={sigma})")
        print(f"  │ 非活跃维度(3-13) 平均差异: {np.mean(inactive_diffs):.6f} (期望 ≈0)")
        print(f"  └{'─'*59}┘")

        step_comparison['active_mean_diff'] = float(np.mean(active_diffs))
        step_comparison['inactive_mean_diff'] = float(np.mean(inactive_diffs))
        comparisons.append(step_comparison)

        # 执行一步随机动作以推进环境
        if t < num_timesteps - 1:
            action = np.random.uniform(-1, 1, 3).astype(np.float32)
            env.step(action)

    # 验证: 活跃维度有显著噪声，非活跃维度无噪声
    all_active = [c['active_mean_diff'] for c in comparisons]
    all_inactive = [c['inactive_mean_diff'] for c in comparisons]
    active_ok = all(d > 0.1 for d in all_active)  # 噪声明显存在
    inactive_ok = all(d < 1e-5 for d in all_inactive)  # 非活跃维度几乎无噪声

    print(f"\n  噪声注入验证:")
    print(f"    活跃维度有噪声: [{'PASS' if active_ok else 'FAIL'}] ({', '.join(f'{d:.4f}' for d in all_active)})")
    print(f"    非活跃维度无噪声: [{'PASS' if inactive_ok else 'FAIL'}] ({', '.join(f'{d:.6f}' for d in all_inactive)})")

    env.close()

    return {
        'name': 'noisy_obs_comparison',
        'noise_pattern': 'pos',
        'sigma': sigma,
        'num_timesteps': num_timesteps,
        'comparisons': comparisons,
        'active_dims_have_noise': active_ok,
        'inactive_dims_clean': inactive_ok,
        'overall_pass': active_ok and inactive_ok,
    }


# ============================================================
# Oracle 1: 完美信息 Oracle (= Baseline 评估结果)
# ============================================================
def oracle_perfect_info(model_path: str = MODEL_PATH, num_episodes: int = 50, seed: int = 42) -> dict:
    """
    完美信息 Oracle: 无噪声 Baseline 策略表现
    预期: 98% 成功率, ~48.7 步, ~15.05m 路径
    """
    print_section("Oracle 1: 完美信息 Oracle (无噪声 Baseline)")

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = DroneEnv()
    env.reset(seed=seed)
    agent = build_ppo_agent()
    agent.load_model(model_path)

    results_list = [run_eval_episode(env, agent, deterministic=True) for _ in range(num_episodes)]

    successes = sum(1 for r in results_list if r['success'])
    collisions = sum(1 for r in results_list if r['collision'])
    success_rate = successes / num_episodes * 100
    collision_rate = collisions / num_episodes * 100

    success_steps = [r['steps'] for r in results_list if r['success']]
    path_lengths = [r['path_length'] for r in results_list]

    print(f"\n  成功率:    {success_rate:.1f}% (目标 ~98%)")
    print(f"  碰撞率:    {collision_rate:.1f}%")
    if success_steps:
        print(f"  成功步数:  mean={np.mean(success_steps):.1f} (目标 ~48.7)")
    print(f"  路径长度:  mean={np.mean(path_lengths):.2f}m (目标 ~15.05m)")

    # 与目标对比
    ref_success = 98.0
    ref_steps = 48.7
    ref_path = 15.05

    print(f"\n  与理论值对比:")
    print(f"    成功率偏差:  {success_rate - ref_success:+.1f}%")
    if success_steps:
        print(f"    平均步数偏差: {np.mean(success_steps) - ref_steps:+.1f}")
    print(f"    路径长度偏差: {np.mean(path_lengths) - ref_path:+.2f}m")

    return {
        'name': 'oracle_perfect_info',
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'avg_success_steps': float(np.mean(success_steps)) if success_steps else None,
        'avg_path_length': float(np.mean(path_lengths)),
        'reference': {
            'success_rate': ref_success,
            'avg_steps': ref_steps,
            'avg_path_length': ref_path,
        },
    }


# ============================================================
# Oracle 2: 随机策略 Oracle
# ============================================================
def oracle_random_policy(num_episodes: int = 50, seed: int = 42) -> dict:
    """
    随机策略 Oracle: 最简单环境中的随机动作下界
    action = np.random.uniform(-1, 1, 3), 记录成功率
    """
    print_section("Oracle 2: 随机策略 Oracle (下界)")

    np.random.seed(seed)

    env = DroneEnv()

    results_list = []
    for ep in range(num_episodes):
        state, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        path_length = 0.0
        prev_pos = state[:3].copy()

        while True:
            action = np.random.uniform(-1, 1, 3).astype(np.float32)
            next_state, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_steps += 1
            path_length += np.linalg.norm(next_state[:3] - prev_pos)
            prev_pos = next_state[:3]

            if terminated or truncated:
                break

        results_list.append({
            'reward': float(ep_reward),
            'steps': ep_steps,
            'path_length': float(path_length),
            'success': bool(info.get('reached_target', False)),
            'collision': bool(info.get('collision', False)),
            'timeout': (not info.get('reached_target', False) and not info.get('collision', False)),
        })

        if (ep + 1) % 10 == 0:
            current_success = sum(1 for r in results_list if r['success'])
            print(f"    Ep {ep+1:3d}/{num_episodes} | 累计成功: {current_success}/{ep+1}")

    successes = sum(1 for r in results_list if r['success'])
    collisions = sum(1 for r in results_list if r['collision'])
    timeouts = sum(1 for r in results_list if r['timeout'])
    success_rate = successes / num_episodes * 100
    collision_rate = collisions / num_episodes * 100

    rewards = [r['reward'] for r in results_list]
    steps = [r['steps'] for r in results_list]

    print(f"\n  成功率:    {success_rate:.1f}% (随机下界)")
    print(f"  碰撞率:    {collision_rate:.1f}%")
    print(f"  超时率:    {timeouts/num_episodes*100:.1f}%")
    print(f"  平均奖励:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  平均步数:  {np.mean(steps):.1f}")

    env.close()

    return {
        'name': 'oracle_random_policy',
        'num_episodes': num_episodes,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeouts / num_episodes * 100,
        'avg_reward': float(np.mean(rewards)),
        'avg_steps': float(np.mean(steps)),
    }


# ============================================================
# Oracle 3: A* Oracle (简化版 3D A*)
# ============================================================
def _is_collision_free(point: np.ndarray, obstacles: np.ndarray,
                       effective_radius: float = EFFECTIVE_OBSTACLE_RADIUS,
                       boundary_min: np.ndarray = None,
                       boundary_max: np.ndarray = None) -> bool:
    """检查点是否在无障碍区域"""
    if boundary_min is not None:
        if np.any(point < boundary_min) or np.any(point > boundary_max):
            return False

    for obs in obstacles:
        if np.linalg.norm(point - obs) < effective_radius:
            return False
    return True


def _segment_collision_free(p1: np.ndarray, p2: np.ndarray, obstacles: np.ndarray,
                            effective_radius: float = EFFECTIVE_OBSTACLE_RADIUS,
                            num_checks: int = 20) -> bool:
    """检查线段 p1→p2 是否与障碍物无碰撞（离散采样检查）"""
    for i in range(num_checks + 1):
        t = i / num_checks
        point = p1 + t * (p2 - p1)
        for obs in obstacles:
            if np.linalg.norm(point - obs) < effective_radius:
                return False
    return True


def astar_3d(start: np.ndarray, goal: np.ndarray,
             obstacles: np.ndarray,
             effective_radius: float = EFFECTIVE_OBSTACLE_RADIUS,
             boundary_min: np.ndarray = None,
             boundary_max: np.ndarray = None,
             grid_resolution: float = 0.5,
             max_expansions: int = 50000) -> Tuple[Optional[List[np.ndarray]], float, dict]:
    """
    简化版 3D A* 路径规划

    在已知全局地图条件下计算理论最优路径。
    当起点到终点直线无障碍时，退化为直线路径。

    Returns:
        (path, path_length, stats)
        - path: 路径点列表，或 None（无可行路径）
        - path_length: 路径总长度
        - stats: 统计信息 (expansions, 碰撞检查次数等)
    """
    if boundary_min is None:
        boundary_min = np.array([-10.0, -10.0, 0.0])
    if boundary_max is None:
        boundary_max = np.array([10.0, 10.0, 10.0])

    # 先检查直线路径（无障碍物时为最优解）
    if _segment_collision_free(start, goal, obstacles, effective_radius):
        straight_dist = np.linalg.norm(goal - start)
        return [start, goal], straight_dist, {
            'expansions': 0,
            'collision_checks': 1,
            'straight_line': True,
        }

    # 离散化起点和终点到网格
    def to_grid(pos):
        idx = ((pos - boundary_min) / grid_resolution).astype(int)
        return tuple(np.clip(idx, 0, np.floor((boundary_max - boundary_min) / grid_resolution).astype(int)))

    def from_grid(idx):
        return boundary_min + np.array(idx) * grid_resolution

    start_idx = to_grid(start)
    goal_idx = to_grid(goal)

    # 26-邻域 (3D 全体邻居)
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbors.append(np.array([dx, dy, dz]))

    grid_shape = tuple(np.floor((boundary_max - boundary_min) / grid_resolution).astype(int) + 1)

    # A* 数据结构
    open_set = []
    start_h = np.linalg.norm(goal - start)  # 欧几里得启发式
    heapq.heappush(open_set, (start_h, 0, start_idx))
    came_from = {}
    g_score = {start_idx: 0.0}
    expansions = 0
    collision_checks = 0

    while open_set and expansions < max_expansions:
        _, _, current_idx = heapq.heappop(open_set)
        current_pos = from_grid(current_idx)

        if current_idx == goal_idx:
            # 重建路径
            path = [goal]
            idx = goal_idx
            while idx in came_from:
                idx = came_from[idx]
                path.append(from_grid(idx))
            path.reverse()

            # 路径后处理：移除冗余点（直线连接缩短路径）
            path = _simplify_path(path, obstacles, effective_radius)

            total_length = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
            return path, total_length, {
                'expansions': expansions,
                'collision_checks': collision_checks,
                'straight_line': False,
                'grid_resolution': grid_resolution,
            }

        expansions += 1

        for offset in neighbors:
            neighbor_idx = (current_idx[0] + offset[0],
                           current_idx[1] + offset[1],
                           current_idx[2] + offset[2])

            # 边界检查
            if not (0 <= neighbor_idx[0] < grid_shape[0] and
                    0 <= neighbor_idx[1] < grid_shape[1] and
                    0 <= neighbor_idx[2] < grid_shape[2]):
                continue

            neighbor_pos = from_grid(neighbor_idx)
            collision_checks += 1

            if not _is_collision_free(neighbor_pos, obstacles, effective_radius,
                                     boundary_min, boundary_max):
                continue

            move_cost = np.linalg.norm(neighbor_pos - current_pos)
            tentative_g = g_score[current_idx] + move_cost

            if neighbor_idx not in g_score or tentative_g < g_score[neighbor_idx]:
                g_score[neighbor_idx] = tentative_g
                h = np.linalg.norm(goal - neighbor_pos)
                f = tentative_g + h
                heapq.heappush(open_set, (f, tentative_g, neighbor_idx))
                came_from[neighbor_idx] = current_idx

    # 无可行路径
    return None, float('inf'), {
        'expansions': expansions,
        'collision_checks': collision_checks,
        'straight_line': False,
    }


def _simplify_path(path: List[np.ndarray], obstacles: np.ndarray,
                   effective_radius: float) -> List[np.ndarray]:
    """后处理：移除路径中可通过直线连接的冗余中间点"""
    if len(path) <= 2:
        return path

    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        # 尽量远跳
        for j in range(len(path) - 1, i, -1):
            if _segment_collision_free(path[i], path[j], obstacles, effective_radius):
                simplified.append(path[j])
                i = j
                break
        else:
            i += 1
            simplified.append(path[i])

    return simplified


def oracle_astar(num_samples: int = 50, grid_resolution: float = 0.5, seed: int = 42) -> dict:
    """
    A* Oracle: 理论最优路径下界

    对随机生成的起点-终点对运行 A*，记录最优路径长度。
    当无障碍物阻碍时退化为直线，有障碍物时绕行。
    """
    print_section("Oracle 3: A* Oracle (理论最优路径下界)")
    print(f"  样本数: {num_samples} | 网格分辨率: {grid_resolution}m")

    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    boundary_min = SPACE_BOUNDS[0]
    boundary_max = SPACE_BOUNDS[1]

    # 起点范围 (与 DroneEnv.reset 一致)
    start_min = boundary_min + 1.0
    start_max = np.array([2.0, 2.0, 2.0])

    path_lengths = []
    straight_line_lengths = []
    straight_line_count = 0
    success_count = 0
    expansions_list = []

    print(f"\n  障碍物: {OBSTACLES.tolist()}")
    print(f"  有效半径: {EFFECTIVE_OBSTACLE_RADIUS}m (radius + threshold)")
    print(f"  起点范围: {start_min} → {start_max}")
    print(f"  目标范围: {TARGET_MIN} → {TARGET_MAX}")

    for i in range(num_samples):
        start = rng.uniform(start_min, start_max)
        goal = rng.uniform(TARGET_MIN, TARGET_MAX)
        straight_dist = np.linalg.norm(goal - start)
        straight_line_lengths.append(straight_dist)

        path, path_len, stats = astar_3d(
            start, goal, OBSTACLES,
            effective_radius=EFFECTIVE_OBSTACLE_RADIUS,
            boundary_min=boundary_min, boundary_max=boundary_max,
            grid_resolution=grid_resolution
        )

        if path is not None:
            path_lengths.append(path_len)
            success_count += 1

        expansions_list.append(stats['expansions'])
        if stats.get('straight_line', False):
            straight_line_count += 1

        if (i + 1) % 10 == 0:
            print(f"    样本 {i+1:3d}/{num_samples} | "
                  f"直线={straight_dist:.2f}m | A*={path_len:.2f}m | "
                  f"expansion={stats['expansions']:5d} | "
                  f"直线占比={straight_line_count}/{i+1}")

    success_rate = success_count / num_samples * 100
    avg_path = np.mean(path_lengths) if path_lengths else 0.0
    avg_straight = np.mean(straight_line_lengths)
    min_path = np.min(path_lengths) if path_lengths else float('inf')
    max_path = np.max(path_lengths) if path_lengths else float('inf')

    print(f"\n  A* 结果 ({success_count}/{num_samples} 成功):")
    print(f"    平均最优路径:  {avg_path:.2f}m (理论下界)")
    print(f"    最小路径:      {min_path:.2f}m")
    print(f"    最大路径:      {max_path:.2f}m")
    print(f"    平均直线距离:  {avg_straight:.2f}m")
    print(f"    直线路径占比:  {straight_line_count}/{num_samples} ({straight_line_count/num_samples*100:.0f}%)")
    print(f"    平均扩展节点:  {np.mean(expansions_list):.0f}")
    print(f"    与直线比:      {avg_path/avg_straight*100:.1f}% (路径效率)")

    # 直线距离作为"无障碍物下界"
    straight_lower_bound = np.min(straight_line_lengths)

    print(f"\n  理论下界 (无障碍直线): {straight_lower_bound:.2f}m")
    print(f"  A* 实际下界:            {min_path:.2f}m")

    return {
        'name': 'oracle_astar',
        'num_samples': num_samples,
        'grid_resolution': grid_resolution,
        'success_count': success_count,
        'success_rate': success_rate,
        'avg_astar_path': float(avg_path),
        'min_astar_path': float(min_path),
        'max_astar_path': float(max_path),
        'avg_straight_line': float(avg_straight),
        'min_straight_line': float(min(np.min(straight_line_lengths), 0)),
        'straight_line_ratio': straight_line_count / num_samples,
        'avg_expansions': float(np.mean(expansions_list)),
        'path_efficiency': float(avg_path / avg_straight * 100) if avg_straight > 0 else 0.0,
    }


# ============================================================
# A* 详细演示（单个场景）
# ============================================================
def astar_detailed_demo():
    """A* 详细演示：对固定场景展示搜索过程"""
    print_section("A* 详细演示 (固定场景)")

    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([7.0, 7.0, 5.0])
    straight_dist = np.linalg.norm(goal - start)

    print(f"  起点: {start}")
    print(f"  终点: {goal}")
    print(f"  直线距离: {straight_dist:.2f}m")
    print(f"  障碍物:")
    for i, obs in enumerate(OBSTACLES):
        dist_start = np.linalg.norm(start - obs)
        dist_goal = np.linalg.norm(goal - obs)
        blocks_path = "是" if dist_start < EFFECTIVE_OBSTACLE_RADIUS or dist_goal < EFFECTIVE_OBSTACLE_RADIUS else "否"
        print(f"    Obs{i+1} @ {obs}: 距起点={dist_start:.1f}m, 距终点={dist_goal:.1f}m, 阻挡={blocks_path}")

    path, path_len, stats = astar_3d(start, goal, OBSTACLES, grid_resolution=0.5)

    if path is not None:
        print(f"\n  A* 路径 (共 {len(path)} 个点, {path_len:.2f}m):")
        for i, pt in enumerate(path):
            if i == 0:
                tag = "起点"
            elif i == len(path) - 1:
                tag = "终点"
            else:
                tag = f"拐点{i}"
            print(f"    [{tag:6s}] ({pt[0]:6.2f}, {pt[1]:6.2f}, {pt[2]:6.2f})")

        print(f"\n  统计:")
        print(f"    路径长度: {path_len:.2f}m")
        print(f"    直线距离: {straight_dist:.2f}m")
        print(f"    绕行距离: {path_len - straight_dist:.2f}m")
        print(f"    路径效率: {straight_dist/path_len*100:.1f}%")
        print(f"    扩展节点: {stats['expansions']}")
        print(f"    碰撞检查: {stats['collision_checks']}")
        print(f"    直线路径: {'是' if stats['straight_line'] else '否'}")
    else:
        print(f"\n  [FAIL] A* 未找到可行路径!")


# ============================================================
# 汇总报告
# ============================================================
def print_summary(all_results: dict):
    """打印 Week 1 全部结果汇总"""
    print_section("Week 1 Benchmark & Oracle — 汇总报告", width=80)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Benchmarks
    print(f"\n  {'─'*76}")
    print(f"  [Benchmarks]")
    print(f"  {'─'*76}")

    if 'baseline' in all_results:
        b = all_results['baseline']
        status = 'PASS' if b['overall_pass'] else 'FAIL'
        print(f"  B1. Baseline 复现:      [{status}]")
        print(f"      成功率: {b['success_rate']:.1f}% (目标 96-100%)")
        print(f"      碰撞率: {b['collision_rate']:.1f}% (目标 ≤4%)")

    if 'noisy_training' in all_results:
        nt = all_results['noisy_training']
        status2 = 'PASS' if nt['pipeline_intact'] else 'FAIL'
        print(f"  B2. 噪声训练管线:       [{status2}]")
        print(f"      完成轮数: {nt['completed_episodes']}/{nt['num_episodes']} | 崩溃: {nt['crashes']}")

    if 'noisy_obs' in all_results:
        no = all_results['noisy_obs']
        status3 = 'PASS' if no['overall_pass'] else 'FAIL'
        print(f"  B3. 噪声观测差异:       [{status3}]")
        print(f"      活跃维度有噪声/非活跃维度无噪声: {no['active_dims_have_noise']}/{no['inactive_dims_clean']}")

    # Oracles
    print(f"\n  {'─'*76}")
    print(f"  [Oracles]")
    print(f"  {'─'*76}")

    if 'oracle_perfect' in all_results:
        op = all_results['oracle_perfect']
        print(f"  O1. 完美信息 Oracle:   成功率={op['success_rate']:.1f}% | 步数={op['avg_success_steps']:.1f} | 路径={op['avg_path_length']:.2f}m")
        print(f"      参考值:             成功率=98% | 步数=48.7 | 路径=15.05m")

    if 'oracle_random' in all_results:
        oro = all_results['oracle_random']
        print(f"  O2. 随机策略 Oracle:   成功率={oro['success_rate']:.1f}% | 碰撞率={oro['collision_rate']:.1f}%")
        print(f"      (随机下界 — 任何有效策略应高于此值)")

    if 'oracle_astar' in all_results:
        oa = all_results['oracle_astar']
        print(f"  O3. A* Oracle:         平均={oa['avg_astar_path']:.2f}m | 最小={oa['min_astar_path']:.2f}m")
        print(f"      直线平均:           {oa['avg_straight_line']:.2f}m")
        print(f"      路径效率:           {oa['path_efficiency']:.1f}%")
        print(f"      (理论下界 — 任何策略的路径长度应 ≥ A* 路径)")

    # Comparison table
    print(f"\n  {'─'*76}")
    print(f"  [Oracle Comparison] (下界 -> 上界)")
    print(f"  {'─'*76}")

    random_sr = all_results.get('oracle_random', {}).get('success_rate', 0)
    baseline_sr = all_results.get('baseline', {}).get('success_rate', 0)
    baseline_path = all_results.get('baseline', {}).get('avg_path_length', 0)
    astar_path = all_results.get('oracle_astar', {}).get('avg_astar_path', 0)
    straight_line = all_results.get('oracle_astar', {}).get('avg_straight_line', 0)

    print(f"  成功率链条:")
    print(f"    随机策略 ({random_sr:.1f}%)  <  PPO策略 ({baseline_sr:.1f}%)  <  100%")
    print(f"  路径长度链条:")
    print(f"    直线距离 ({straight_line:.2f}m)  ≤  A*最优 ({astar_path:.2f}m)  ≤  PPO实际 ({baseline_path:.2f}m)")

    print(f"\n  {'═'*76}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Week 1 Benchmark & Oracle')
    parser.add_argument('--skip-training', action='store_true',
                        help='跳过耗时训练 (Benchmark 2)')
    parser.add_argument('--only-oracles', action='store_true',
                        help='仅运行 Oracle 部分')
    parser.add_argument('--only-benchmarks', action='store_true',
                        help='仅运行 Benchmark 部分')
    parser.add_argument('--fast', action='store_true',
                        help='快速模式: 减少评估轮数')
    args = parser.parse_args()

    run_benchmarks = not args.only_oracles
    run_oracles = not args.only_benchmarks
    do_training = not args.skip_training

    if args.fast:
        num_eval = 10
        num_train = 10
        num_astar = 10
        print("[FAST] 快速模式: 减少评估轮数")
    else:
        num_eval = 50
        num_train = 100
        num_astar = 50

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("  Week 1 Benchmark & Oracle — UAV RL 3D Path Planning")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  设备: {DEVICE}")
    print("=" * 80)

    # ==================== Benchmarks ====================
    if run_benchmarks:
        # B1: Baseline 复现
        all_results['baseline'] = benchmark_baseline(
            model_path=MODEL_PATH, num_episodes=num_eval, seed=42
        )

        # B2: 噪声环境训练管线 (耗时)
        if do_training:
            all_results['noisy_training'] = benchmark_noisy_training(
                num_episodes=num_train, sigma=0.5, seed=42
            )
        else:
            print("\n  [SKIP] 跳过 B2: 噪声环境训练 (--skip-training)")

        # B3: 噪声观测差异
        all_results['noisy_obs'] = benchmark_noisy_obs_comparison(
            sigma=0.5, num_timesteps=3
        )

    # ==================== Oracles ====================
    if run_oracles:
        # O1: 完美信息 Oracle (= baseline)
        if 'baseline' in all_results:
            all_results['oracle_perfect'] = all_results['baseline'].copy()
            all_results['oracle_perfect']['name'] = 'oracle_perfect_info'
            print_section("Oracle 1: 完美信息 Oracle")
            print("  (使用 B1 Baseline 结果，跳过重复评估)")
        else:
            all_results['oracle_perfect'] = oracle_perfect_info(
                model_path=MODEL_PATH, num_episodes=num_eval, seed=42
            )

        # O2: 随机策略 Oracle
        all_results['oracle_random'] = oracle_random_policy(
            num_episodes=num_eval, seed=42
        )

        # O3: A* Oracle
        all_results['oracle_astar'] = oracle_astar(
            num_samples=num_astar, grid_resolution=0.5, seed=42
        )

        # A* 详细演示
        astar_detailed_demo()

    # ==================== 保存结果 ====================
    # 清理 comparisons 中的 numpy 类型（避免 JSON 序列化问题）
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_serializable = make_serializable(all_results)

    result_path = os.path.join(RESULTS_DIR, f'week1_benchmark_{timestamp}.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(DEVICE),
            'fast_mode': args.fast,
            'results': results_serializable,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  [SAVED] 结果已保存: {result_path}")

    # ==================== 汇总 ====================
    print_summary(all_results)

    print("\n[DONE] Week 1 Benchmark & Oracle 完成!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
