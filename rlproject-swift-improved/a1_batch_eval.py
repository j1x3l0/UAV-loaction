"""
A1 噪声衰减曲线批量评估脚本
==============================
对 5 种噪声模式 (Pos, Vel, Target, Obs, Full) × 5 个噪声水平，
每种组合跑 50 次评估，输出 CSV + JSON。

输出:
  - eval_results/a1_batch_results_<timestamp>.csv
  - eval_results/a1_batch_results_<timestamp>.json
  - eval_results/a1_per_episode_<timestamp>.json  (逐episode明细)

Usage:
  python a1_batch_eval.py
  python a1_batch_eval.py --fast           # 每个水平只跑10次(快速验证)
  python a1_batch_eval.py --pattern pos    # 只跑指定模式
"""

import numpy as np
import torch
import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone_env import DroneEnv
from drone_env_noisy import NoisyDroneEnv
from ppo_agent import PPO, DEVICE

# ============================================================
# 路径
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'ppo_swift_3000ep_20260712_115059')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'eval_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# A1 实验网格定义
# ============================================================
# 每个实验定义: (实验名, pattern, 环境工厂函数)
# 返回 NoisyDroneEnv 实例

def _make_pos_env(sigma):
    return NoisyDroneEnv.from_pattern('pos', sigma=sigma)

def _make_vel_env(sigma):
    return NoisyDroneEnv.from_pattern('vel', sigma=sigma)

def _make_target_env(sigma):
    return NoisyDroneEnv.from_pattern('target', sigma=sigma)

def _make_full_env(sigma):
    return NoisyDroneEnv.from_pattern('full', sigma=sigma)

def _make_obs_env(sigma_dir, sigma_dist):
    """Obs模式: 方向噪声和距离噪声分别控制"""
    return NoisyDroneEnv(noise_config={
        'sigma_obs_dir': sigma_dir,
        'sigma_obs_dist': sigma_dist,
    })

A1_EXPERIMENTS = [
    # (label, pattern_display, env_builder)
    # env_builder is a callable that returns NoisyDroneEnv
]

# Pos: 5 levels
for s in [0.1, 0.5, 1.0, 2.0, 5.0]:
    A1_EXPERIMENTS.append((f"A1-Pos_s={s}", "pos", lambda s=s: _make_pos_env(s), s))

# Vel: 5 levels
for s in [0.1, 0.5, 1.0, 3.0, 5.0]:
    A1_EXPERIMENTS.append((f"A1-Vel_s={s}", "vel", lambda s=s: _make_vel_env(s), s))

# Target: 5 levels
for s in [0.1, 0.5, 1.0, 2.0, 5.0]:
    A1_EXPERIMENTS.append((f"A1-Target_s={s}", "target", lambda s=s: _make_target_env(s), s))

# Obs: 5 combos (dir, dist)
OBS_COMBOS = [
    (0.1, 0.1),
    (0.3, 0.1),
    (0.5, 0.1),
    (0.5, 0.5),
    (0.5, 1.0),
]
for sd, ss in OBS_COMBOS:
    A1_EXPERIMENTS.append((
        f"A1-Obs_dir={sd}_dist={ss}",
        "obs",
        lambda sd=sd, ss=ss: _make_obs_env(sd, ss),
        f"dir={sd},dist={ss}"
    ))

# Full: 5 levels
for s in [0.1, 0.3, 0.5, 1.0, 2.0]:
    A1_EXPERIMENTS.append((f"A1-Full_s={s}", "full", lambda s=s: _make_full_env(s), s))


# ============================================================
# 辅助函数
# ============================================================
def build_agent():
    return PPO(
        state_dim=14, action_dim=3, action_max=1.0,
        lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
        epochs=10, minibatch_size=64, hidden_dim=128,
        use_adaptive_entropy=True, num_envs=1
    )


def run_one_eval(env, agent, seed):
    """单次评估, 返回指标字典"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)

    state, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_steps = 0
    path_length = 0.0
    prev_pos = state[:3].copy()
    min_obs_dist = float('inf')
    reward_components = defaultdict(list)

    while True:
        action = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, info = env.step(action)

        ep_reward += float(reward)
        ep_steps += 1
        current_pos = next_state[:3]
        path_length += float(np.linalg.norm(current_pos - prev_pos))
        prev_pos = current_pos.copy()

        if 'current_pos' in info:
            obs_dist = env._get_min_obstacle_distance(info['current_pos'])
            min_obs_dist = min(min_obs_dist, obs_dist)

        if 'reward_components' in info:
            for k, v in info['reward_components'].items():
                reward_components[k].append(float(v))

        if terminated or truncated:
            break

        state = next_state

    return {
        'success': bool(info.get('reached_target', False)),
        'collision': bool(info.get('collision', False)),
        'timeout': (not info.get('reached_target', False) and not info.get('collision', False)),
        'reward': float(ep_reward),
        'steps': ep_steps,
        'path_length': float(path_length),
        'min_obs_dist': float(min_obs_dist),
        'reward_components': {k: float(np.mean(v)) for k, v in reward_components.items()},
    }


def aggregate_results(episodes):
    """汇总多个episode的指标"""
    n = len(episodes)
    successes = sum(1 for e in episodes if e['success'])
    collisions = sum(1 for e in episodes if e['collision'])
    timeouts  = sum(1 for e in episodes if e['timeout'])

    success_steps = [e['steps'] for e in episodes if e['success']]
    rewards = [e['reward'] for e in episodes]
    path_lengths = [e['path_length'] for e in episodes]
    min_dists = [e['min_obs_dist'] for e in episodes]

    return {
        'total_episodes': n,
        'success_count': successes,
        'collision_count': collisions,
        'timeout_count': timeouts,
        'success_rate': successes / n * 100,
        'collision_rate': collisions / n * 100,
        'timeout_rate': timeouts / n * 100,
        'avg_reward': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'avg_steps': float(np.mean(success_steps)) if success_steps else None,
        'avg_path_length': float(np.mean(path_lengths)),
        'avg_min_obs_dist': float(np.mean(min_dists)),
        'min_path_length': float(np.min(path_lengths)),
        'max_path_length': float(np.max(path_lengths)),
    }


def format_sigma_label(label, sigma_val):
    """格式化噪声标签"""
    return f"{label} (sigma={sigma_val})"


# ============================================================
# 主流程
# ============================================================
def run_a1_batch(model_path, num_episodes=50, fast=False, pattern_filter=None):
    """运行 A1 全量批量评估"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_experiments = len(A1_EXPERIMENTS)

    print("=" * 80)
    print("  A1 噪声衰减曲线 — 批量评估")
    print(f"  模型: {model_path}")
    print(f"  每组评估数: {'10 (fast)' if fast else num_episodes}")
    print(f"  实验总数: {total_experiments}")
    print(f"  设备: {DEVICE}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 加载模型一次
    agent = build_agent()
    agent.load_model(model_path)

    summary_rows = []    # 每个实验一行汇总
    all_episodes = {}    # 逐episode明细
    start_time = time.time()

    for idx, (name, pattern, env_factory, sigma_label) in enumerate(A1_EXPERIMENTS):
        if pattern_filter and pattern != pattern_filter:
            continue

        n_eps = 10 if fast else num_episodes
        print(f"\n{'─'*70}")
        print(f"  [{idx+1}/{total_experiments}] {name}")
        print(f"{'─'*70}")

        env = env_factory()
        episodes = []

        for ep in range(n_eps):
            result = run_one_eval(env, agent, seed=42 + ep * 100)
            episodes.append(result)

        agg = aggregate_results(episodes)
        elapsed = time.time() - start_time

        # 打印摘要
        print(f"  成功率: {agg['success_rate']:.1f}% | "
              f"碰撞率: {agg['collision_rate']:.1f}% | "
              f"超时率: {agg['timeout_rate']:.1f}%")
        print(f"  平均步数: {agg['avg_steps']:.1f}" if agg['avg_steps'] else f"  平均步数: N/A")
        print(f"  平均路径: {agg['avg_path_length']:.2f}m | "
              f"平均最近障碍: {agg['avg_min_obs_dist']:.2f}m")
        print(f"  平均奖励: {agg['avg_reward']:.2f} +- {agg['reward_std']:.2f}")
        print(f"  耗时: {elapsed:.1f}s")

        # 汇总行
        row = {
            'experiment': name,
            'pattern': pattern,
            'sigma': str(sigma_label),
            'num_episodes': n_eps,
            **agg,
        }
        summary_rows.append(row)
        all_episodes[name] = episodes

        env.close()

    total_time = time.time() - start_time

    # ============ 保存 CSV ============
    csv_path = os.path.join(RESULTS_DIR, f'a1_batch_results_{timestamp}.csv')
    csv_fields = [
        'experiment', 'pattern', 'sigma', 'num_episodes',
        'success_rate', 'collision_rate', 'timeout_rate',
        'avg_reward', 'reward_std', 'avg_steps',
        'avg_path_length', 'avg_min_obs_dist',
        'min_path_length', 'max_path_length',
        'success_count', 'collision_count', 'timeout_count',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n  [CSV] {csv_path}")

    # ============ 保存 JSON (汇总) ============
    json_path = os.path.join(RESULTS_DIR, f'a1_batch_results_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'device': str(DEVICE),
            'fast_mode': fast,
            'num_experiments': len(summary_rows),
            'total_time_seconds': total_time,
            'results': summary_rows,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [JSON] {json_path}")

    # ============ 保存 JSON (逐episode明细) ============
    detail_path = os.path.join(RESULTS_DIR, f'a1_per_episode_{timestamp}.json')
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'fast_mode': fast,
            'episodes': all_episodes,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [DETAIL] {detail_path}")

    # ============ 终端汇总表 ============
    print(f"\n{'='*80}")
    print(f"  A1 实验结果汇总")
    print(f"{'='*80}")
    print(f"\n  {'实验':<30} {'模式':<10} {'σ':>8} {'成功率':>8} {'碰撞率':>8} {'超时率':>8} {'步数':>7} {'路径':>7} {'障碍距':>7}")
    print(f"  {'─'*30} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7}")

    for row in summary_rows:
        steps_str = f"{row['avg_steps']:.1f}" if row['avg_steps'] else "N/A"
        print(f"  {row['experiment']:<30} {row['pattern']:<10} {str(row['sigma']):>8} "
              f"{row['success_rate']:7.1f}% {row['collision_rate']:7.1f}% {row['timeout_rate']:7.1f}% "
              f"{steps_str:>7} {row['avg_path_length']:6.2f}m {row['avg_min_obs_dist']:6.2f}m")

    # 按模式分组打印简洁衰减表
    print(f"\n{'='*80}")
    print(f"  噪声衰减曲线 (成功率 by σ)")
    print(f"{'='*80}")

    by_pattern = defaultdict(list)
    for row in summary_rows:
        by_pattern[row['pattern']].append(row)

    for pattern in ['pos', 'vel', 'target', 'obs', 'full']:
        if pattern not in by_pattern:
            continue
        rows = by_pattern[pattern]
        print(f"\n  [{pattern.upper()}]")
        print(f"  {'σ':>15}  {'成功率':>8}  {'碰撞率':>8}  {'路径(m)':>8}  {'障碍距(m)':>8}")
        print(f"  {'─'*15}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
        for r in rows:
            print(f"  {str(r['sigma']):>15}  {r['success_rate']:7.1f}%  {r['collision_rate']:7.1f}%  "
                  f"{r['avg_path_length']:7.2f}m  {r['avg_min_obs_dist']:7.2f}m")

    print(f"\n  总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  [DONE] A1 批量评估完成")

    return summary_rows, all_episodes


def main():
    parser = argparse.ArgumentParser(description='A1 噪声衰减曲线批量评估')
    parser.add_argument('--fast', action='store_true',
                        help='快速模式: 每组只跑10次评估')
    parser.add_argument('--pattern', type=str, default=None,
                        choices=['pos', 'vel', 'target', 'obs', 'full'],
                        help='只跑指定噪声模式')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='模型路径')
    args = parser.parse_args()

    run_a1_batch(
        model_path=args.model,
        num_episodes=50,
        fast=args.fast,
        pattern_filter=args.pattern,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
