"""
Baseline 评估脚本
加载最佳模型，在标准环境中跑 50 次评估，确认 96-100% 成功率
"""
import numpy as np
import torch
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone_env import DroneEnv
from ppo_agent import PPO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_path, num_episodes=50, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = DroneEnv()
    env.reset(seed=seed)

    ppo = PPO(
        state_dim=14,
        action_dim=3,
        action_max=1.0,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=10,
        minibatch_size=64,
        hidden_dim=128,
        use_adaptive_entropy=True,
        num_envs=1
    )

    print(f"Loading model: {model_path}")
    ppo.load_model(model_path)
    print(f"Model loaded. Device: {DEVICE}")

    successes = 0
    collisions = 0
    timeouts = 0
    rewards = []
    steps_list = []
    path_lengths = []
    min_obs_dists = []
    reward_components_all = {k: [] for k in
        ['r_dist', 'r_heading', 'r_obs', 'r_smooth', 'r_goal', 'r_collision', 'r_timeout']}

    for ep in range(num_episodes):
        state, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        path_length = 0.0
        prev_pos = state[:3].copy()
        min_dist = float('inf')

        while True:
            action = ppo.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_steps += 1
            path_length += np.linalg.norm(next_state[:3] - prev_pos)
            prev_pos = next_state[:3]

            # track minimum obstacle distance
            if 'current_pos' in info:
                obs_dist = env._get_min_obstacle_distance(info['current_pos'])
                min_dist = min(min_dist, obs_dist)

            # collect reward components
            if 'reward_components' in info:
                for k in reward_components_all:
                    if k in info['reward_components']:
                        reward_components_all[k].append(info['reward_components'][k])

            if terminated or truncated:
                if info.get('reached_target', False):
                    successes += 1
                elif info.get('collision', False):
                    collisions += 1
                else:
                    timeouts += 1
                break

            state = next_state

        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        path_lengths.append(path_length)
        min_obs_dists.append(min_dist)

    # === Print Results ===
    print()
    print("=" * 60)
    print("  Swift PPO Baseline — 50 次评估结果")
    print("=" * 60)
    print(f"  模型: {model_path}")
    print(f"  评估轮数: {num_episodes}")
    print(f"  随机种子: {seed}")
    print()
    print(f"  成功率:    {successes}/{num_episodes} = {successes/num_episodes*100:.1f}%")
    print(f"  碰撞率:    {collisions}/{num_episodes} = {collisions/num_episodes*100:.1f}%")
    print(f"  超时率:    {timeouts}/{num_episodes} = {timeouts/num_episodes*100:.1f}%")
    print()
    print(f"  平均累计奖励:     {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  成功平均步数:     {np.mean([s for i,s in enumerate(steps_list) if i < successes]) if successes > 0 else 'N/A'}")
    print(f"  平均路径长度:     {np.mean(path_lengths):.2f}m")
    print(f"  平均最近障碍距离: {np.mean(min_obs_dists):.2f}m")

    # Reward component breakdown
    print()
    print("  奖励组件分解（每步平均）:")
    for k, vals in reward_components_all.items():
        if vals:
            print(f"    {k:15s}: {np.mean(vals):8.4f}")

    # Pass/fail
    success_rate = successes / num_episodes * 100
    pass_check = 96.0 <= success_rate <= 100.0
    print()
    status = "PASS" if pass_check else "FAIL"
    print(f"  [{status}] (目标: 96-100%, 实际: {success_rate:.1f}%)")
    print("=" * 60)

    # Save to JSON
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': model_path,
        'num_episodes': num_episodes,
        'seed': seed,
        'success_rate': successes / num_episodes,
        'collision_rate': collisions / num_episodes,
        'timeout_rate': timeouts / num_episodes,
        'avg_reward': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'avg_path_length': float(np.mean(path_lengths)),
        'avg_min_obs_dist': float(np.mean(min_obs_dists)),
        'reward_components': {k: float(np.mean(v)) for k, v in reward_components_all.items() if v},
    }

    out_path = os.path.join(os.path.dirname(__file__), 'eval_results', 'baseline_20260718.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  结果已保存: {out_path}")

    return results


if __name__ == '__main__':
    model_path = os.path.join(
        os.path.dirname(__file__),
        'saved_models', 'ppo_swift_3000ep_20260712_115059'
    )
    evaluate(model_path, num_episodes=50)
