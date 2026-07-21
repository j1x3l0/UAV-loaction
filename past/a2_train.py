"""
A2 鲁棒性训练 + 对比评估流水线
================================
训练:
  A2-Fixed:  PPO + full noise σ=0.5 (per-step) × 500ep
  A2-Rand:   PPO + full noise σ~U(0,2.0) (per-episode固定) × 500ep
  A2-Curric: PPO + full noise σ: 0→2.0 线性增长 × 500ep
  SAC:       SAC on clean DroneEnv × 500ep

评估:
  Clean baseline vs Fixed vs Rand vs Curric vs SAC
  统一在 5个full噪声水平 [0, 0.1, 0.5, 1.0, 2.0] 下各50次评估
  输出 CSV + JSON

Usage:
  python a2_train.py                          # 全部训练+评估
  python a2_train.py --skip-training          # 仅评估(需已有模型)
  python a2_train.py --train-only fixed       # 仅训练指定变体
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
from gymnasium.vector import SyncVectorEnv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone_env import DroneEnv
from drone_env_noisy import NoisyDroneEnv, NOISE_PATTERNS
from ppo_agent import PPO, DEVICE
from sac_agent_v2 import SAC

# ============================================================
# Paths
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'eval_results')
BASELINE_MODEL = os.path.join(MODEL_DIR, 'ppo_swift_3000ep_20260712_115059')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# A2 模型保存路径
A2_MODELS = {
    'fixed':  os.path.join(MODEL_DIR, 'a2_fixed_s0.5_500ep'),
    'rand':   os.path.join(MODEL_DIR, 'a2_rand_u0_2.0_500ep'),
    'curric': os.path.join(MODEL_DIR, 'a2_curric_0_2.0_500ep'),
    'sac':    os.path.join(MODEL_DIR, 'sac_v2_500ep'),
}


# ============================================================
# Per-Episode Noisy Environment
# ============================================================
class PerEpisodeNoisyEnv(NoisyDroneEnv):
    """
    扩展 NoisyDroneEnv, 支持 per-episode 固定的噪声 sigma。

    模式:
      'fixed':  sigma 始终为固定值
      'random': 每次 reset() 从 [sigma_min, sigma_max] 均匀采样
      'curric': sigma = sigma_min + progress*(sigma_max - sigma_min),
                进度由外部 set_curric_progress() 控制
    """

    def __init__(self, mode='fixed', sigma_val=0.5, sigma_min=0.0, sigma_max=2.0):
        self._mode = mode
        self._sigma_val = sigma_val
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._curric_progress = 0.0
        self._current_sigma = sigma_val

        # 先构造 noise_config
        noise_config = self._build_noise_config(sigma_val)
        super().__init__(noise_config=noise_config)

    def _build_noise_config(self, sigma):
        """将标量 sigma 映射到 full 模式的 noise_config"""
        return {k: sigma for k in NOISE_PATTERNS['full']}

    def set_curric_progress(self, progress):
        self._curric_progress = np.clip(progress, 0.0, 1.0)

    def _sample_sigma(self):
        if self._mode == 'fixed':
            return self._sigma_val
        elif self._mode == 'random':
            return np.random.uniform(self._sigma_min, self._sigma_max)
        elif self._mode == 'curric':
            return self._sigma_min + self._curric_progress * (self._sigma_max - self._sigma_min)
        return self._sigma_val

    def reset(self, seed=None, options=None):
        self._current_sigma = self._sample_sigma()
        # 重建噪声 std 向量
        new_config = self._build_noise_config(self._current_sigma)
        self.noise_config = new_config
        self._rebuild_noise_std()
        return super().reset(seed=seed, options=options)

    @property
    def current_sigma(self):
        return self._current_sigma


# ============================================================
# PPO Training
# ============================================================
def train_ppo_variant(name, make_env_fn, num_envs, rollout_steps, max_episodes, lr=3e-4):
    """
    训练一个 PPO 变体。

    Args:
        name: 用于日志和保存的名称
        make_env_fn: 环境工厂 (lambda: env)
        max_episodes: 训练轮数
    Returns: 训练好的 PPO agent
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"训练 {name} | {max_episodes}ep | {num_envs}envs | {rollout_steps}steps")
    logger.info(f"{'='*70}")

    env = SyncVectorEnv([make_env_fn for _ in range(num_envs)])
    eval_env = make_env_fn()

    agent = PPO(
        state_dim=14, action_dim=3, action_max=1.0,
        lr=lr, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
        epochs=10, minibatch_size=64, hidden_dim=128,
        use_adaptive_entropy=True, num_envs=num_envs,
    )

    total_steps = 0
    start_time = time.time()
    best_reward = -float('inf')
    episode_rewards = []

    try:
        states = env.reset()[0]

        for ep in range(max_episodes):
            ep_rewards = np.zeros(num_envs)
            ep_lengths = np.zeros(num_envs)

            for step in range(rollout_steps):
                total_steps += num_envs

                actions, log_probs_list, values_list, entropies_list = [], [], [], []
                for i in range(num_envs):
                    a, lp, v, ent = agent.get_action(states[i], deterministic=False)
                    actions.append(a); log_probs_list.append(lp)
                    values_list.append(v); entropies_list.append(ent)

                actions_np = np.array(actions)
                next_states, rewards, terminateds, truncateds, infos = env.step(actions_np)
                dones = np.logical_or(terminateds, truncateds)

                for i in range(num_envs):
                    agent.store_transition(
                        state=states[i], action=actions_np[i],
                        reward=float(rewards[i]), next_state=next_states[i],
                        done=bool(dones[i]), log_prob=log_probs_list[i],
                        value=values_list[i], entropy=entropies_list[i],
                    )
                    ep_rewards[i] += rewards[i]
                    ep_lengths[i] += 1

                states = next_states

            # LR decay
            progress = ep / max_episodes
            agent.set_lr(lr * (1 - progress))

            # PPO update
            update_result = agent.update()

            avg_reward = np.mean(ep_rewards)
            episode_rewards.append(avg_reward)

            if ep % 20 == 0 or ep == max_episodes - 1:
                elapsed = time.time() - start_time
                eta = elapsed / (ep + 1) * (max_episodes - ep - 1) if ep > 0 else 0
                logger.info(f"Ep {ep+1:4d}/{max_episodes} | "
                           f"reward={avg_reward:8.2f} | loss={update_result['total_loss']:.2f} | "
                           f"entropy={update_result['entropy']:.3f} | "
                           f"elapsed={elapsed:.0f}s | eta={eta:.0f}s")

                sys.stdout.flush()

            if avg_reward > best_reward:
                best_reward = avg_reward

        total_time = time.time() - start_time
        logger.info(f"训练完成: {total_time:.0f}s | best_reward={best_reward:.2f}")

    except KeyboardInterrupt:
        logger.info("训练被中断")
    finally:
        env.close()
        eval_env.close()

    return agent


# ============================================================
# SAC Training
# ============================================================
def train_sac(max_episodes=500, lr=1e-4):
    """训练 SAC agent 在标准 DroneEnv 上"""
    name = "SAC-v2 (clean env)"
    logger.info(f"\n{'='*70}")
    logger.info(f"训练 {name} | {max_episodes}ep")
    logger.info(f"{'='*70}")

    env = DroneEnv()
    agent = SAC(state_dim=14, action_dim=3, hidden_dim=128, lr=lr, batch_size=256, buffer_size=100000)

    start_time = time.time()
    episode_rewards = []
    total_steps = 0

    for ep in range(max_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        while True:
            action, _ = agent.get_action(state, deterministic=False)
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)

            # SAC: 每步都做更新 (off-policy)
            update_result = agent.update()

            state = next_state
            ep_reward += reward
            ep_steps += 1
            total_steps += 1

            if done:
                break

        # LR decay
        progress = ep / max_episodes
        agent.set_lr(lr * (1 - progress))

        episode_rewards.append(ep_reward)

        if ep % 20 == 0 or ep == max_episodes - 1:
            elapsed = time.time() - start_time
            recent_avg = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            logger.info(f"Ep {ep+1:4d}/{max_episodes} | "
                       f"reward={ep_reward:8.2f} | recent20={recent_avg:8.2f} | "
                       f"steps={ep_steps:4d} | alpha={update_result['alpha']:.3f} | "
                       f"critic_loss={update_result['critic_loss']:.3f} | "
                       f"elapsed={elapsed:.0f}s")

            sys.stdout.flush()

    total_time = time.time() - start_time
    logger.info(f"SAC训练完成: {total_time:.0f}s | final_reward={np.mean(episode_rewards[-20:]):.2f}")
    env.close()

    return agent


# ============================================================
# 评估
# ============================================================
def evaluate_agent(agent, env, num_episodes=50, agent_type='ppo', seed=42):
    """评估单个 agent, 返回汇总指标"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = []
    for _ in range(num_episodes):
        state, info = env.reset()
        ep_reward, ep_steps, path_length = 0.0, 0, 0.0
        prev_pos = state[:3].copy()
        min_obs_dist = float('inf')

        while True:
            if agent_type == 'sac':
                action = agent.select_action(state, deterministic=True)
            else:
                action = agent.select_action(state, deterministic=True)

            next_state, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)
            ep_steps += 1
            path_length += float(np.linalg.norm(next_state[:3] - prev_pos))
            prev_pos = next_state[:3]

            if 'current_pos' in info:
                obs_dist = env._get_min_obstacle_distance(info['current_pos'])
                min_obs_dist = min(min_obs_dist, obs_dist)

            if terminated or truncated:
                break
            state = next_state

        results.append({
            'success': bool(info.get('reached_target', False)),
            'collision': bool(info.get('collision', False)),
            'timeout': not info.get('reached_target', False) and not info.get('collision', False),
            'reward': float(ep_reward),
            'steps': ep_steps,
            'path_length': float(path_length),
            'min_obs_dist': float(min_obs_dist),
        })

    n = len(results)
    successes = sum(1 for r in results if r['success'])
    collisions = sum(1 for r in results if r['collision'])
    timeouts = sum(1 for r in results if r['timeout'])
    rewards = [r['reward'] for r in results]
    success_steps = [r['steps'] for r in results if r['success']]
    path_lengths = [r['path_length'] for r in results]

    return {
        'success_rate': successes / n * 100,
        'collision_rate': collisions / n * 100,
        'timeout_rate': timeouts / n * 100,
        'avg_reward': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'avg_steps': float(np.mean(success_steps)) if success_steps else None,
        'avg_path_length': float(np.mean(path_lengths)),
        'avg_min_obs_dist': float(np.mean([r['min_obs_dist'] for r in results])),
        'num_episodes': n,
    }


def run_comparison_eval(models_dict, noise_levels=[0, 0.1, 0.5, 1.0, 2.0], num_episodes=50):
    """
    四组对比评估: 每个模型在5个噪声水平下各跑50次

    models_dict: {name: (agent, agent_type)}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"\n{'='*70}")
    logger.info(f"A2 四组对比评估 | {len(models_dict)} models × {len(noise_levels)} levels × {num_episodes}ep")
    logger.info(f"{'='*70}")

    rows = []
    for model_name, (agent, agent_type) in models_dict.items():
        for sigma in noise_levels:
            env = NoisyDroneEnv.from_pattern('full', sigma=sigma) if sigma > 0 else DroneEnv()
            label = f"{model_name}_full_s{sigma}"
            logger.info(f"  评估 {label}...")
            result = evaluate_agent(agent, env, num_episodes=num_episodes, agent_type=agent_type)
            rows.append({
                'model': model_name,
                'noise_sigma': sigma,
                **result,
            })
            env.close()
            logger.info(f"    success={result['success_rate']:.1f}% collision={result['collision_rate']:.1f}%")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f'a2_comparison_{timestamp}.csv')
    fields = ['model', 'noise_sigma', 'num_episodes', 'success_rate', 'collision_rate', 'timeout_rate',
              'avg_reward', 'reward_std', 'avg_steps', 'avg_path_length', 'avg_min_obs_dist']
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, f'a2_comparison_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'timestamp': timestamp, 'results': rows}, f, indent=2, ensure_ascii=False)

    # Print table
    print_comparison_table(rows)

    logger.info(f"\n[CSV] {csv_path}")
    logger.info(f"[JSON] {json_path}")
    return rows


def print_comparison_table(rows):
    """打印对比表"""
    print(f"\n{'='*90}")
    print(f"  A2 四组对比评估 — 成功率 (%)")
    print(f"{'='*90}")

    models = sorted(set(r['model'] for r in rows))
    sigmas = sorted(set(r['noise_sigma'] for r in rows))

    # Header
    header = f"  {'Model':<15}"
    for s in sigmas:
        header += f" {'sigma='+str(s):>10}"
    print(header)
    print(f"  {'-'*15}{' '.join(['-'*10 for _ in sigmas])}")

    for model in models:
        line = f"  {model:<15}"
        for s in sigmas:
            match = [r for r in rows if r['model'] == model and r['noise_sigma'] == s]
            if match:
                line += f" {match[0]['success_rate']:9.1f}%"
            else:
                line += f" {'N/A':>10}"
        print(line)

    # 碰撞率表
    print(f"\n  A2 四组对比评估 — 碰撞率 (%)")
    header2 = f"  {'Model':<15}"
    for s in sigmas:
        header2 += f" {'sigma='+str(s):>10}"
    print(header2)
    print(f"  {'-'*15}{' '.join(['-'*10 for _ in sigmas])}")
    for model in models:
        line = f"  {model:<15}"
        for s in sigmas:
            match = [r for r in rows if r['model'] == model and r['noise_sigma'] == s]
            if match:
                line += f" {match[0]['collision_rate']:9.1f}%"
            else:
                line += f" {'N/A':>10}"
        print(line)
    print(f"{'='*90}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='A2 鲁棒性训练 + 对比评估')
    parser.add_argument('--skip-training', action='store_true', help='跳过训练, 仅评估已有模型')
    parser.add_argument('--train-only', type=str, default=None,
                        choices=['fixed', 'rand', 'curric', 'sac', 'all'],
                        help='仅训练指定变体')
    parser.add_argument('--episodes', type=int, default=500, help='训练轮数 (默认500)')
    parser.add_argument('--num-envs', type=int, default=4, help='PPO并行环境数 (默认4)')
    parser.add_argument('--fast', action='store_true', help='快速模式: 50ep训练 + 10ep评估')
    args = parser.parse_args()

    max_ep = 50 if args.fast else args.episodes
    num_envs = args.num_envs
    eval_eps = 10 if args.fast else 50
    rollout_steps = 2048 // num_envs

    models_available = {}

    # ==================== Training ====================
    if not args.skip_training:
        train_targets = [args.train_only] if args.train_only and args.train_only != 'all' else ['fixed', 'rand', 'curric', 'sac']

        # --- A2-Fixed ---
        if 'fixed' in train_targets:
            logger.info("\n" + "#"*70)
            logger.info("# A2-Fixed: full noise sigma=0.5 (per-step)")
            logger.info("#"*70)
            agent = train_ppo_variant(
                "A2-Fixed",
                make_env_fn=lambda: NoisyDroneEnv.from_pattern('full', sigma=0.5),
                num_envs=num_envs, rollout_steps=rollout_steps, max_episodes=max_ep,
            )
            agent.save_model(A2_MODELS['fixed'])

        # --- A2-Rand ---
        if 'rand' in train_targets:
            logger.info("\n" + "#"*70)
            logger.info("# A2-Rand: full noise sigma~U(0,2.0) (per-episode固定)")
            logger.info("#"*70)
            agent = train_ppo_variant(
                "A2-Rand",
                make_env_fn=lambda: PerEpisodeNoisyEnv(mode='random', sigma_min=0.0, sigma_max=2.0),
                num_envs=num_envs, rollout_steps=rollout_steps, max_episodes=max_ep,
            )
            agent.save_model(A2_MODELS['rand'])

        # --- A2-Curric ---
        if 'curric' in train_targets:
            logger.info("\n" + "#"*70)
            logger.info("# A2-Curric: full noise sigma 0->2.0 线性增长")
            logger.info("#"*70)

            # Curriculum: 需要每个episode更新进度
            # 使用单环境以精确控制curriculum进度
            curric_env = PerEpisodeNoisyEnv(mode='curric', sigma_min=0.0, sigma_max=2.0)
            eval_env_curric = PerEpisodeNoisyEnv(mode='curric', sigma_min=0.0, sigma_max=2.0)

            agent = PPO(
                state_dim=14, action_dim=3, action_max=1.0,
                lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                epochs=10, minibatch_size=64, hidden_dim=128,
                use_adaptive_entropy=True, num_envs=1,
            )

            start_time = time.time()
            total_steps = 0

            for ep in range(max_ep):
                progress = ep / max_ep
                curric_env.set_curric_progress(progress)
                state, _ = curric_env.reset()
                ep_reward, ep_length = 0.0, 0

                for step in range(512):  # 单环境 rollout
                    total_steps += 1
                    a, lp, v, ent = agent.get_action(state, deterministic=False)
                    next_state, reward, terminated, truncated, info = curric_env.step(a)
                    done = terminated or truncated

                    agent.store_transition(state=state, action=a, reward=float(reward),
                                          next_state=next_state, done=done,
                                          log_prob=lp, value=v, entropy=ent)
                    ep_reward += float(reward)
                    ep_length += 1
                    state = next_state
                    if done:
                        # 不break, 继续收集直到rollout填满
                        state, _ = curric_env.reset()

                lr = 3e-4 * (1 - progress)
                agent.set_lr(lr)
                update_result = agent.update()

                if ep % 20 == 0 or ep == max_ep - 1:
                    elapsed = time.time() - start_time
                    logger.info(f"Ep {ep+1:4d}/{max_ep} | sigma={curric_env.current_sigma:.3f} | "
                               f"reward={ep_reward:8.2f} | loss={update_result['total_loss']:.2f} | "
                               f"elapsed={elapsed:.0f}s")

                sys.stdout.flush()

            logger.info(f"A2-Curric完成: {time.time()-start_time:.0f}s")
            agent.save_model(A2_MODELS['curric'])
            curric_env.close()

        # --- SAC ---
        if 'sac' in train_targets:
            logger.info("\n" + "#"*70)
            logger.info("# SAC: 标准DroneEnv, 纯14D向量")
            logger.info("#"*70)
            sac_agent = train_sac(max_episodes=max_ep)
            sac_agent.save_model(A2_MODELS['sac'])

    # ==================== Loading Models ====================
    logger.info("\n" + "="*70)
    logger.info("加载模型用于评估...")
    logger.info("="*70)

    # Clean baseline
    clean_ppo = PPO(state_dim=14, action_dim=3, action_max=1.0,
                    lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                    epochs=10, minibatch_size=64, hidden_dim=128,
                    use_adaptive_entropy=True, num_envs=1)
    if os.path.exists(BASELINE_MODEL):
        clean_ppo.load_model(BASELINE_MODEL)
        models_available['Clean(Baseline)'] = (clean_ppo, 'ppo')
        logger.info("Clean(Baseline) 已加载")
    else:
        logger.warning(f"Baseline模型未找到: {BASELINE_MODEL}")

    # Load trained A2 models
    for key, name in [('fixed', 'A2-Fixed'), ('rand', 'A2-Rand'), ('curric', 'A2-Curric')]:
        path = A2_MODELS[key]
        if os.path.exists(path):
            ppo = PPO(state_dim=14, action_dim=3, action_max=1.0,
                      lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                      epochs=10, minibatch_size=64, hidden_dim=128,
                      use_adaptive_entropy=True, num_envs=1)
            ppo.load_model(path)
            models_available[name] = (ppo, 'ppo')
            logger.info(f"{name} 已加载")
        else:
            logger.warning(f"{name} 模型未找到: {path}")

    # Load SAC
    sac_path = A2_MODELS['sac']
    if os.path.exists(sac_path):
        sac = SAC(state_dim=14, action_dim=3, hidden_dim=128)
        sac.load_model(sac_path)
        models_available['SAC(Clean)'] = (sac, 'sac')
        logger.info("SAC(Clean) 已加载")

    # ==================== Comparison Evaluation ====================
    if models_available:
        noise_levels = [0, 0.1, 0.5, 1.0, 2.0]
        run_comparison_eval(models_available, noise_levels=noise_levels, num_episodes=eval_eps)
    else:
        logger.error("无可用模型, 评估终止")

    logger.info("\n[DONE] A2 pipeline 完成")


if __name__ == '__main__':
    main()
