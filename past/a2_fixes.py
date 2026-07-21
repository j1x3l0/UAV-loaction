"""
A2 修复脚本 — 针对 Week 1-3 复盘中的三个问题
==============================================
Fix 1: Curric v2 — 从预训练baseline开始 + 更慢的增长曲线
Fix 2: SAC v2   — 2000ep + lr=3e-4 + warm-start buffer
Fix 3: Rand v2  — 窄范围 U(0, 1.0) 减少clean-env性能损失

运行后自动做四组对比评估 (v1 models vs v2 fixes)

Usage:
  python a2_fixes.py                     # 全部fixes
  python a2_fixes.py --fix curric        # 仅Curric
  python a2_fixes.py --fix sac           # 仅SAC
  python a2_fixes.py --skip-training     # 仅评估
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
from drone_env_noisy import NoisyDroneEnv
from ppo_agent import PPO, DEVICE
from sac_agent_v2 import SAC
from a2_train import (PerEpisodeNoisyEnv, evaluate_agent, run_comparison_eval,
                       A2_MODELS, BASELINE_MODEL, RESULTS_DIR, MODEL_DIR)

# ============================================================
# Fix 1: Curric v2 — Pre-trained start + slower growth
# ============================================================
def fix_curric_v2(max_episodes=500):
    """
    Curric v2 改进:
      a) 从预训练baseline模型开始 (而非从头训练)
      b) 增长曲线改为指数型 (前期慢, 后期快)
         sigma(t) = sigma_max * (t/T)^2  (二次函数, 前半段非常慢)
      c) 降低clip_eps后期值或者使用更小学习率
    """
    name = "A2-Curric-v2"
    model_path = os.path.join(MODEL_DIR, 'a2_curric_v2_500ep')

    logger.info(f"\n{'#'*70}")
    logger.info(f"# {name}: 预训练起点 + 二次增长曲线 σ_max*(t/T)^2")
    logger.info(f"{'#'*70}")

    curric_env = PerEpisodeNoisyEnv(mode='curric', sigma_min=0.0, sigma_max=2.0)

    agent = PPO(
        state_dim=14, action_dim=3, action_max=1.0,
        lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
        epochs=10, minibatch_size=64, hidden_dim=128,
        use_adaptive_entropy=True, num_envs=1,
    )
    # 【核心修复】加载预训练baseline权重
    agent.load_model(BASELINE_MODEL)
    logger.info(f"已加载预训练模型: {BASELINE_MODEL}")

    start_time = time.time()
    episode_rewards = []

    for ep in range(max_episodes):
        # 【核心修复】二次增长曲线: 前半段极慢
        progress = (ep / max_episodes) ** 2  # 0→0.25→1.0 (前250轮仅到25%)
        curric_env.set_curric_progress(progress)
        state, _ = curric_env.reset()
        ep_reward, ep_length = 0.0, 0

        for step in range(512):
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
                state, _ = curric_env.reset()

        lr = 3e-4 * (1 - progress)  # 也跟随progress衰减
        agent.set_lr(lr)
        update_result = agent.update()
        episode_rewards.append(ep_reward)

        if ep % 25 == 0 or ep == max_episodes - 1:
            elapsed = time.time() - start_time
            recent_avg = np.mean(episode_rewards[-25:]) if len(episode_rewards) >= 25 else np.mean(episode_rewards)
            logger.info(f"Ep {ep+1:4d}/{max_episodes} | sigma={curric_env.current_sigma:.3f} | "
                       f"progress={progress:.3f} | reward={ep_reward:8.2f} | recent25={recent_avg:8.2f} | "
                       f"loss={update_result['total_loss']:.2f} | elapsed={elapsed:.0f}s")
            sys.stdout.flush()

    total_time = time.time() - start_time
    logger.info(f"{name}完成: {total_time:.0f}s")
    agent.save_model(model_path)
    curric_env.close()

    return model_path


# ============================================================
# Fix 2: SAC v2 — 2000ep + tuned hyperparams
# ============================================================
def fix_sac_v2(max_episodes=2000):
    """
    SAC v2 改进:
      a) 增加到 2000 轮
      b) lr=3e-4 (从1e-4提高)
      c) batch_size=512 (更快消耗buffer)
      d) buffer_size=50000 (更快替换早期差数据)
      e) 每步做2次更新 (UTD ratio=2)
    """
    name = "SAC-v2"
    model_path = os.path.join(MODEL_DIR, 'sac_v2_2000ep')

    logger.info(f"\n{'#'*70}")
    logger.info(f"# {name}: {max_episodes}ep, lr=3e-4, batch=512, UTD=2")
    logger.info(f"{'#'*70}")

    env = DroneEnv()
    agent = SAC(state_dim=14, action_dim=3, hidden_dim=128,
                lr=3e-4, gamma=0.99, tau=0.005,
                buffer_size=50000, batch_size=512, initial_alpha=0.2)

    start_time = time.time()
    episode_rewards = []
    total_steps = 0
    n_updates_per_step = 2  # UTD ratio

    for ep in range(max_episodes):
        state, _ = env.reset()
        ep_reward, ep_steps = 0.0, 0

        while True:
            action, _ = agent.get_action(state, deterministic=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)

            # Multiple updates per step
            update_result = {}
            for _ in range(n_updates_per_step):
                update_result = agent.update()

            state = next_state
            ep_reward += float(reward)
            ep_steps += 1
            total_steps += 1

            if done:
                break

        progress = ep / max_episodes
        agent.set_lr(3e-4 * (1 - progress))
        episode_rewards.append(ep_reward)

        if ep % 50 == 0 or ep == max_episodes - 1:
            elapsed = time.time() - start_time
            recent_avg = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            recent_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else recent_avg
            logger.info(f"Ep {ep+1:5d}/{max_episodes} | "
                       f"reward={ep_reward:8.2f} | recent50={recent_avg:8.2f} | "
                       f"recent100={recent_100:8.2f} | steps={ep_steps:4d} | "
                       f"alpha={update_result.get('alpha', 0):.3f} | "
                       f"critic_loss={update_result.get('critic_loss', 0):.3f} | "
                       f"elapsed={elapsed:.0f}s")
            sys.stdout.flush()

    total_time = time.time() - start_time
    final_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    logger.info(f"{name}完成: {total_time:.0f}s | final100_avg_reward={final_avg:.2f}")
    agent.save_model(model_path)
    env.close()

    return model_path


# ============================================================
# Fix 3: Rand v2 — 窄范围 U(0, 1.0)
# ============================================================
def fix_rand_v2(max_episodes=500):
    """
    Rand v2 改进:
      缩小随机化范围: U(0, 1.0) 替代 U(0, 2.0)
      → 期望能在clean env下保持更高性能
    """
    name = "A2-Rand-v2"
    model_path = os.path.join(MODEL_DIR, 'a2_rand_v2_u0_1.0_500ep')

    logger.info(f"\n{'#'*70}")
    logger.info(f"# {name}: σ~U(0, 1.0), {max_episodes}ep")
    logger.info(f"{'#'*70}")

    num_envs = 4
    rollout_steps = 512
    env = SyncVectorEnv([lambda: PerEpisodeNoisyEnv(mode='random', sigma_min=0.0, sigma_max=1.0)
                         for _ in range(num_envs)])
    eval_env = PerEpisodeNoisyEnv(mode='random', sigma_min=0.0, sigma_max=1.0)

    agent = PPO(
        state_dim=14, action_dim=3, action_max=1.0,
        lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
        epochs=10, minibatch_size=64, hidden_dim=128,
        use_adaptive_entropy=True, num_envs=num_envs,
    )

    start_time = time.time()
    total_steps = 0
    best_reward = -float('inf')

    try:
        states = env.reset()[0]
        for ep in range(max_episodes):
            ep_rewards = np.zeros(num_envs)

            for step in range(rollout_steps):
                total_steps += num_envs
                actions, lps, vs, ents = [], [], [], []
                for i in range(num_envs):
                    a, lp, v, ent = agent.get_action(states[i], deterministic=False)
                    actions.append(a); lps.append(lp); vs.append(v); ents.append(ent)

                actions_np = np.array(actions)
                next_states, rewards, terminateds, truncateds, infos = env.step(actions_np)
                dones = np.logical_or(terminateds, truncateds)

                for i in range(num_envs):
                    agent.store_transition(
                        state=states[i], action=actions_np[i],
                        reward=float(rewards[i]), next_state=next_states[i],
                        done=bool(dones[i]), log_prob=lps[i], value=vs[i], entropy=ents[i])
                    ep_rewards[i] += rewards[i]

                states = next_states

            progress = ep / max_episodes
            agent.set_lr(3e-4 * (1 - progress))
            update_result = agent.update()
            avg_reward = np.mean(ep_rewards)

            if avg_reward > best_reward:
                best_reward = avg_reward

            if ep % 20 == 0 or ep == max_episodes - 1:
                elapsed = time.time() - start_time
                logger.info(f"Ep {ep+1:4d}/{max_episodes} | reward={avg_reward:8.2f} | "
                           f"loss={update_result['total_loss']:.2f} | elapsed={elapsed:.0f}s")
                sys.stdout.flush()

    finally:
        env.close()
        eval_env.close()

    logger.info(f"{name}完成: {time.time()-start_time:.0f}s | best_reward={best_reward:.2f}")
    agent.save_model(model_path)
    return model_path


# ============================================================
# Load all models for comparison
# ============================================================
def load_all_models():
    """加载 v1 + v2 所有模型"""
    models = {}

    # v1 models (from a2_train.py)
    for key, name in [('fixed', 'A2-Fixed-v1'), ('rand', 'A2-Rand-v1'),
                       ('curric', 'A2-Curric-v1')]:
        path = A2_MODELS[key]
        if os.path.exists(path):
            ppo = PPO(state_dim=14, action_dim=3, action_max=1.0,
                      lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                      epochs=10, minibatch_size=64, hidden_dim=128,
                      use_adaptive_entropy=True, num_envs=1)
            ppo.load_model(path)
            models[name] = (ppo, 'ppo')

    # SAC v1
    sac_v1_path = A2_MODELS['sac']
    if os.path.exists(sac_v1_path):
        sac = SAC(state_dim=14, action_dim=3, hidden_dim=128)
        sac.load_model(sac_v1_path)
        models['SAC-v1'] = (sac, 'sac')

    # Clean baseline
    clean = PPO(state_dim=14, action_dim=3, action_max=1.0,
                lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                epochs=10, minibatch_size=64, hidden_dim=128,
                use_adaptive_entropy=True, num_envs=1)
    clean.load_model(BASELINE_MODEL)
    models['Clean(Baseline)'] = (clean, 'ppo')

    # v2 models
    v2_paths = {
        'A2-Curric-v2': os.path.join(MODEL_DIR, 'a2_curric_v2_500ep'),
        'SAC-v2': os.path.join(MODEL_DIR, 'sac_v2_2000ep'),
        'A2-Rand-v2': os.path.join(MODEL_DIR, 'a2_rand_v2_u0_1.0_500ep'),
    }
    for name, path in v2_paths.items():
        if os.path.exists(path):
            if 'SAC' in name:
                agent = SAC(state_dim=14, action_dim=3, hidden_dim=128)
                agent.load_model(path)
                models[name] = (agent, 'sac')
            else:
                ppo = PPO(state_dim=14, action_dim=3, action_max=1.0,
                          lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                          epochs=10, minibatch_size=64, hidden_dim=128,
                          use_adaptive_entropy=True, num_envs=1)
                ppo.load_model(path)
                models[name] = (ppo, 'ppo')

    return models


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='A2 修复脚本')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--fix', type=str, default='all',
                        choices=['curric', 'sac', 'rand', 'all'])
    parser.add_argument('--fast', action='store_true',
                        help='快速测试: 50ep训练 + 10ep评估')
    args = parser.parse_args()

    train_eps = 50 if args.fast else 500
    sac_eps = 50 if args.fast else 2000
    eval_eps = 10 if args.fast else 50

    if not args.skip_training:
        if args.fix in ('curric', 'all'):
            fix_curric_v2(max_episodes=train_eps)
        if args.fix in ('sac', 'all'):
            fix_sac_v2(max_episodes=sac_eps)
        if args.fix in ('rand', 'all'):
            fix_rand_v2(max_episodes=train_eps)

    # Evaluation
    models = load_all_models()
    if models:
        logger.info(f"\n加载了 {len(models)} 个模型: {list(models.keys())}")
        noise_levels = [0, 0.1, 0.5, 1.0, 2.0]
        run_comparison_eval(models, noise_levels=noise_levels, num_episodes=eval_eps)
    else:
        logger.error("无可用模型")

    logger.info("\n[DONE] A2 修复脚本完成")


if __name__ == '__main__':
    main()
