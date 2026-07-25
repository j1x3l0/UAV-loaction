"""
train_visual.py — 视觉PPO训练管线 v2

架构位置: scripts/ (Application层)
WHY 这个设计:
  - 基于v1 train.py改造，适配VisualDroneEnv + VisualPPO
  - 支持单env (smoke test) 和多env (正式训练)
  - 支持命令行控制退化配置
数据流: VisualDroneEnv → Dict{depth,vec} → VisualPPO → action → env.step
边界: 不负责模型架构、不负责环境实现、不负责评估
风险: 3DGS渲染是瓶颈 → 正式版需多env并行 + GPU加速

用法:
  python scripts/train_visual.py --episodes 20             # 冒烟测试 (mock)
  python scripts/train_visual.py --episodes 3000 --envs 8  # mock 训练
  python scripts/train_visual.py --episodes 3000 --envs 8 \
      --renderer gsplat --ply data/gs_data/ply_exports/gate_mid_new_gs.ply  # 真实3DGS
  python scripts/train_visual.py --episodes 3000 --degradation rand  # V3-Rand
"""

import numpy as np
import torch
import os, sys, time, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 项目根 (rlproject-swift-improved 的上一级), 用于解析 data/ 下的相对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)

from envs.visual_drone_env import VisualDroneEnv
from core.visual_ppo_agent import VisualPPO
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_time(s):
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s//60:.0f}m{s%60:.0f}s"
    return f"{s//3600:.0f}h{(s%3600)//60:.0f}m"


def evaluate_model(agent, env, eval_episodes=50):
    """评估: 成功率/碰撞率/平均奖励/平均步数"""
    successes = collisions = timeouts = 0
    rewards_list = []
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                if info.get('reached_target'): successes += 1
                elif info.get('collision'): collisions += 1
                else: timeouts += 1
                break
        rewards_list.append(ep_reward)

    return {
        'success_rate': successes / eval_episodes * 100,
        'collision_rate': collisions / eval_episodes * 100,
        'timeout_rate': timeouts / eval_episodes * 100,
        'avg_reward': np.mean(rewards_list),
        'reward_std': np.std(rewards_list),
    }


def resolve_ply_path(ply_arg):
    """
    解析 .ply 路径: 支持绝对路径、cwd 相对路径、或 repo data/ 下的相对路径。
    WHY: 文档命令 `--ply data/gs_data/...` 是相对 repo 根的, 但训练从
         rlproject-swift-improved/ 启动, 直接相对 cwd 会找不到文件。
    """
    if os.path.exists(ply_arg):
        return ply_arg
    alt = os.path.join(REPO_ROOT, ply_arg)
    if os.path.exists(alt):
        return alt
    # 最后尝试只看文件名 (data/gs_data/ply_exports/<name>)
    alt2 = os.path.join(REPO_ROOT, 'data', 'gs_data', 'ply_exports',
                        os.path.basename(ply_arg))
    if os.path.exists(alt2):
        return alt2
    raise FileNotFoundError(
        f"ply not found: {ply_arg} (tried cwd-relative, repo-relative, ply_exports)")


def make_env(degradation_config=None, renderer='mock', ply_path=None):
    """环境工厂"""
    cfg = {'renderer': renderer}
    if renderer == 'gsplat':
        if not ply_path:
            raise ValueError("--renderer gsplat requires --ply <path to .ply>")
        cfg['ply_path'] = resolve_ply_path(ply_path)
    if degradation_config and degradation_config != 'clean':
        # 简化退化配置 (Phase 0冒烟测试用)
        if degradation_config == 'rand':
            level = np.random.choice([100, 50, 25, 10, 5])
            cfg['degradation'] = {'resolution': max(16, int(64 * level / 100))}
    return VisualDroneEnv(config=cfg)


def train_visual(config):
    """视觉PPO训练主循环"""
    num_envs = config.get('num_envs', 2)
    rollout_steps = config.get('rollout_steps', 256)
    max_episodes = config.get('max_episodes', 3000)
    eval_interval = config.get('eval_interval', 50)
    eval_episodes = config.get('eval_episodes', 20)

    degradation = config.get('degradation', 'clean')
    renderer = config.get('renderer', 'mock')
    ply_path = config.get('ply_path')

    # 创建环境
    envs = [make_env(degradation, renderer, ply_path) for _ in range(num_envs)]
    eval_env = make_env('clean', renderer, ply_path)

    # 创建PPO (per-env rollout 需要调整num_envs)
    ppo = VisualPPO(
        vec_dim=6, action_dim=3, action_max=1.0,
        lr=config.get('lr', 3e-4),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_eps=config.get('clip_eps', 0.2),
        epochs=config.get('epochs', 5),
        minibatch_size=config.get('minibatch_size', 32),
        hidden_dim=config.get('hidden_dim', 128),
        use_adaptive_entropy=True,
        num_envs=num_envs,
    )

    # 重置环境
    observations = [env.reset()[0] for env in envs]
    step_count = 0
    start_time = time.time()
    best_sr = 0.0

    logger.info(f"Training: {max_episodes}ep × {num_envs}envs × "
                f"{rollout_steps}steps, degradation={degradation}, "
                f"renderer={renderer}")

    for episode in range(max_episodes):
        # Rollout
        for step in range(rollout_steps):
            step_count += num_envs

            actions = []; log_probs = []; values = []; entropies = []
            for i in range(num_envs):
                a, lp, v, ent = ppo.get_action(observations[i])
                actions.append(a); log_probs.append(lp)
                values.append(v); entropies.append(ent)

            for i in range(num_envs):
                next_obs, reward, terminated, truncated, info = envs[i].step(actions[i])
                done = terminated or truncated
                ppo.store_transition(observations[i], actions[i], reward,
                                     next_obs, done, log_probs[i], values[i],
                                     entropies[i])
                observations[i] = next_obs
                if done:
                    observations[i], _ = envs[i].reset()

        # LR衰减
        progress = episode / max_episodes
        ppo.set_lr(config.get('lr', 3e-4) * (1 - progress))

        # PPO update
        result = ppo.update()

        # 日志
        if episode % config.get('print_interval', 10) == 0 or episode == max_episodes - 1:
            elapsed = time.time() - start_time
            fps = step_count / elapsed if elapsed > 0 else 0
            eta = format_time((max_episodes - episode - 1) * elapsed / (episode + 1)
                              if elapsed > 0 else 0)
            logger.info(f"Ep {episode + 1}/{max_episodes} | "
                        f"steps: {step_count} | fps: {fps:.0f} | "
                        f"loss: {result['total_loss']:.3f} | "
                        f"actor: {result['actor_loss']:.3f} | "
                        f"critic: {result['critic_loss']:.3f} | "
                        f"entropy: {result['entropy']:.2f} | "
                        f"lr: {ppo.current_lr:.2e} | eta: {eta}")

        # 评估
        if episode % eval_interval == 0 or episode == max_episodes - 1:
            eval_result = evaluate_model(ppo, eval_env, eval_episodes)
            logger.info(f"  Eval: SR={eval_result['success_rate']:.1f}% | "
                        f"CR={eval_result['collision_rate']:.1f}% | "
                        f"avgR={eval_result['avg_reward']:.2f}")
            if eval_result['success_rate'] > best_sr:
                best_sr = eval_result['success_rate']
                ppo.save_model(f"saved_models/visual_ppo_best.pth")
                logger.info(f"  New best model (SR={best_sr:.1f}%)")

    elapsed = time.time() - start_time
    logger.info(f"Training done | time={format_time(elapsed)} | "
                f"best_SR={best_sr:.1f}% | steps={step_count}")
    return {'best_success_rate': best_sr, 'time': elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--envs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--degradation', type=str, default='clean',
                       choices=['clean', 'rand'])
    parser.add_argument('--rollout-steps', type=int, default=256)
    parser.add_argument('--renderer', type=str, default='mock',
                       choices=['mock', 'gsplat'])
    parser.add_argument('--ply', type=str, default=None,
                       help='3DGS .ply 路径 (--renderer gsplat 时必填)')
    args = parser.parse_args()

    config = {
        'max_episodes': args.episodes,
        'num_envs': args.envs,
        'lr': args.lr,
        'degradation': args.degradation,
        'rollout_steps': args.rollout_steps,
        'renderer': args.renderer,
        'ply_path': args.ply,
        'minibatch_size': 32,
        'epochs': 5,
        'eval_interval': max(5, args.episodes // 10),
        'eval_episodes': 20,
    }

    result = train_visual(config)
    logger.info(f"Final: best_SR={result['best_success_rate']:.1f}%")


if __name__ == "__main__":
    main()
