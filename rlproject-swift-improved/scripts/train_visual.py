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
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 项目根 (rlproject-swift-improved 的上一级), 用于解析 data/ 下的相对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)

from envs.visual_drone_env import VisualDroneEnv
from core.visual_ppo_agent import VisualPPO
from utils.metrics import wilson_confidence_interval
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_time(s):
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s//60:.0f}m{s%60:.0f}s"
    return f"{s//3600:.0f}h{(s%3600)//60:.0f}m"


def evaluate_model(agent, env, eval_episodes=100, base_seed=None):
    """评估: 成功率/碰撞率/平均奖励/平均步数 (Fix: ≥100ep 稳定评估)"""
    successes = collisions = timeouts = 0
    rewards_list = []
    episodes_detail = []  # per-episode 记录 (Fix 4)
    for ep_idx in range(eval_episodes):
        eval_seed = None if base_seed is None else base_seed + ep_idx
        obs, _ = env.reset(seed=eval_seed)
        ep_reward = 0.0
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                if info.get('reached_target'):
                    successes += 1
                    result_type = 'success'
                elif info.get('collision'):
                    collisions += 1
                    result_type = 'collision'
                else:
                    timeouts += 1
                    result_type = 'timeout'
                break
        rewards_list.append(ep_reward)
        episodes_detail.append({
            'episode': ep_idx,
            'result': result_type,
            'reward': float(ep_reward),
        })

    n = max(eval_episodes, 1)
    sr = successes / n
    sr_ci_low, sr_ci_high = wilson_confidence_interval(sr, n)

    return {
        'success_rate': sr * 100,
        'sr_ci_low': sr_ci_low,
        'sr_ci_high': sr_ci_high,
        'collision_rate': collisions / n * 100,
        'timeout_rate': timeouts / n * 100,
        'avg_reward': np.mean(rewards_list),
        'reward_std': np.std(rewards_list),
        'episodes_detail': episodes_detail,
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
    seed = config.get('seed', 0)
    model_out = config.get('model_out', 'saved_models/visual_ppo_best.pth')

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    observations = [env.reset(seed=seed + i)[0]
                    for i, env in enumerate(envs)]
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

            # 批量推理: 单次 CNN forward 处理所有环境 (P1-1 修复)
            actions, log_probs, values, entropies = \
                ppo.get_actions_batch(observations)

            for i in range(num_envs):
                next_obs, reward, terminated, truncated, info = envs[i].step(actions[i])
                done = terminated or truncated
                ppo.store_transition(observations[i], actions[i], reward,
                                     next_obs, done, float(log_probs[i]),
                                     float(values[i]), float(entropies[i]))
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
            eval_result = evaluate_model(
                ppo, eval_env, eval_episodes, base_seed=seed + 100_000)
            logger.info(f"  Eval: SR={eval_result['success_rate']:.1f}% "
                        f"(95%CI {eval_result['sr_ci_low']:.1f}–{eval_result['sr_ci_high']:.1f}) | "
                        f"CR={eval_result['collision_rate']:.1f}% | "
                        f"avgR={eval_result['avg_reward']:.2f}")
            if eval_result['success_rate'] > best_sr:
                best_sr = eval_result['success_rate']
                ppo.save_model(model_out)
                logger.info(f"  New best model (SR={best_sr:.1f}%)")

    elapsed = time.time() - start_time
    logger.info(f"Training done | time={format_time(elapsed)} | "
                f"best_SR={best_sr:.1f}% | steps={step_count}")
    return {'best_success_rate': best_sr, 'time': elapsed}


def fine_tune_visual(config):
    """
    Clean fine-tuning: 从 robust-best checkpoint 短程微调提升 clean SR

    架构位置: scripts/ (Application层)
    WHY 独立函数:
      - 加载已有 checkpoint 而非从头创建
      - 支持冻结 CNN 避免破坏尺度鲁棒性
      - 训练更短 + 评估更频 + gate check 验证
    数据流: checkpoint → VisualPPO → clean env → short training → gate eval
    边界: 不退化训练、不多 seed
    """
    num_envs = config.get('num_envs', 8)
    rollout_steps = config.get('rollout_steps', 256)
    max_updates = config.get('fine_tune_updates', 100)
    eval_interval = config.get('eval_interval', 10)
    eval_episodes = config.get('eval_episodes', 200)
    seed = config.get('seed', 0)
    freeze_cnn = config.get('freeze_cnn', True)
    checkpoint_path = config.get('checkpoint')
    model_out = config.get('model_out', 'saved_models/visual_ppo_finetuned.pth')
    lr = config.get('lr', 1e-4)
    run_gate = config.get('gate_check', False)
    renderer = config.get('renderer', 'gsplat')
    ply_path = config.get('ply_path')

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 创建 clean 环境
    envs = [make_env('clean', renderer, ply_path) for _ in range(num_envs)]
    eval_env = make_env('clean', renderer, ply_path)

    # 创建 PPO + 加载 checkpoint
    ppo = VisualPPO(
        vec_dim=6, action_dim=3, action_max=1.0,
        lr=lr,
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_eps=config.get('clip_eps', 0.2),
        epochs=config.get('epochs', 5),
        minibatch_size=config.get('minibatch_size', 32),
        hidden_dim=config.get('hidden_dim', 128),
        use_adaptive_entropy=True,
        num_envs=num_envs,
    )
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ppo.load_model(checkpoint_path)

    # 可选: 冻结 CNN 编码器
    if freeze_cnn:
        logger.info("Freezing CNN encoder — preserving scale robustness")
        for param in ppo.model.visual_encoder.parameters():
            param.requires_grad = False
        trainable = filter(lambda p: p.requires_grad, ppo.model.parameters())
        ppo.optimizer = torch.optim.Adam(trainable, lr=lr)
        n_frozen = sum(not p.requires_grad for p in ppo.model.parameters())
        n_total = sum(1 for _ in ppo.model.parameters())
        logger.info(f"Frozen: {n_frozen}/{n_total} param groups | "
                    f"Trainable: {n_total - n_frozen} groups")

    # Reset envs
    observations = [env.reset(seed=seed + i)[0] for i, env in enumerate(envs)]
    step_count = 0
    start_time = time.time()
    best_sr = 0.0
    best_update = 0

    logger.info(f"Fine-tuning: {max_updates} updates × {num_envs}envs × "
                f"{rollout_steps}steps | lr={lr:.1e} | freeze_cnn={freeze_cnn}")

    for update in range(max_updates):
        # Rollout
        for _ in range(rollout_steps):
            step_count += num_envs
            actions, log_probs, values, entropies = ppo.get_actions_batch(observations)
            for i in range(num_envs):
                next_obs, reward, terminated, truncated, info = envs[i].step(actions[i])
                done = terminated or truncated
                ppo.store_transition(observations[i], actions[i], reward,
                                     next_obs, done, float(log_probs[i]),
                                     float(values[i]), float(entropies[i]))
                observations[i] = next_obs
                if done:
                    observations[i], _ = envs[i].reset()

        # LR 衰减
        progress = update / max_updates
        ppo.set_lr(lr * (1 - progress))

        # PPO update
        result = ppo.update()

        # 日志
        if update % max(1, max_updates // 10) == 0 or update == max_updates - 1:
            elapsed = time.time() - start_time
            fps = step_count / elapsed if elapsed > 0 else 0
            logger.info(f"FT {update+1}/{max_updates} | "
                        f"loss={result['total_loss']:.3f} | "
                        f"entropy={result['entropy']:.2f} | "
                        f"coeff={result['entropy_coeff']:.4f} | "
                        f"fps={fps:.0f}")

        # 评估
        if update % eval_interval == 0 or update == max_updates - 1:
            eval_result = evaluate_model(ppo, eval_env, eval_episodes,
                                         base_seed=seed + 100_000)
            logger.info(f"  [FT Eval] SR={eval_result['success_rate']:.1f}% "
                        f"(95%CI {eval_result['sr_ci_low']:.1f}–"
                        f"{eval_result['sr_ci_high']:.1f}) | "
                        f"CR={eval_result['collision_rate']:.1f}%")
            if eval_result['success_rate'] > best_sr:
                best_sr = eval_result['success_rate']
                best_update = update + 1
                ppo.save_model(model_out)
                logger.info(f"  → New best (SR={best_sr:.1f}%) @ update {best_update}")

    elapsed = time.time() - start_time
    logger.info(f"Fine-tuning done | time={format_time(elapsed)} | "
                f"best_SR={best_sr:.1f}% @ update {best_update} | "
                f"model={model_out}")

    # Gate check
    gate_result = None
    if run_gate:
        logger.info("Running gate check after fine-tuning...")
        gate_result = run_gate_check(ppo, renderer, ply_path,
                                     eval_episodes=100,
                                     seed=seed + 200_000)

    return {'best_success_rate': best_sr, 'time': elapsed,
            'best_update': best_update, 'gate_check': gate_result}


def run_gate_check(agent, renderer='gsplat', ply_path=None,
                   depth_scales=None, eval_episodes=100, seed=20260729):
    """
    五尺度深度门控评估

    WHY: 验证 fine-tune 后 clean SR≥80% 的同时，
         其他尺度没有退化（≥70%），timeout≤10%。
    数据流: agent → depth_scale(1.0/0.75/0.5/0.25/0.1) × 100ep → metrics
    """
    if depth_scales is None:
        depth_scales = [1.0, 0.75, 0.5, 0.25, 0.1]

    env_config = {'renderer': renderer}
    if renderer == 'gsplat' and ply_path:
        env_config['ply_path'] = ply_path

    env = VisualDroneEnv(config=env_config)
    results = {}

    logger.info("=" * 60)
    logger.info(f"Gate Check: {len(depth_scales)} depth scales × "
                f"{eval_episodes} episodes")
    logger.info("-" * 60)

    for scale in depth_scales:
        successes = collisions = timeouts = 0
        rewards = []

        for ep in range(eval_episodes):
            obs, _ = env.reset(seed=seed + ep)
            obs['depth'] = obs['depth'] * scale
            ep_reward = 0.0

            while True:
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                obs['depth'] = obs['depth'] * scale  # 每步缩放深度
                ep_reward += reward

                if terminated or truncated:
                    if info.get('reached_target'):
                        successes += 1
                    elif info.get('collision'):
                        collisions += 1
                    else:
                        timeouts += 1
                    break

            rewards.append(ep_reward)

        n = eval_episodes
        sr_prop = successes / n
        sr_low, sr_high = wilson_confidence_interval(sr_prop, n)
        results[scale] = {
            'sr': sr_prop * 100,
            'sr_ci_low': sr_low * 100,
            'sr_ci_high': sr_high * 100,
            'cr': collisions / n * 100,
            'timeout': timeouts / n * 100,
            'avg_reward': float(np.mean(rewards)),
        }
        logger.info(f"  {scale:.2f}×: SR={results[scale]['sr']:.1f}% "
                    f"({sr_low*100:.1f}–{sr_high*100:.1f}) | "
                    f"CR={results[scale]['cr']:.1f}% "
                    f"TO={results[scale]['timeout']:.1f}%")

    env.close()

    # 门控判定
    clean_sr = results[1.0]['sr']
    min_sr = min(r['sr'] for r in results.values())
    max_to = max(r['timeout'] for r in results.values())

    criteria = [
        ('Clean SR ≥80%', clean_sr >= 80, f"{clean_sr:.1f}%"),
        ('All scales ≥70%', min_sr >= 70, f"min={min_sr:.1f}%"),
        ('Timeout ≤10%', max_to <= 10, f"max={max_to:.1f}%"),
    ]
    n_pass = sum(1 for _, p, _ in criteria if p)

    logger.info("-" * 60)
    logger.info(f"Gate: {n_pass}/{len(criteria)} passed")
    for desc, passed, val in criteria:
        logger.info(f"  {'✅' if passed else '❌'} {desc}: {val}")
    logger.info("=" * 60)

    return {'results': results, 'criteria': criteria,
            'n_passed': n_pass, 'total_criteria': len(criteria)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'fine-tune'],
                       help='train: 从头训练 | fine-tune: 从 checkpoint 继续微调')
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-out', type=str,
                        default='saved_models/visual_ppo_best.pth')
    # Fine-tune 参数
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='fine-tune 起始 checkpoint (*.pth)')
    parser.add_argument('--freeze-cnn', action='store_true',
                       help='fine-tune 时冻结 CNN 编码器')
    parser.add_argument('--gate-check', action='store_true',
                       help='fine-tune 后执行五尺度门控评估')
    parser.add_argument('--fine-tune-updates', type=int, default=100,
                       help='fine-tune 更新次数 (默认 100)')
    args = parser.parse_args()

    if args.mode == 'fine-tune':
        if not args.checkpoint:
            parser.error("--mode fine-tune 需要 --checkpoint <path>")
        # Fine-tune 使用更低 lr
        ft_lr = args.lr if args.lr != 3e-4 else 1e-4
        config = {
            'max_episodes': args.episodes,    # 用于未来兼容
            'fine_tune_updates': args.fine_tune_updates,
            'num_envs': args.envs,
            'lr': ft_lr,
            'degradation': 'clean',
            'rollout_steps': args.rollout_steps,
            'renderer': args.renderer,
            'ply_path': args.ply,
            'minibatch_size': 32,
            'epochs': 5,
            'eval_interval': max(5, args.fine_tune_updates // 10),
            'eval_episodes': 200,
            'seed': args.seed,
            'checkpoint': args.checkpoint,
            'freeze_cnn': args.freeze_cnn,
            'gate_check': args.gate_check,
            'model_out': args.model_out or 'saved_models/visual_ppo_finetuned.pth',
        }
        result = fine_tune_visual(config)
        logger.info(f"Fine-tune final: best_SR={result['best_success_rate']:.1f}%")
        if result.get('gate_check'):
            logger.info("Gate check completed — see log above")
        return

    # 默认: train mode
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
        'eval_episodes': 100,  # Fix 4: ≥100ep 稳定评估 (was 20)
        'seed': args.seed,
        'model_out': args.model_out,
    }

    result = train_visual(config)
    logger.info(f"Final: best_SR={result['best_success_rate']:.1f}%")


if __name__ == "__main__":
    main()
