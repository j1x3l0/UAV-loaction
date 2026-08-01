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
from utils.metrics import wilson_confidence_interval
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEPTH_SCALE_LEVELS = [1.0, 0.75, 0.5, 0.25, 0.1]
SCALE_CURRICULUM = (
    (0.0, 'foundation', [0.35, 0.25, 0.20, 0.10, 0.10]),
    (0.3, 'transition', [0.25, 0.20, 0.30, 0.15, 0.10]),
    (0.7, 'robustness', [0.25, 0.15, 0.25, 0.20, 0.15]),
)
AVOIDANCE_CURRICULUM = (
    (0.0, 'clear_foundation', 0.10),
    (0.3, 'mixed_transition', 0.30),
    (0.7, 'balanced_avoidance', 0.50),
)


def get_scale_curriculum_stage(progress):
    """Return the active curriculum name and probabilities."""
    if not 0.0 <= progress <= 1.0:
        raise ValueError("curriculum progress must be in [0, 1]")
    active = SCALE_CURRICULUM[0]
    for stage in SCALE_CURRICULUM:
        if progress >= stage[0]:
            active = stage
    return active[1], active[2]


def get_avoidance_curriculum_stage(progress):
    """Return curriculum name and blocked-episode probability."""
    if not 0.0 <= progress <= 1.0:
        raise ValueError("curriculum progress must be in [0, 1]")
    active = AVOIDANCE_CURRICULUM[0]
    for stage in AVOIDANCE_CURRICULUM:
        if progress >= stage[0]:
            active = stage
    return active[1], active[2]


def get_checkpoint_paths(clean_best_path):
    """Derive explicit robust-best and final paths without breaking callers."""
    root, extension = os.path.splitext(clean_best_path)
    if not extension:
        extension = '.pth'
    variant_root = root[:-5] if root.endswith('_best') else root
    return {
        'clean_best': clean_best_path,
        'robust_best': f'{variant_root}_robust_best{extension}',
        'final': f'{variant_root}_final{extension}',
    }


def robust_validation_score(scale_results):
    """Lexicographic score: protect the worst scale, then maximize the mean."""
    success_rates = [result['success_rate'] for result in scale_results]
    if not success_rates:
        raise ValueError("scale_results must not be empty")
    return min(success_rates), float(np.mean(success_rates))


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


def make_env(degradation_config=None, renderer='mock', ply_path=None,
             ablation_config=None, scene_config=None):
    """环境工厂"""
    cfg = {'renderer': renderer}
    if renderer == 'gsplat':
        if not ply_path:
            raise ValueError("--renderer gsplat requires --ply <path to .ply>")
        cfg['ply_path'] = resolve_ply_path(ply_path)
    if ablation_config:
        cfg['ablation'] = dict(ablation_config)
    if scene_config:
        cfg.update(scene_config)
    if degradation_config and degradation_config != 'clean':
        if degradation_config == 'rand':
            level = np.random.choice([100, 50, 25, 10, 5])
            cfg['degradation'] = {'resolution': max(16, int(64 * level / 100))}
        elif degradation_config == 'scale_rand':
            cfg['randomize_depth_scale'] = True
            cfg['depth_scale_levels'] = DEPTH_SCALE_LEVELS
        elif degradation_config == 'scale_weighted':
            # V3 follow-up: concentrate training on the unstable 0.5x
            # transition while retaining clean and extreme-scale exposure.
            cfg['randomize_depth_scale'] = True
            cfg['depth_scale_levels'] = DEPTH_SCALE_LEVELS
            cfg['depth_scale_probabilities'] = [0.2, 0.2, 0.4, 0.1, 0.1]
        elif degradation_config == 'scale_curriculum':
            cfg['randomize_depth_scale'] = True
            cfg['depth_scale_levels'] = DEPTH_SCALE_LEVELS
            _, cfg['depth_scale_probabilities'] = \
                get_scale_curriculum_stage(0.0)
        elif degradation_config == 'scale_recovery':
            cfg['randomize_depth_scale'] = True
            cfg['depth_scale_levels'] = DEPTH_SCALE_LEVELS
            cfg['depth_scale_probabilities'] = [0.60, 0.15, 0.10, 0.10, 0.05]
    return VisualDroneEnv(config=cfg)


def make_fixed_depth_scale_env(renderer, ply_path, depth_scale,
                               ablation_config=None, scene_config=None):
    """Create a validation environment with one fixed depth calibration."""
    cfg = {
        'renderer': renderer,
        'degradation': {'depth_scale': depth_scale},
    }
    if renderer == 'gsplat':
        if not ply_path:
            raise ValueError("--renderer gsplat requires --ply <path to .ply>")
        cfg['ply_path'] = resolve_ply_path(ply_path)
    if ablation_config:
        cfg['ablation'] = dict(ablation_config)
    if scene_config:
        cfg.update(scene_config)
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
    resume_model = config.get('resume_model')
    checkpoint_paths = get_checkpoint_paths(model_out)
    for checkpoint_path in checkpoint_paths.values():
        os.makedirs(
            os.path.dirname(os.path.abspath(checkpoint_path)),
            exist_ok=True,
        )

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    degradation = config.get('degradation', 'clean')
    renderer = config.get('renderer', 'mock')
    ply_path = config.get('ply_path')
    ablation_config = config.get('ablation')
    scene_config = dict(config.get('scene_config') or {})
    avoidance_curriculum = bool(config.get('avoidance_curriculum', False))
    train_scene_config = dict(scene_config)
    eval_scene_config = dict(scene_config)
    if avoidance_curriculum:
        _, initial_avoidance_probability = \
            get_avoidance_curriculum_stage(0.0)
        train_scene_config['avoidance_episode_probability'] = \
            initial_avoidance_probability
        # Evaluation remains balanced throughout training.
        eval_scene_config['avoidance_episode_probability'] = 0.5

    # 创建环境
    envs = [
        make_env(
            degradation, renderer, ply_path, ablation_config,
            train_scene_config)
        for _ in range(num_envs)
    ]
    eval_env = make_env(
        'clean', renderer, ply_path, ablation_config, eval_scene_config)
    robust_eval_envs = None
    if degradation in ('scale_curriculum', 'scale_recovery'):
        robust_eval_envs = [
            make_fixed_depth_scale_env(
                renderer, ply_path, scale, ablation_config,
                eval_scene_config)
            for scale in DEPTH_SCALE_LEVELS
        ]

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
    if resume_model:
        resolved_resume_model = os.path.abspath(resume_model)
        if not os.path.isfile(resolved_resume_model):
            raise FileNotFoundError(
                f"resume checkpoint not found: {resume_model}")
        if not ppo.load_model(resolved_resume_model):
            raise RuntimeError(
                f"failed to load resume checkpoint: {resolved_resume_model}")
        logger.info(f"Resuming training from: {resolved_resume_model}")

    # 重置环境
    observations = [env.reset(seed=seed + i)[0]
                    for i, env in enumerate(envs)]
    step_count = 0
    start_time = time.time()
    # Start below the valid SR range so even a 0%-SR smoke run produces a
    # loadable clean-best checkpoint.
    best_sr = -1.0
    best_robust_score = (-1.0, -1.0)
    active_curriculum_stage = (
        'recovery' if degradation == 'scale_recovery' else None)
    active_avoidance_stage = None

    logger.info(f"Training: {max_episodes}ep × {num_envs}envs × "
                f"{rollout_steps}steps, degradation={degradation}, "
                f"renderer={renderer}")

    for episode in range(max_episodes):
        if avoidance_curriculum:
            curriculum_progress = episode / max_episodes
            avoidance_stage, avoidance_probability = \
                get_avoidance_curriculum_stage(curriculum_progress)
            if avoidance_stage != active_avoidance_stage:
                active_avoidance_stage = avoidance_stage
                for env in envs:
                    env.set_avoidance_episode_probability(
                        avoidance_probability)
                    env.reset_avoidance_sample_counts()
                logger.info(
                    f"Avoidance curriculum stage={avoidance_stage} "
                    f"progress={curriculum_progress:.1%} "
                    f"blocked_probability={avoidance_probability:.2f}")
        if degradation == 'scale_curriculum':
            curriculum_progress = episode / max_episodes
            stage_name, probabilities = get_scale_curriculum_stage(
                curriculum_progress)
            if stage_name != active_curriculum_stage:
                active_curriculum_stage = stage_name
                for env in envs:
                    env.set_depth_scale_probabilities(probabilities)
                    env.reset_depth_scale_sample_counts()
                logger.info(
                    f"Curriculum stage={stage_name} "
                    f"progress={curriculum_progress:.1%} "
                    f"probabilities={probabilities}")

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
                        f"alpha: {result['entropy_coeff']:.5f} | "
                        f"lr: {ppo.current_lr:.2e} | eta: {eta}")
            if degradation in ('scale_curriculum', 'scale_recovery'):
                sample_counts = np.sum(
                    [env.depth_scale_sample_counts for env in envs], axis=0)
                sample_total = int(sample_counts.sum())
                frequencies = (
                    sample_counts / sample_total
                    if sample_total else np.zeros_like(
                        sample_counts, dtype=np.float64))
                logger.info(
                    f"  Scale samples ({active_curriculum_stage}, "
                    f"n={sample_total}): "
                    f"{dict(zip(DEPTH_SCALE_LEVELS, frequencies.round(3)))}")
            if avoidance_curriculum:
                avoidance_counts = np.sum(
                    [env.avoidance_sample_counts for env in envs], axis=0)
                avoidance_total = int(avoidance_counts.sum())
                avoidance_frequencies = (
                    avoidance_counts / avoidance_total
                    if avoidance_total else np.zeros(
                        2, dtype=np.float64))
                logger.info(
                    f"  Task samples ({active_avoidance_stage}, "
                    f"n={avoidance_total}): "
                    f"clear={avoidance_frequencies[0]:.3f}, "
                    f"avoidance={avoidance_frequencies[1]:.3f}")

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
                ppo.save_model(checkpoint_paths['clean_best'])
                logger.info(f"  New best model (SR={best_sr:.1f}%)")

            if robust_eval_envs is not None:
                robust_results = [
                    evaluate_model(
                        ppo, robust_env,
                        config.get('robust_eval_episodes', 20),
                        base_seed=seed + 200_000)
                    for robust_env in robust_eval_envs
                ]
                robust_score = robust_validation_score(robust_results)
                scale_summary = ', '.join(
                    f'{scale:g}x={result["success_rate"]:.1f}%'
                    for scale, result in zip(
                        DEPTH_SCALE_LEVELS, robust_results))
                logger.info(
                    f"  Robust eval: {scale_summary} | "
                    f"min={robust_score[0]:.1f}% "
                    f"mean={robust_score[1]:.1f}%")
                if robust_score > best_robust_score:
                    best_robust_score = robust_score
                    ppo.save_model(checkpoint_paths['robust_best'])
                    logger.info(
                        f"  New robust-best model "
                        f"(min={robust_score[0]:.1f}%, "
                        f"mean={robust_score[1]:.1f}%)")

    elapsed = time.time() - start_time
    ppo.save_model(checkpoint_paths['final'])
    logger.info(f"Final checkpoint saved: {checkpoint_paths['final']}")
    logger.info(f"Training done | time={format_time(elapsed)} | "
                f"best_SR={best_sr:.1f}% | steps={step_count}")
    return {
        'best_success_rate': best_sr,
        'best_robust_min_success_rate': best_robust_score[0],
        'best_robust_mean_success_rate': best_robust_score[1],
        'checkpoint_paths': checkpoint_paths,
        'time': elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--envs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--degradation', type=str, default='clean',
                       choices=['clean', 'rand', 'scale_rand',
                                'scale_weighted', 'scale_curriculum',
                                'scale_recovery'])
    parser.add_argument('--rollout-steps', type=int, default=256)
    parser.add_argument('--renderer', type=str, default='mock',
                       choices=['mock', 'gsplat'])
    parser.add_argument('--ply', type=str, default=None,
                       help='3DGS .ply 路径 (--renderer gsplat 时必填)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-out', type=str,
                        default='saved_models/visual_ppo_best.pth')
    parser.add_argument('--robust-eval-episodes', type=int, default=20,
                        help='curriculum每个尺度的checkpoint选择评估数')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='clean checkpoint评估episode数')
    parser.add_argument('--resume-model', type=str, default=None,
                        help='从现有VisualPPO checkpoint继续训练')
    parser.add_argument('--ablation', choices=['none', 'no_velocity'],
                        default='none',
                        help='训练和评估时使用相同的输入消融')
    parser.add_argument('--collision-ply', default=None,
                        help='与渲染场景同坐标系的稠密碰撞点云')
    parser.add_argument('--camera-tracks-motion', action='store_true',
                        help='相机光轴随速度方向变化，低速时朝向目标')
    parser.add_argument('--geodesic-reward', action='store_true',
                        help='使用沿自由空间最短路的进度奖励')
    parser.add_argument('--geodesic-progress-scale', type=float, default=10.0,
                        help='测地势能差奖励权重')
    parser.add_argument('--geodesic-heading-weight', type=float, default=2.0,
                        help='局部安全路径方向奖励权重')
    parser.add_argument('--geodesic-waypoint-lookahead', type=float,
                        default=0.9,
                        help='训练奖励使用的路径点前视距离（米）')
    parser.add_argument(
        '--waypoint-observation', action='store_true',
        help='诊断模式：用局部安全路径方向替代最终目标方向观测')
    parser.add_argument('--avoidance-curriculum', action='store_true',
                        help='避障任务比例按10%%/30%%/50%%分阶段增加')
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
        'eval_episodes': args.eval_episodes,
        'seed': args.seed,
        'model_out': args.model_out,
        'robust_eval_episodes': args.robust_eval_episodes,
        'resume_model': args.resume_model,
        'ablation': (
            {'no_velocity': True}
            if args.ablation == 'no_velocity' else None
        ),
        'avoidance_curriculum': args.avoidance_curriculum,
        'scene_config': ({
            'collision_ply_path': args.collision_ply,
            'auto_scene_bounds': True,
            'camera_tracks_motion': args.camera_tracks_motion,
            'use_geodesic_reward': args.geodesic_reward,
            'geodesic_progress_scale': args.geodesic_progress_scale,
            'geodesic_heading_weight': args.geodesic_heading_weight,
            'geodesic_waypoint_lookahead':
                args.geodesic_waypoint_lookahead,
            'use_waypoint_observation': args.waypoint_observation,
        } if args.collision_ply else {
            'camera_tracks_motion': args.camera_tracks_motion,
        }),
    }

    result = train_visual(config)
    logger.info(f"Final: best_SR={result['best_success_rate']:.1f}%")


if __name__ == "__main__":
    main()
