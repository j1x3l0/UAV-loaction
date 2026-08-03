"""
eval_degradation.py — 批量退化评估脚本 v2

架构位置: scripts/ (Application层)
WHY: V2核心实验的执行工具 — 5退化轴×5水平×N场景的自动化评估
数据流: model + degradation_config → N episodes → metrics → CSV/JSON
边界: 不负责训练、不负责退化工具实现、不负责绘图
风险: 单轴评估时间可能很长 → 支持断点续跑 + 按轴分GPU

用法:
  python scripts/eval_degradation.py --model saved_models/visual_ppo_best.pth
  python scripts/eval_degradation.py --model ... --axis gaussian --levels 100,50,25,10,5
  python scripts/eval_degradation.py --model ... --axis all --episodes 50
"""

import numpy as np
import os, sys, argparse, json, csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.visual_drone_env import VisualDroneEnv
from envs.degradation_utils import (
    DEGRADATION_AXES,
    apply_resolution_downscale,
    apply_perlin_depth_noise,
)

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def make_degraded_env(axis, level, base_config=None):
    """
    创建带特定退化配置的环境

    Args:
        axis: 退化轴名称 ('gaussian'|'resolution'|'depth_noise'|'lighting'|'viewpoint_uncertainty')
        level: 退化水平 (对应DEGRADATION_AXES中的levels)
        base_config: 基础环境配置
    Returns:
        VisualDroneEnv 实例
    """
    config = base_config.copy() if base_config else {}
    deg = dict(config.get('degradation', {}))

    # 将退化水平写入配置 (5轴全覆盖)
    deg[axis] = level

    config['degradation'] = deg
    return VisualDroneEnv(config=config)


def evaluate_single_config(agent, env, num_episodes=50, base_seed=20260726):
    """
    在单个退化配置下评估模型

    Returns:
        dict: {success_rate, collision_rate, timeout_rate,
               avg_reward, reward_std, avg_steps, avg_path_length}
    """
    successes = 0; collisions = 0; timeouts = 0
    rewards = []; steps_list = []

    for ep in range(num_episodes):
        # Common random numbers: every degradation level sees exactly the
        # same start/target sequence, so differences are caused by degradation.
        obs, _ = env.reset(seed=base_seed + ep)
        ep_reward = 0.0; ep_steps = 0

        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward; ep_steps += 1

            if terminated or truncated:
                if info.get('reached_target'):
                    successes += 1
                elif info.get('collision'):
                    collisions += 1
                else:
                    timeouts += 1
                break

        rewards.append(ep_reward)
        if info.get('reached_target'):
            steps_list.append(ep_steps)

    n = max(num_episodes, 1)
    return {
        'success_rate': successes / n * 100,
        'collision_rate': collisions / n * 100,
        'timeout_rate': timeouts / n * 100,
        'avg_reward': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'avg_steps': float(np.mean(steps_list)) if steps_list else 0.0,
        'episodes': num_episodes,
    }


def run_degradation_axis(agent, axis_name, levels, episodes_per_level=50,
                          base_config=None, base_seed=20260726):
    """
    测量单条退化轴的完整衰减曲线

    Returns:
        list[dict]: 每个水平一条记录
    """
    axis_info = DEGRADATION_AXES[axis_name]
    logger.info(f"Axis: {axis_info['name']} — {len(levels)} levels × "
                f"{episodes_per_level}ep")

    results = []
    for level in levels:
        env = make_degraded_env(axis_name, level, base_config)
        metrics = evaluate_single_config(
            agent, env, episodes_per_level, base_seed=base_seed)
        record = {
            'axis': axis_name,
            'axis_name': axis_info['name'],
            'level': level,
            'unit': axis_info['unit'],
            **metrics
        }
        results.append(record)
        logger.info(f"  level={level}{axis_info['unit']}: "
                    f"SR={metrics['success_rate']:.1f}% "
                    f"CR={metrics['collision_rate']:.1f}% "
                    f"avgR={metrics['avg_reward']:.1f}")
        env.close()

    return results


def save_results(all_results, output_dir, timestamp):
    """保存结果为 CSV + JSON"""
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, f"degradation_{timestamp}.csv")
    if all_results:
        fieldnames = ['axis', 'axis_name', 'level', 'unit',
                      'success_rate', 'collision_rate', 'timeout_rate',
                      'avg_reward', 'reward_std', 'avg_steps', 'episodes']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"CSV saved: {csv_path}")

    # JSON (含退化轴定义)
    json_path = os.path.join(output_dir, f"degradation_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump({
            'axes_definition': {k: {kk: vv for kk, vv in v.items()
                                    if kk != 'description'}
                               for k, v in DEGRADATION_AXES.items()},
            'results': all_results,
            'timestamp': timestamp,
        }, f, indent=2)
    logger.info(f"JSON saved: {json_path}")


def load_agent_for_eval(model_path):
    """加载模型用于评估"""
    from core.visual_ppo_agent import VisualPPO
    agent = VisualPPO(vec_dim=6, action_dim=3)
    agent.load_model(model_path)
    return agent


def main():
    parser = argparse.ArgumentParser(description='批量退化评估')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径 (.pth)')
    parser.add_argument('--axis', type=str, default='all',
                       help='退化轴: all | original | structural | 单个轴名')
    parser.add_argument('--levels', type=str, default=None,
                       help='自定义退化水平, 逗号分隔 (覆盖默认)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='每水平评估episode数')
    parser.add_argument('--output', type=str, default='eval_results',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=20260726,
                        help='各档共享的首个 episode seed')
    parser.add_argument('--renderer', choices=['mock', 'gsplat'],
                        default='mock')
    parser.add_argument('--ply', type=str, default=None,
                        help='真实 3DGS PLY 路径')
    parser.add_argument('--collision-ply', type=str, default=None,
                        help='与渲染场景同坐标系的稠密碰撞点云')
    parser.add_argument('--camera-tracks-motion', action='store_true')
    parser.add_argument('--alignment', type=str, default=None,
                        help='PX4 对齐配置 JSON：对齐相机（fx≈97.14 + 统一 c2w）')
    parser.add_argument('--collision-radius', type=float, default=None,
                        help='无人机碰撞半径 (m)，对齐任务标定用')
    args = parser.parse_args()
    if args.renderer == 'gsplat':
        if not args.ply or not os.path.isfile(args.ply):
            raise FileNotFoundError("--renderer gsplat requires an existing --ply")
    base_config = {'renderer': args.renderer}
    if args.ply:
        base_config['ply_path'] = args.ply
    if args.collision_ply:
        if not os.path.isfile(args.collision_ply):
            raise FileNotFoundError("--collision-ply must exist")
        base_config['collision_ply_path'] = args.collision_ply
        base_config['auto_scene_bounds'] = True
    base_config['camera_tracks_motion'] = args.camera_tracks_motion
    if args.alignment:
        from scripts.train_visual import load_camera_intrinsics
        base_config['camera_intrinsics'] = load_camera_intrinsics(args.alignment)
        base_config['alignment_config'] = args.alignment
    if args.collision_radius is not None:
        base_config['drone_collision_radius'] = args.collision_radius

    # 加载模型
    logger.info(f"Loading model: {args.model}")
    agent = load_agent_for_eval(args.model)

    # 确定退化轴
    if args.axis == 'all':
        axes_to_run = list(DEGRADATION_AXES.keys())
    elif args.axis == 'original':
        axes_to_run = [
            'gaussian', 'resolution', 'depth_noise', 'lighting',
            'viewpoint_uncertainty',
        ]
    elif args.axis == 'structural':
        axes_to_run = [
            'depth_failure', 'occlusion', 'depth_scale', 'combined',
        ]
    else:
        if args.axis not in DEGRADATION_AXES:
            raise ValueError(f"Unknown axis: {args.axis}. "
                             f"Choose from: {list(DEGRADATION_AXES.keys())}")
        axes_to_run = [args.axis]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for axis_name in axes_to_run:
        axis_info = DEGRADATION_AXES[axis_name]
        levels = axis_info['levels']
        if args.levels and args.axis != 'all':
            levels = [int(x) if x.isdigit() else float(x)
                      for x in args.levels.split(',')]

        axis_results = run_degradation_axis(
            agent, axis_name, levels,
            episodes_per_level=args.episodes,
            base_config=base_config,
            base_seed=args.seed)
        all_results.extend(axis_results)

    # 保存
    save_results(all_results, args.output, timestamp)

    # 汇总
    if len(axes_to_run) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("Summary: critical points (SR drops below 50%)")
        for axis_name in axes_to_run:
            axis_data = [r for r in all_results if r['axis'] == axis_name]
            critical = None
            for r in axis_data:
                if r['success_rate'] < 50:
                    critical = r['level']
                    break
            info = DEGRADATION_AXES[axis_name]
            crit_str = f"{critical}{info['unit']}" if critical else "N/A (>50%)"
            logger.info(f"  {info['name']:12s}: σ_c = {crit_str}")


if __name__ == "__main__":
    main()
