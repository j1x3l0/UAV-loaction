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

from envs.visual_drone_env import (
    VisualDroneEnv,
    apply_resolution_downscale,
    apply_perlin_depth_noise,
)

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ─── 退化轴定义 ──────────────────────────────────────────────────
# WHY 集中定义: 所有退化轴在这里统一声明，eval和绘图脚本共用

DEGRADATION_AXES = {
    'gaussian': {
        'name': '高斯球稀疏化',
        'levels': [100, 50, 25, 10, 5],
        'unit': '%',
        'description': '按重要性保留的高斯球比例',
    },
    'resolution': {
        'name': '渲染分辨率',
        'levels': [64, 32, 16, 8, 4],
        'unit': 'px',
        'description': '深度图降采样分辨率（上采样回64×64）',
    },
    'depth_noise': {
        'name': '深度噪声',
        'levels': [0.0, 0.01, 0.05, 0.1, 0.2],
        'unit': 'σ',
        'description': '空间相关Perlin噪声标准差',
    },
    'lighting': {
        'name': '光照偏移',
        'levels': [0, 1, 2, 3, 4],
        'unit': 'EV',
        'description': 'RGB曝光偏移（EV档）',
    },
    'viewpoint': {
        'name': '视角覆盖',
        'levels': [360, 270, 180, 90, 45],
        'unit': '°',
        'description': '允许的相机朝向角度范围',
    },
}


def make_degraded_env(axis, level, base_config=None):
    """
    创建带特定退化配置的环境

    Args:
        axis: 退化轴名称 ('gaussian'|'resolution'|'depth_noise'|'lighting'|'viewpoint')
        level: 退化水平 (对应DEGRADATION_AXES中的levels)
        base_config: 基础环境配置
    Returns:
        VisualDroneEnv 实例
    """
    config = base_config.copy() if base_config else {}
    deg = config.get('degradation', {})

    if axis == 'resolution':
        deg['resolution'] = level
    elif axis == 'depth_noise':
        deg['depth_noise'] = level

    config['degradation'] = deg
    return VisualDroneEnv(config=config)


def evaluate_single_config(agent, env, num_episodes=50):
    """
    在单个退化配置下评估模型

    Returns:
        dict: {success_rate, collision_rate, timeout_rate,
               avg_reward, reward_std, avg_steps, avg_path_length}
    """
    successes = 0; collisions = 0; timeouts = 0
    rewards = []; steps_list = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
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
                          base_config=None):
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
        metrics = evaluate_single_config(agent, env, episodes_per_level)
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
                       help='退化轴: all | gaussian | resolution | '
                            'depth_noise | lighting | viewpoint')
    parser.add_argument('--levels', type=str, default=None,
                       help='自定义退化水平, 逗号分隔 (覆盖默认)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='每水平评估episode数')
    parser.add_argument('--output', type=str, default='eval_results',
                       help='输出目录')
    args = parser.parse_args()

    # 加载模型
    logger.info(f"Loading model: {args.model}")
    agent = load_agent_for_eval(args.model)

    # 确定退化轴
    if args.axis == 'all':
        axes_to_run = list(DEGRADATION_AXES.keys())
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
            episodes_per_level=args.episodes)
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
