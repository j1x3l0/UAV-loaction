"""
eval_cross_scene.py — Phase V3c 跨场景零样本泛化评估

架构位置: scripts/ (Application层)
WHY: 验证 gate_mid 上训练的策略能否泛化到其他 3DGS 场景
      这是论文中泛化能力最直接的实验证据
数据流: model + scene_list → per-scene eval → comparison table → JSON
边界: 不负责训练、不负责 3DGS 场景构建

用法:
  # 评估多个场景
  python scripts/eval_cross_scene.py \
    --model saved_models/visual_ppo_best.pth \
    --ply-dir data/gs_data/ply_exports/ \
    --scenes gate_mid_new_gs gate_left gate_right

  # 显式指定场景路径
  python scripts/eval_cross_scene.py \
    --model saved_models/visual_ppo_best.pth \
    --ply-paths data/gs_data/ply_exports/gate_mid_new_gs.ply \
                 data/gs_data/ply_exports/office0.ply
"""

import numpy as np
import os, sys, argparse, json, csv
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.visual_drone_env import VisualDroneEnv
from core.visual_ppo_agent import VisualPPO
from utils.metrics import wilson_confidence_interval
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_scene(agent, ply_path: str, episodes: int = 200,
                   base_seed: int = 20260729) -> Dict[str, Any]:
    """在单个场景上评估模型 (clean, 无退化)"""
    env = VisualDroneEnv(config={
        'renderer': 'gsplat',
        'ply_path': ply_path,
    })

    successes = collisions = timeouts = 0
    rewards = []
    episode_details = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_reward = 0.0

        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                if info.get('reached_target'):
                    successes += 1
                    result = 'success'
                elif info.get('collision'):
                    collisions += 1
                    result = 'collision'
                else:
                    timeouts += 1
                    result = 'timeout'
                break

        rewards.append(ep_reward)
        episode_details.append({
            'episode': ep,
            'result': result,
            'reward': float(ep_reward),
        })

    env.close()
    n = max(episodes, 1)
    sr_prop = successes / n
    sr_low, sr_high = wilson_confidence_interval(sr_prop, n)
    scene_name = os.path.splitext(os.path.basename(ply_path))[0]

    return {
        'scene': scene_name,
        'ply_path': ply_path,
        'success_rate': sr_prop * 100,
        'sr_ci_low': sr_low * 100,
        'sr_ci_high': sr_high * 100,
        'collision_rate': collisions / n * 100,
        'timeout_rate': timeouts / n * 100,
        'avg_reward': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'episodes': episodes,
        'episode_details': episode_details,
    }


def find_ply_files(ply_dir: str, scene_names: Optional[List[str]] = None,
                   ply_paths: Optional[List[str]] = None) -> List[str]:
    """解析场景路径列表: 支持目录名列表或显式路径"""
    if ply_paths:
        # 验证显式路径
        for p in ply_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"PLY not found: {p}")
        return ply_paths

    if scene_names and ply_dir:
        found = []
        for name in scene_names:
            # 尝试 name.ply 或 name.gs.ply
            candidates = [
                os.path.join(ply_dir, f"{name}.ply"),
                os.path.join(ply_dir, f"{name}_gs.ply"),
                os.path.join(ply_dir, f"{name}.gs.ply"),
            ]
            matched = [c for c in candidates if os.path.isfile(c)]
            if not matched:
                # 尝试模糊匹配: 目录下包含 name 的 .ply
                all_plys = [f for f in os.listdir(ply_dir) if f.endswith('.ply')]
                fuzzy = [os.path.join(ply_dir, f) for f in all_plys if name in f]
                if fuzzy:
                    matched = fuzzy
            if matched:
                found.append(matched[0])
                logger.info(f"  Scene '{name}' → {matched[0]}")
            else:
                logger.warning(f"  Scene '{name}' NOT FOUND in {ply_dir}")
        return found

    # 默认: 目录下所有 .ply
    if ply_dir and os.path.isdir(ply_dir):
        all_plys = sorted([os.path.join(ply_dir, f) for f in os.listdir(ply_dir)
                          if f.endswith('.ply')])
        logger.info(f"  Auto-detected {len(all_plys)} .ply files in {ply_dir}")
        return all_plys

    raise ValueError("No .ply files found. Specify --ply-dir + --scenes or --ply-paths")


def print_comparison(results: List[Dict[str, Any]]):
    """打印场景对比表"""
    print()
    print("=" * 80)
    print(f"{'Scene':25s} {'SR':>8s} {'95% CI':>15s} {'CR':>8s} {'TO':>8s} {'AvgR':>8s}")
    print("-" * 80)
    for r in results:
        ci = f"[{r['sr_ci_low']:.1f}–{r['sr_ci_high']:.1f}]"
        print(f"{r['scene']:25s} {r['success_rate']:>7.1f}% "
              f"{ci:>15s} "
              f"{r['collision_rate']:>6.1f}% "
              f"{r['timeout_rate']:>6.1f}% "
              f"{r['avg_reward']:>7.1f}")
    print("=" * 80)
    print()


def save_results(results: List[Dict[str, Any]], output_dir: str, timestamp: str):
    """保存结果为 JSON + CSV"""
    os.makedirs(output_dir, exist_ok=True)

    # CSV (不含 episode 明细)
    csv_path = os.path.join(output_dir, f"cross_scene_{timestamp}.csv")
    fieldnames = ['scene', 'success_rate', 'sr_ci_low', 'sr_ci_high',
                  'collision_rate', 'timeout_rate', 'avg_reward', 'reward_std',
                  'episodes', 'ply_path']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"CSV saved: {csv_path}")

    # JSON (含 episode 明细)
    json_path = os.path.join(output_dir, f"cross_scene_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'results': results,
            'summary': {
                'n_scenes': len(results),
                'mean_sr': np.mean([r['success_rate'] for r in results]),
                'min_sr': min(r['success_rate'] for r in results),
                'max_sr': max(r['success_rate'] for r in results),
            },
        }, f, indent=2)
    logger.info(f"JSON saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='V3c 跨场景零样本泛化评估')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径 (.pth)')
    parser.add_argument('--ply-dir', type=str,
                       default='data/gs_data/ply_exports',
                       help='3DGS .ply 目录')
    parser.add_argument('--scenes', type=str, nargs='*',
                       default=['gate_mid_new_gs', 'gate_left', 'gate_right'],
                       help='场景名列表 (在 --ply-dir 下查找)')
    parser.add_argument('--ply-paths', type=str, nargs='*', default=None,
                       help='显式指定 .ply 路径 (覆盖 --ply-dir/--scenes)')
    parser.add_argument('--episodes', type=int, default=200,
                       help='每场景评估 episode 数')
    parser.add_argument('--seed', type=int, default=20260729,
                       help='全局基础 seed')
    parser.add_argument('--output', type=str, default='eval_results/cross_scene',
                       help='输出目录')
    args = parser.parse_args()

    # 解析场景列表
    ply_files = find_ply_files(args.ply_dir, args.scenes, args.ply_paths)
    if not ply_files:
        logger.error("No .ply files found to evaluate. Check --ply-dir/--scenes/--ply-paths")
        sys.exit(1)

    logger.info(f"Evaluating model: {args.model}")
    logger.info(f"Scenes ({len(ply_files)}): {[os.path.basename(p) for p in ply_files]}")
    logger.info(f"Episodes per scene: {args.episodes}")

    # 加载模型
    agent = VisualPPO(vec_dim=6, action_dim=3, num_envs=1)
    agent.load_model(args.model)

    # 逐场景评估
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    for ply_path in ply_files:
        scene_name = os.path.splitext(os.path.basename(ply_path))[0]
        logger.info(f"\n{'='*60}")
        logger.info(f"Scene: {scene_name}")
        logger.info(f"PLY:   {ply_path}")
        logger.info(f"{'='*60}")

        result = evaluate_scene(agent, ply_path, episodes=args.episodes,
                                base_seed=args.seed)

        logger.info(f"  SR={result['success_rate']:.1f}% "
                    f"(95%CI {result['sr_ci_low']:.1f}–{result['sr_ci_high']:.1f}) | "
                    f"CR={result['collision_rate']:.1f}% | "
                    f"TO={result['timeout_rate']:.1f}%")
        results.append(result)

    # 输出
    print_comparison(results)
    save_results(results, args.output, timestamp)

    # 汇总
    mean_sr = np.mean([r['success_rate'] for r in results])
    logger.info(f"Summary: {len(results)} scenes, "
                f"mean SR={mean_sr:.1f}% "
                f"[{min(r['success_rate'] for r in results):.1f}%–"
                f"{max(r['success_rate'] for r in results):.1f}%]")


if __name__ == "__main__":
    main()
