"""
eval_ablation.py — 消融实验评估脚本 (P0 诊断)

目的: 检验模型是否真的依赖视觉特征和目标方向信息

消融 1: const_depth
  将深度图置为常数 (5.0m) → 模型只能依赖 velocity + target_direction
  IF SR 仍然很高 → 模型未学到真正的视觉特征，问题在深度退化定义

消融 2: no_target_dir  
  去除目标方向向量 → 模型只能依赖 depth + velocity
  IF SR 大幅下降 → 当前网络过度依赖完美的目标方向信息（这在真机不可用）

用法:
  python scripts/eval_ablation.py --model saved_models/visual_ppo_best.pth
  python scripts/eval_ablation.py --model ... --episodes 100
"""

import numpy as np
import os, sys, argparse, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.visual_drone_env import VisualDroneEnv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_ablation(agent, ablation_config, episodes=50, base_seed=20260726):
    """
    在单个消融配置下评估模型
    
    Args:
        agent: 已训练的策略
        ablation_config: {'const_depth': bool, 'no_target_dir': bool}
        episodes: 评估 episode 数
        base_seed: 共享 seed
    
    Returns:
        dict: {success_rate, collision_rate, timeout_rate, avg_reward}
    """
    env_config = {
        'renderer': 'mock',
        'degradation': {},
        'ablation': ablation_config
    }
    env = VisualDroneEnv(config=env_config)
    
    successes = 0
    collisions = 0
    timeouts = 0
    rewards_list = []
    
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
                elif info.get('collision'):
                    collisions += 1
                else:
                    timeouts += 1
                break
        
        rewards_list.append(ep_reward)
    
    env.close()
    
    n = max(episodes, 1)
    return {
        'success_rate': successes / n * 100,
        'collision_rate': collisions / n * 100,
        'timeout_rate': timeouts / n * 100,
        'avg_reward': float(np.mean(rewards_list)),
        'reward_std': float(np.std(rewards_list)),
    }


def main():
    parser = argparse.ArgumentParser(description='消融实验评估')
    parser.add_argument('--model', type=str, required=True, help='模型路径 (.pth)')
    parser.add_argument('--episodes', type=int, default=50, help='每配置评估 episode 数')
    parser.add_argument('--output', type=str, default='eval_results', help='输出目录')
    parser.add_argument('--seed', type=int, default=20260726, help='首个 episode seed')
    args = parser.parse_args()
    
    # 加载模型
    logger.info(f"Loading model: {args.model}")
    from core.visual_ppo_agent import VisualPPO
    agent = VisualPPO(vec_dim=6, action_dim=3)
    agent.load_model(args.model)
    
    # Baseline (无消融)
    logger.info("\n" + "="*60)
    logger.info("BASELINE (no ablation)")
    logger.info("="*60)
    baseline = evaluate_ablation(agent, {}, episodes=args.episodes, base_seed=args.seed)
    logger.info(f"  SR: {baseline['success_rate']:.1f}%")
    logger.info(f"  CR: {baseline['collision_rate']:.1f}%")
    logger.info(f"  avgR: {baseline['avg_reward']:.1f}")
    
    # 消融 1: 常数深度
    logger.info("\n" + "="*60)
    logger.info("ABLATION 1: const_depth (depth=5.0m)")
    logger.info("诊断: 模型是否真的用深度信息?")
    logger.info("="*60)
    ablation_const_depth = evaluate_ablation(
        agent, {'const_depth': True}, 
        episodes=args.episodes, base_seed=args.seed)
    logger.info(f"  SR: {ablation_const_depth['success_rate']:.1f}%")
    logger.info(f"  CR: {ablation_const_depth['collision_rate']:.1f}%")
    logger.info(f"  avgR: {ablation_const_depth['avg_reward']:.1f}")
    delta_sr = ablation_const_depth['success_rate'] - baseline['success_rate']
    logger.info(f"  ΔSR: {delta_sr:+.1f}% {'⚠️ 严重问题!' if abs(delta_sr) < 5 else '正常'}")
    
    # 消融 2: 无目标方向
    logger.info("\n" + "="*60)
    logger.info("ABLATION 2: no_target_dir (target=0,0,0)")
    logger.info("诊断: 模型是否过度依赖完美的目标信息?")
    logger.info("="*60)
    ablation_no_target = evaluate_ablation(
        agent, {'no_target_dir': True}, 
        episodes=args.episodes, base_seed=args.seed)
    logger.info(f"  SR: {ablation_no_target['success_rate']:.1f}%")
    logger.info(f"  CR: {ablation_no_target['collision_rate']:.1f}%")
    logger.info(f"  avgR: {ablation_no_target['avg_reward']:.1f}")
    delta_sr_2 = ablation_no_target['success_rate'] - baseline['success_rate']
    logger.info(f"  ΔSR: {delta_sr_2:+.1f}% {'正常' if delta_sr_2 < -10 else '⚠️ 过度依赖!'}")
    
    # 消融 3: 两个都有
    logger.info("\n" + "="*60)
    logger.info("ABLATION 3: const_depth + no_target_dir (最极端)")
    logger.info("诊断: 模型完全丧失视觉和目标信息后会怎样?")
    logger.info("="*60)
    ablation_both = evaluate_ablation(
        agent, {'const_depth': True, 'no_target_dir': True}, 
        episodes=args.episodes, base_seed=args.seed)
    logger.info(f"  SR: {ablation_both['success_rate']:.1f}%")
    logger.info(f"  CR: {ablation_both['collision_rate']:.1f}%")
    logger.info(f"  avgR: {ablation_both['avg_reward']:.1f}")
    
    # 汇总
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    results = {
        'baseline': baseline,
        'ablation_const_depth': ablation_const_depth,
        'ablation_no_target_dir': ablation_no_target,
        'ablation_both': ablation_both,
        'analysis': {
            'const_depth_sr_delta': delta_sr,
            'no_target_dir_sr_delta': delta_sr_2,
            'interpretation': {
                'const_depth': (
                    f"SR 从 {baseline['success_rate']:.1f}% "
                    f"{'保持' if abs(delta_sr) < 5 else '下降到'} "
                    f"{ablation_const_depth['success_rate']:.1f}% "
                    f"→ 模型 {'NOT' if abs(delta_sr) < 5 else ''} 依赖深度信息"
                ),
                'no_target_dir': (
                    f"SR 从 {baseline['success_rate']:.1f}% "
                    f"下降到 {ablation_no_target['success_rate']:.1f}% "
                    f"→ 模型 {'过度依赖' if delta_sr_2 > -20 else '部分依赖'} 目标方向"
                ),
            }
        }
    }
    
    logger.info(json.dumps(results['analysis'], indent=2, ensure_ascii=False))
    
    # 保存
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output, f"ablation_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
