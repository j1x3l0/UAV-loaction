"""Evaluate whether a policy actually uses depth and target direction.

V3b 消融实验:
  1. 无深度图 (RGB替代) — 深度图 vs RGB 对比
  2. 无速度向量 — 去掉 velocity 输入
  3. 浅CNN (1层) — 编码器深度消融
  4. 无特权Critic — critic 是否依赖完整 state
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.visual_ppo_agent import VisualPPO
from envs.visual_drone_env import VisualDroneEnv


def evaluate(agent, base_config, ablation, episodes, base_seed):
    config = dict(base_config)
    config["ablation"] = dict(ablation)
    env = VisualDroneEnv(config=config)
    counts = {"success": 0, "collision": 0, "timeout": 0}
    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=base_seed + episode)
        total_reward = 0.0
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                if info.get("reached_target"):
                    counts["success"] += 1
                elif info.get("collision"):
                    counts["collision"] += 1
                else:
                    counts["timeout"] += 1
                break
        rewards.append(total_reward)

    env.close()
    return {
        "success_rate": counts["success"] / episodes * 100,
        "collision_rate": counts["collision"] / episodes * 100,
        "timeout_rate": counts["timeout"] / episodes * 100,
        "avg_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--renderer", choices=["mock", "gsplat"],
                        default="gsplat")
    parser.add_argument("--ply", help="Required for --renderer gsplat")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", default="eval_results/ablation")
    parser.add_argument("--ablation", choices=[
        "baseline", "const_depth", "no_target_dir", "no_velocity",
        "rgb_only", "shallow_cnn", "no_privileged_critic",
        "no_vec", "both", "all"
    ], default="all", help="Ablation config to test")
    args = parser.parse_args()

    if args.renderer == "gsplat" and (
            not args.ply or not os.path.isfile(args.ply)):
        parser.error("--renderer gsplat requires an existing --ply")

    base_config = {"renderer": args.renderer}
    if args.ply:
        base_config["ply_path"] = args.ply

    # V3b 消融配置: 包含所有消融实验
    configs = {
        "baseline": {},
        # V2 原有配置
        "const_depth": {"const_depth": True},
        "no_target_dir": {"no_target_dir": True},
        "no_velocity": {"no_velocity": True},
        "no_vec": {"no_velocity": True, "no_target_dir": True},
        "both": {"const_depth": True, "no_target_dir": True},
        # V3b 新增配置
        "rgb_only": {"rgb_only": True},  # 1. 无深度图 (RGB替代)
        "shallow_cnn": {"shallow_cnn": True},  # 3. 浅CNN (1层)
        "no_privileged_critic": {"no_privileged_critic": True},  # 4. 无特权Critic
    }

    # 根据 --ablation 参数选择要测试的配置
    if args.ablation == "all":
        selected_configs = configs
    elif args.ablation in configs:
        selected_configs = {args.ablation: configs[args.ablation]}
    else:
        parser.error(f"Unknown ablation: {args.ablation}")

    results = {}
    for name, ablation_config in selected_configs.items():
        print(f"\n{'='*60}")
        print(f"Running ablation: {name}")
        print(f"Config: {ablation_config}")
        print(f"{'='*60}")

        # 为每个消融配置创建新的 agent (因为模型结构可能不同)
        agent = VisualPPO(
            vec_dim=6,
            action_dim=3,
            ablation_config=ablation_config
        )
        agent.load_model(args.model)

        # 调整输入通道 (rgb_only 需要 3 通道)
        if ablation_config.get("rgb_only", False):
            agent.model.visual_encoder = type(agent.model.visual_encoder)(
                in_channels=3 if ablation_config.get("rgb_only", False) else 1,
                feature_dim=128
            ).to(next(agent.model.parameters()).device)
            # 重新加载权重 (如果模型支持)
            try:
                agent.load_model(args.model)
            except Exception:
                print("Warning: Cannot load RGB weights, using random init for RGB ablation")

        result = evaluate(agent, base_config, ablation_config,
                         args.episodes, args.seed)
        results[name] = result
        print(f"Result: SR={result['success_rate']:.1f}% | "
              f"CR={result['collision_rate']:.1f}% | "
              f"TR={result['timeout_rate']:.1f}%")

    # 计算相对 baseline 的 delta
    if "baseline" in results:
        baseline_sr = results["baseline"]["success_rate"]
        results["analysis"] = {}
        for name in selected_configs:
            if name != "baseline" and name in results:
                delta = results[name]["success_rate"] - baseline_sr
                results["analysis"][f"{name}_delta"] = round(delta, 2)

    results["metadata"] = {
        "renderer": args.renderer,
        "ply": args.ply,
        "seed": args.seed,
        "episodes": args.episodes,
        "timestamp": datetime.now().isoformat(),
        "ablation_type": args.ablation,
    }

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(
        args.output, datetime.now().strftime("ablation_%Y%m%d_%H%M%S.json"))
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
