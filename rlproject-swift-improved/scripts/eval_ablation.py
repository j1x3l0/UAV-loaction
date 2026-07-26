"""Evaluate whether a policy actually uses depth and target direction."""

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
    args = parser.parse_args()

    if args.renderer == "gsplat" and (
            not args.ply or not os.path.isfile(args.ply)):
        parser.error("--renderer gsplat requires an existing --ply")

    base_config = {"renderer": args.renderer}
    if args.ply:
        base_config["ply_path"] = args.ply

    agent = VisualPPO(vec_dim=6, action_dim=3)
    agent.load_model(args.model)
    configs = {
        "baseline": {},
        "const_depth": {"const_depth": True},
        "no_target_dir": {"no_target_dir": True},
        "both": {"const_depth": True, "no_target_dir": True},
    }
    results = {
        name: evaluate(agent, base_config, config, args.episodes, args.seed)
        for name, config in configs.items()
    }
    baseline_sr = results["baseline"]["success_rate"]
    results["analysis"] = {
        "const_depth_delta": results["const_depth"]["success_rate"] - baseline_sr,
        "no_target_dir_delta":
            results["no_target_dir"]["success_rate"] - baseline_sr,
        "both_delta": results["both"]["success_rate"] - baseline_sr,
    }
    results["metadata"] = {
        "renderer": args.renderer,
        "ply": args.ply,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(
        args.output, datetime.now().strftime("ablation_%Y%m%d_%H%M%S.json"))
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
