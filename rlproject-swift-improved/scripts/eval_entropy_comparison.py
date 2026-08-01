"""Independent clean evaluation for high-entropy and entropy-fixed policies."""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.visual_ppo_agent import VisualPPO
from envs.visual_drone_env import VisualDroneEnv
from utils.metrics import wilson_confidence_interval


def evaluate(model_path, config, episodes, base_seed):
    agent = VisualPPO(vec_dim=6, action_dim=3)
    agent.load_model(model_path)
    env = VisualDroneEnv(config=config)
    counts = {"success": 0, "collision": 0, "timeout": 0}
    rewards = []
    outcomes = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=base_seed + episode)
        total_reward = 0.0
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                if info.get("reached_target"):
                    outcome = "success"
                elif info.get("collision"):
                    outcome = "collision"
                else:
                    outcome = "timeout"
                counts[outcome] += 1
                outcomes.append(outcome)
                rewards.append(total_reward)
                break

    env.close()
    success_fraction = counts["success"] / episodes
    ci_low, ci_high = wilson_confidence_interval(success_fraction, episodes)
    return {
        "success_rate": success_fraction * 100,
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "collision_rate": counts["collision"] / episodes * 100,
        "timeout_rate": counts["timeout"] / episodes * 100,
        "avg_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "episodes": episodes,
        "outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeat as label=/absolute/path/to/model.pth",
    )
    parser.add_argument("--renderer", choices=["mock", "gsplat"], default="gsplat")
    parser.add_argument("--ply")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", default="eval_results/entropy_comparison")
    args = parser.parse_args()

    if args.renderer == "gsplat" and (
        not args.ply or not os.path.isfile(args.ply)
    ):
        parser.error("--renderer gsplat requires an existing --ply")

    models = []
    for item in args.model:
        if "=" not in item:
            parser.error("--model must use label=path")
        label, path = item.split("=", 1)
        if not os.path.isfile(path):
            parser.error(f"model does not exist: {path}")
        models.append((label, path))

    config = {"renderer": args.renderer}
    if args.ply:
        config["ply_path"] = args.ply

    results = {}
    for label, path in models:
        print(f"Evaluating {label}: {path}", flush=True)
        metrics = evaluate(path, config, args.episodes, args.seed)
        metrics["model_path"] = path
        results[label] = metrics
        print(
            f"{label}: SR={metrics['success_rate']:.1f}% "
            f"(95% CI {metrics['success_ci_low']:.1f}–"
            f"{metrics['success_ci_high']:.1f}), "
            f"CR={metrics['collision_rate']:.1f}%",
            flush=True,
        )

    metadata = {
        "renderer": args.renderer,
        "ply": args.ply,
        "episodes_per_model": args.episodes,
        "base_seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    payload = {"metadata": metadata, "results": results}
    os.makedirs(args.output, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output, f"entropy_comparison_{stamp}.json")
    csv_path = os.path.join(args.output, f"entropy_comparison_{stamp}.csv")
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    fields = [
        "label",
        "success_rate",
        "success_ci_low",
        "success_ci_high",
        "collision_rate",
        "timeout_rate",
        "avg_reward",
        "reward_std",
        "episodes",
        "model_path",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, metrics in results.items():
            writer.writerow(
                {"label": label, **{key: metrics[key] for key in fields[1:]}}
            )
    print(f"Saved: {json_path}", flush=True)
    print(f"Saved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
