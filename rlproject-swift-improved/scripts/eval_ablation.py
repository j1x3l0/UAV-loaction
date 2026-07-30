"""Evaluate whether a policy actually uses depth and target direction."""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.visual_ppo_agent import VisualPPO
from envs.visual_drone_env import VisualDroneEnv


def wilson_interval(successes, total, z=1.959963984540054):
    """Return the Wilson 95% confidence interval as percentages."""
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z * math.sqrt(
            p * (1.0 - p) / total + z * z / (4.0 * total * total)
        ) / denominator
    )
    return [100.0 * (centre - margin), 100.0 * (centre + margin)]


def evaluate(agent, base_config, ablation, episodes, base_seed):
    config = dict(base_config)
    config["ablation"] = dict(ablation)
    env = VisualDroneEnv(config=config)
    counts = {"success": 0, "collision": 0, "timeout": 0}
    strata = {
        "clear": {"success": 0, "collision": 0, "timeout": 0, "episodes": 0},
        "avoidance": {
            "success": 0, "collision": 0, "timeout": 0, "episodes": 0},
        "unknown": {"success": 0, "collision": 0, "timeout": 0, "episodes": 0},
    }
    rewards = []

    for episode in range(episodes):
        obs, reset_info = env.reset(seed=base_seed + episode)
        requires_avoidance = reset_info.get("requires_avoidance")
        stratum = (
            "avoidance" if requires_avoidance is True
            else "clear" if requires_avoidance is False
            else "unknown"
        )
        strata[stratum]["episodes"] += 1
        total_reward = 0.0
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                if info.get("reached_target"):
                    counts["success"] += 1
                    strata[stratum]["success"] += 1
                elif info.get("collision"):
                    counts["collision"] += 1
                    strata[stratum]["collision"] += 1
                else:
                    counts["timeout"] += 1
                    strata[stratum]["timeout"] += 1
                break
        rewards.append(total_reward)

    env.close()
    result = {
        "success_rate": counts["success"] / episodes * 100,
        "collision_rate": counts["collision"] / episodes * 100,
        "timeout_rate": counts["timeout"] / episodes * 100,
        "avg_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "episodes": episodes,
    }
    result["success_wilson_95"] = wilson_interval(
        counts["success"], episodes)
    result["strata"] = {}
    for name, stratum_counts in strata.items():
        total = stratum_counts["episodes"]
        if total == 0:
            continue
        result["strata"][name] = {
            **stratum_counts,
            "success_rate": 100.0 * stratum_counts["success"] / total,
            "collision_rate": 100.0 * stratum_counts["collision"] / total,
            "timeout_rate": 100.0 * stratum_counts["timeout"] / total,
            "success_wilson_95": wilson_interval(
                stratum_counts["success"], total),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--renderer", choices=["mock", "gsplat"],
                        default="gsplat")
    parser.add_argument("--ply", help="Required for --renderer gsplat")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", default="eval_results/ablation")
    parser.add_argument("--collision-ply",
                        help="Dense collision cloud in renderer coordinates")
    parser.add_argument("--camera-tracks-motion", action="store_true")
    parser.add_argument(
        "--geodesic-reward", action="store_true",
        help="Use the same free-space progress reward as training")
    parser.add_argument("--geodesic-progress-scale", type=float, default=10.0)
    parser.add_argument("--geodesic-heading-weight", type=float, default=2.0)
    parser.add_argument("--geodesic-waypoint-lookahead", type=float,
                        default=0.9)
    parser.add_argument(
        "--waypoint-observation", action="store_true",
        help="Use local safe-path direction instead of final-goal direction")
    parser.add_argument(
        "--configs", default="all",
        help="Comma-separated subset or 'all': baseline,const_depth,"
             "no_velocity,no_target_dir,no_depth_no_velocity,"
             "no_velocity_no_target_dir,all_inputs_ablated")
    args = parser.parse_args()

    if args.renderer == "gsplat" and (
            not args.ply or not os.path.isfile(args.ply)):
        parser.error("--renderer gsplat requires an existing --ply")

    base_config = {"renderer": args.renderer}
    if args.ply:
        base_config["ply_path"] = args.ply
    if args.collision_ply:
        if not os.path.isfile(args.collision_ply):
            parser.error("--collision-ply must be an existing PLY")
        base_config["collision_ply_path"] = args.collision_ply
        base_config["auto_scene_bounds"] = True
    base_config["camera_tracks_motion"] = args.camera_tracks_motion
    base_config["use_geodesic_reward"] = args.geodesic_reward
    base_config["geodesic_progress_scale"] = args.geodesic_progress_scale
    base_config["geodesic_heading_weight"] = args.geodesic_heading_weight
    base_config["geodesic_waypoint_lookahead"] = \
        args.geodesic_waypoint_lookahead
    base_config["use_waypoint_observation"] = args.waypoint_observation

    agent = VisualPPO(vec_dim=6, action_dim=3)
    agent.load_model(args.model)
    configs = {
        "baseline": {},
        "const_depth": {"const_depth": True},
        "no_velocity": {"no_velocity": True},
        "no_target_dir": {"no_target_dir": True},
        "no_depth_no_velocity": {
            "const_depth": True,
            "no_velocity": True,
        },
        "no_velocity_no_target_dir": {
            "no_velocity": True,
            "no_target_dir": True,
        },
        "all_inputs_ablated": {
            "const_depth": True,
            "no_velocity": True,
            "no_target_dir": True,
        },
    }
    if args.configs != "all":
        requested = [name.strip() for name in args.configs.split(",")]
        unknown = sorted(set(requested) - set(configs))
        if unknown:
            parser.error(f"unknown --configs values: {unknown}")
        configs = {name: configs[name] for name in requested}
    results = {
        name: evaluate(agent, base_config, config, args.episodes, args.seed)
        for name, config in configs.items()
    }
    baseline_sr = (
        results["baseline"]["success_rate"]
        if "baseline" in results else None)
    results["analysis"] = {
        f"{name}_delta": result["success_rate"] - baseline_sr
        for name, result in results.items()
        if baseline_sr is not None and name != "baseline"
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
    csv_path = os.path.splitext(output_path)[0] + ".csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "ablation", "episodes", "success_rate", "success_ci_low",
            "success_ci_high", "collision_rate", "timeout_rate",
            "avg_reward", "reward_std", "delta_vs_baseline",
        ])
        for name, result in results.items():
            if name in {"analysis", "metadata"}:
                continue
            writer.writerow([
                name,
                result["episodes"],
                result["success_rate"],
                result["success_wilson_95"][0],
                result["success_wilson_95"][1],
                result["collision_rate"],
                result["timeout_rate"],
                result["avg_reward"],
                result["reward_std"],
                (result["success_rate"] - baseline_sr
                 if baseline_sr is not None else ""),
            ])
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
