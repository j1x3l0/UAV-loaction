#!/usr/bin/env python3
"""Evaluate saved V3 models across depth scales with the aligned env.

Three-way comparison for the formal V3 (clean-best vs curriculum
robust-best vs curriculum final) at the aligned depth scales
[1.0, 0.75, 0.5, 0.25], narrow-FOV camera (fx≈97.14) and 0.5 m clearance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.visual_ppo_agent import VisualPPO
from scripts.train_visual import (
    DEPTH_SCALE_LEVELS,
    evaluate_model,
    make_fixed_depth_scale_env,
)


def load_policy(checkpoint: str):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if data.get("action_dim") != 3:
        raise ValueError(f"{checkpoint} action_dim must be 3")
    policy = VisualPPO(vec_dim=6, action_dim=3)
    policy.model.load_state_dict(data["model_state_dict"], strict=True)
    policy.model.eval()
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True,
                        help="label=checkpoint path (repeat for each model)")
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--collision-ply", required=True)
    parser.add_argument("--collision-radius", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    models = [item.split("=", 1) for item in args.model]
    results = {}
    detail = {}
    for label, path in models:
        print(f"=== {label}: {path} ===", flush=True)
        policy = load_policy(path)
        scale_results = {}
        scale_detail = {}
        for scale in DEPTH_SCALE_LEVELS:
            # Must mirror the training robust-eval env: collision geometry,
            # auto scene bounds and the 0.5 m clearance. Without
            # collision_ply_path the env silently falls back to the mock task.
            scene_config = {
                "drone_collision_radius": args.collision_radius,
                "collision_ply_path": args.collision_ply,
                "auto_scene_bounds": True,
            }
            env = make_fixed_depth_scale_env(
                "gsplat", args.ply, scale,
                scene_config=scene_config,
                alignment_config=args.alignment)
            eval_result = evaluate_model(
                policy, env, eval_episodes=args.episodes, base_seed=args.seed)
            scale_results[str(scale)] = round(eval_result["success_rate"], 1)
            # Save per-episode outcomes for paired McNemar / bootstrap tests.
            scale_detail[str(scale)] = [
                {
                    "episode": row["episode"],
                    "result": row["result"],
                    "reward": round(float(row["reward"]), 4),
                }
                for row in eval_result["episodes_detail"]
            ]
            env.close()
            print(f"  scale {scale}x: SR={scale_results[str(scale)]}%", flush=True)
        results[label] = scale_results
        detail[label] = scale_detail

    summary = {
        "models": {label: {"checkpoint": path} for label, path in models},
        "scales": [str(scale) for scale in DEPTH_SCALE_LEVELS],
        "episodes_per_scale": args.episodes,
        "seed": args.seed,
        "results": results,
        "episodes_detail": detail,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== summary (SR %) ===")
    header = f"{'model':<26}" + "".join(
        f"{str(scale):>10}" for scale in DEPTH_SCALE_LEVELS)
    print(header)
    for label, scale_results in results.items():
        row = f"{label:<26}" + "".join(
            f"{scale_results.get(str(scale), 0.0):>9.1f}%"
            for scale in DEPTH_SCALE_LEVELS)
        print(row)
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
