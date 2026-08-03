#!/usr/bin/env python3
"""Evaluate the D3-lite corridor policy: baseline vs const_depth.

All episodes are avoidance episodes in the dense corridor. If baseline
(with depth) is learnable and const_depth collapses, depth is necessary.
No alignment config (the corridor uses the default camera).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from core.visual_ppo_agent import VisualPPO
from scripts.train_visual import evaluate_model, make_env

ABLATIONS = {
    "baseline": {},
    "const_depth": {"const_depth": True},
}


def load_policy(checkpoint: str):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy = VisualPPO(vec_dim=6, action_dim=3)
    policy.model.load_state_dict(data["model_state_dict"], strict=True)
    policy.model.eval()
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--collision-ply", required=True)
    parser.add_argument("--collision-radius", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    policy = load_policy(args.model)
    scene_config = {
        "collision_ply_path": args.collision_ply,
        "drone_collision_radius": args.collision_radius,
        "auto_scene_bounds": True,
        "avoidance_episode_probability": 1.0,
        "scene_boundary_margin": (0.2, 0.2, 0.2),
    }
    results = {}
    for name, ablation in ABLATIONS.items():
        env = make_env("clean", "gsplat", args.ply,
                       ablation_config=ablation, scene_config=scene_config)
        eval_result = evaluate_model(
            policy, env, eval_episodes=args.episodes, base_seed=args.seed)
        results[name] = round(eval_result["success_rate"], 1)
        env.close()
        print(f"{name}: SR={results[name]}%", flush=True)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({
            "model": args.model,
            "episodes_per_cell": args.episodes,
            "results": results,
            "gate_pass": results["baseline"] >= 30.0
                         and results["baseline"] - results["const_depth"] >= 30.0,
        }, handle, indent=2)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
