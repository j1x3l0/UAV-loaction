#!/usr/bin/env python3
"""Evaluate the D2-trained policy under the D2 task (wrapped env).

Tests whether the D2 policy needs depth: compares baseline vs const_depth
on pure-clear (avoid=0.0) and pure-blocked (avoid=1.0) episodes, all under
the D2FixedGoalWrapper (fixed gate goal, hidden target vector). If baseline
is high and const_depth collapses, depth necessity is induced.
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
from experiments.visual_necessity.d2_wrapper import D2FixedGoalWrapper

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
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--collision-ply", required=True)
    parser.add_argument("--collision-radius", type=float, default=0.5)
    parser.add_argument("--target", default="0.0,0.0,1.73")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    target = np.asarray([float(v) for v in args.target.split(",")], dtype=np.float64)
    policy = load_policy(args.model)
    results = {}
    for avoidance in (0.0, 1.0):
        scene_config = {
            "collision_ply_path": args.collision_ply,
            "drone_collision_radius": args.collision_radius,
            "auto_scene_bounds": True,
            "avoidance_episode_probability": avoidance,
        }
        results[str(avoidance)] = {}
        for name, ablation in ABLATIONS.items():
            base = make_env("clean", "gsplat", args.ply,
                            ablation_config=ablation,
                            scene_config=scene_config,
                            alignment_config=args.alignment)
            env = D2FixedGoalWrapper(base, target)
            eval_result = evaluate_model(
                policy, env, eval_episodes=args.episodes, base_seed=args.seed)
            results[str(avoidance)][name] = round(eval_result["success_rate"], 1)
            env.close()
            print(f"avoid={avoidance} {name}: SR={results[str(avoidance)][name]}%",
                  flush=True)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({
            "model": args.model,
            "target_scene": target.tolist(),
            "episodes_per_cell": args.episodes,
            "results": results,
        }, handle, indent=2)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
