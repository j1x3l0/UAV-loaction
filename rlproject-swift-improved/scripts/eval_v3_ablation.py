#!/usr/bin/env python3
"""Input-dependency ablation for the aligned V3 models.

Tests whether the aligned curriculum / clean policies actually use depth,
velocity and target direction, evaluated on the aligned narrow-FOV task
(fx≈97.14, collision geometry, 0.5 m clearance).
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
from scripts.train_visual import evaluate_model, make_env

ABLATIONS = {
    "baseline": {},
    "const_depth": {"const_depth": True},
    "no_velocity": {"no_velocity": True},
    "no_target_dir": {"no_target_dir": True},
    "all_inputs_ablated": {
        "const_depth": True,
        "no_velocity": True,
        "no_target_dir": True,
    },
}


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
    parser.add_argument("--ablation", default="all",
                        help="comma-separated subset or 'all'")
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--collision-ply", required=True)
    parser.add_argument("--collision-radius", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--avoidance-probability", type=float, default=None,
                        help="0.0 = clear-only, 1.0 = avoidance-only episodes; "
                             "default keeps the env default (0.5). Use to "
                             "isolate whether depth matters for obstacle "
                             "avoidance (visual necessity).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.avoidance_probability is not None \
            and not 0.0 <= args.avoidance_probability <= 1.0:
        raise ValueError("--avoidance-probability must be in [0, 1]")

    selected = (
        list(ABLATIONS) if args.ablation == "all"
        else [name.strip() for name in args.ablation.split(",")]
    )
    for name in selected:
        if name not in ABLATIONS:
            raise ValueError(f"unknown ablation {name!r}; choose from {list(ABLATIONS)}")

    scene_config = {
        "collision_ply_path": args.collision_ply,
        "auto_scene_bounds": True,
        "drone_collision_radius": args.collision_radius,
    }
    if args.avoidance_probability is not None:
        scene_config["avoidance_episode_probability"] = \
            args.avoidance_probability
    models = [item.split("=", 1) for item in args.model]
    results = {}
    for label, path in models:
        print(f"=== {label}: {path} ===", flush=True)
        policy = load_policy(path)
        results[label] = {}
        for name in selected:
            env = make_env(
                "clean", "gsplat", args.ply,
                ablation_config=ABLATIONS[name],
                scene_config=scene_config,
                alignment_config=args.alignment)
            eval_result = evaluate_model(
                policy, env, eval_episodes=args.episodes, base_seed=args.seed)
            results[label][name] = round(eval_result["success_rate"], 1)
            env.close()
            print(f"  {name}: SR={results[label][name]}%", flush=True)

    summary = {
        "models": {label: {"checkpoint": path} for label, path in models},
        "ablations": selected,
        "episodes_per_ablation": args.episodes,
        "results": results,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== ablation SR (%) ===")
    header = f"{'model':<24}" + "".join(f"{name:>18}" for name in selected)
    print(header)
    for label, ablation_results in results.items():
        print(f"{label:<24}" + "".join(
            f"{ablation_results.get(name, 0.0):>17.1f}%"
            for name in selected))
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
