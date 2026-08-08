#!/usr/bin/env python3
"""Evaluate a behavioral-cloning policy on a scene (SR comparison vs RL).

Loads a BCPolicy checkpoint and runs the standard env/evaluate loop,
producing success_rate comparable to the RL baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_bc import BCPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--collision-ply", required=True)
    parser.add_argument("--alignment", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from scripts.train_visual import evaluate_model, make_env

    model = BCPolicy()
    model.load_state_dict(torch.load(args.model, map_location="cpu",
                                     weights_only=True)["model_state_dict"])
    model.eval()

    def select_action(obs, deterministic=True):
        depth = torch.tensor(obs["depth"], dtype=torch.float32
                             ).unsqueeze(0).permute(0, 3, 1, 2)
        vec = torch.tensor(obs["vec"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return model(depth, vec).squeeze(0).numpy()

    # Wrap model with a select_action interface compatible with evaluate_model.
    class _Agent:
        def select_action(self, observation, deterministic=True):
            return select_action(observation, deterministic)

    scene_config = {
        "collision_ply_path": args.collision_ply,
        "drone_collision_radius": 0.5,
        "auto_scene_bounds": True,
    }
    env = make_env("clean", "gsplat", args.ply,
                   scene_config=scene_config, alignment_config=args.alignment)
    result = evaluate_model(_Agent(), env, eval_episodes=args.episodes,
                            base_seed=args.seed)
    env.close()

    summary = {
        "model": args.model,
        "arch": "bc_cnn_mlp",
        "episodes": args.episodes,
        "seed": args.seed,
        "success_rate": round(result["success_rate"], 1),
        "collision_rate": round(result["collision_rate"], 1),
        "timeout_rate": round(result["timeout_rate"], 1),
        "avg_reward": round(float(result["avg_reward"]), 3),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"BC SR={summary['success_rate']}% CR={summary['collision_rate']}%")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
