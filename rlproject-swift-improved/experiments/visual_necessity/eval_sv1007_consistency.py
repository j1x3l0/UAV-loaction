#!/usr/bin/env python3
"""Evaluate the sv_1007 baseline model on raw vs cleaned collision geometry.

One model (trained on the raw cloud) is evaluated on the same scene with the
collision cloud swapped: raw (as-trained) vs cleaned (sv_1007 low-fidelity
cleanup, ground + 5 obstacle clusters). If SR stays within a small band, the
cleanup is verified not to change the task — a zero-retrain consistency check.

Isolated experiment; does not modify any core code.
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


def load_policy(checkpoint: str):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy = VisualPPO(vec_dim=6, action_dim=3)
    policy.model.load_state_dict(data["model_state_dict"], strict=True)
    policy.model.eval()
    return policy


def _run(policy, ply, collision_ply, alignment, episodes, seed):
    """Evaluate on one collision geometry; returns SR + per-episode results."""
    scene_config = {
        "collision_ply_path": collision_ply,
        "drone_collision_radius": 0.5,
        "auto_scene_bounds": True,
    }
    env = make_env("clean", "gsplat", ply,
                   scene_config=scene_config,
                   alignment_config=alignment)
    try:
        result = evaluate_model(policy, env, eval_episodes=episodes,
                                base_seed=seed)
        detail = [row["result"] for row in result["episodes_detail"]]
        return {
            "success_rate": round(result["success_rate"], 1),
            "collision_rate": round(result["collision_rate"], 1),
            "timeout_rate": round(result["timeout_rate"], 1),
            "avg_reward": round(float(result["avg_reward"]), 3),
            "sr_ci_low": round(result["sr_ci_low"], 1),
            "sr_ci_high": round(result["sr_ci_high"], 1),
            "episodes_detail": detail,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ply", required=True,
                        help="renderer gsplat ply (kept identical across runs)")
    parser.add_argument("--raw-collision-ply", required=True)
    parser.add_argument("--clean-collision-ply", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    policy = load_policy(args.model)

    results = {
        "model": args.model,
        "ply": args.ply,
        "episodes": args.episodes,
        "seed": args.seed,
        "raw_collision_ply": args.raw_collision_ply,
        "clean_collision_ply": args.clean_collision_ply,
    }
    for label, collision in [
        ("raw", args.raw_collision_ply),
        ("cleaned", args.clean_collision_ply),
    ]:
        r = _run(policy, args.ply, collision, args.alignment,
                 args.episodes, args.seed)
        results[label] = r
        print(f"{label}: SR={r['success_rate']}% "
              f"CR={r['collision_rate']}% avgR={r['avg_reward']}",
              flush=True)

    raw_vec = np.array([1 if r == "success" else 0
                        for r in results["raw"]["episodes_detail"]])
    clean_vec = np.array([1 if r == "success" else 0
                          for r in results["cleaned"]["episodes_detail"]])
    diff = results["raw"]["success_rate"] - results["cleaned"]["success_rate"]
    results["delta_pp"] = round(diff, 1)
    results["paired_discordant"] = int((raw_vec != clean_vec).sum())
    results["consistent"] = abs(diff) <= 3.0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"delta (raw - cleaned) = {results['delta_pp']}pp | "
          f"discordant pairs: {results['paired_discordant']} | "
          f"consistent(<=3pp): {results['consistent']}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
