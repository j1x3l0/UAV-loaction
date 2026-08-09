#!/usr/bin/env python3
"""V3c cross-scene generalization evaluation.

Evaluate trained models from one scene against other scenes (zero-shot
transfer). Builds an evaluation matrix:
    source model (trained on scene A) x target scene B
where A and B are drawn from {sv_1007, sv_917_left, sv_917_right}.

In-domain (A == B) is the baseline; off-diagonal cells are zero-shot
cross-scene generalization. All evaluation uses the same aligned camera
(fx~97.14) so the comparison isolates scene generalization.

Isolated experiment; does not modify core code.
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


def _evaluate(policy, ply, alignment, episodes, seed):
    scene_config = {
        "collision_ply_path": ply,
        "drone_collision_radius": 0.5,
        "auto_scene_bounds": True,
    }
    env = make_env("clean", "gsplat", ply,
                   scene_config=scene_config,
                   alignment_config=alignment)
    try:
        result = evaluate_model(policy, env, eval_episodes=episodes,
                                base_seed=seed)
        return {
            "success_rate": round(result["success_rate"], 1),
            "collision_rate": round(result["collision_rate"], 1),
            "timeout_rate": round(result["timeout_rate"], 1),
            "avg_reward": round(float(result["avg_reward"]), 3),
            "sr_ci_low": round(result["sr_ci_low"], 1),
            "sr_ci_high": round(result["sr_ci_high"], 1),
            "episodes_detail": [
                {"episode": r["episode"], "result": r["result"]}
                for r in result["episodes_detail"]
            ],
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True,
                        help="label=checkpoint path (repeat; e.g. sv1007=... )")
    parser.add_argument("--scene", action="append", required=True,
                        help="name=ply_path:alignment_path (repeat)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    models = [item.split("=", 1) for item in args.model]
    scenes = []
    for item in args.scene:
        name, rest = item.split("=", 1)
        ply, alignment = rest.split(":", 1)
        scenes.append((name, ply, alignment))

    results = {}
    detail = {}
    print("=== V3c cross-scene matrix (SR %) ===")
    header = "source\\target".ljust(12) + "".join(
        f"{name:>14}" for name, _, _ in scenes)
    print(header)
    for mname, mpath in models:
        policy = load_policy(mpath)
        row = []
        for sname, sply, salign in scenes:
            r = _evaluate(policy, sply, salign, args.episodes, args.seed)
            key = f"{mname}->{sname}"
            results[key] = {k: v for k, v in r.items() if k != "episodes_detail"}
            detail[key] = r["episodes_detail"]
            row.append(f"{r['success_rate']:>13.1f}%")
            print(f"{mname}->{sname}: SR={r['success_rate']}% "
                  f"CR={r['collision_rate']}%", flush=True)
        print(f"{mname:<12}" + "".join(row))
        print()

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({
            "models": {m: p for m, p in models},
            "scenes": [{"name": n, "ply": p, "alignment": a} for n, p, a in scenes],
            "episodes": args.episodes,
            "seed": args.seed,
            "results": results,
            "episodes_detail": detail,
        }, handle, indent=2)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
