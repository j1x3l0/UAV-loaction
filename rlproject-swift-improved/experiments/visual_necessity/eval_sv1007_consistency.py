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
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from core.visual_ppo_agent import VisualPPO
from scripts.train_visual import evaluate_model, make_env


def _paired_stats(raw, clean, n_boot=10000, seed=20260805):
    """Paired McNemar + paired bootstrap 95% CI for (raw_success - clean_success).

    Raw/clean are per-episode booleans from the SAME rollout seeds, so a
    paired test is the right statistic. The bootstrap CI including 0 means
    the raw-vs-cleaned SR gap is within evaluation noise.
    """
    b = int(np.sum(raw & ~clean))
    c = int(np.sum(~raw & clean))
    total = b + c
    if total == 0:
        mcnemar_p = 1.0
    else:
        chi2 = (abs(b - c) - 1.0) ** 2 / total
        mcnemar_p = float(stats.chi2.sf(chi2, df=1))
    rng = np.random.default_rng(seed)
    n = len(raw)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(raw[idx].mean() - clean[idx].mean()))
    diffs = np.asarray(diffs)
    return {
        "mcnemar_p": mcnemar_p,
        "discordant_b_raw_success": int(b),
        "discordant_c_clean_success": int(c),
        "bootstrap_95ci_pp": [
            round(float(np.percentile(diffs, 2.5)) * 100, 1),
            round(float(np.percentile(diffs, 97.5)) * 100, 1),
        ],
    }


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

    raw_vec = np.array([r == "success"
                        for r in results["raw"]["episodes_detail"]])
    clean_vec = np.array([r == "success"
                          for r in results["cleaned"]["episodes_detail"]])
    diff = results["raw"]["success_rate"] - results["cleaned"]["success_rate"]
    results["delta_pp"] = round(diff, 1)
    results["paired"] = _paired_stats(raw_vec, clean_vec)
    # Consistency = the gap is within paired-evaluation noise (CI includes 0
    # and McNemar is not significant), NOT an arbitrary pp threshold. A 3pp
    # hard cutoff on 50 episodes misreads noise as a real change.
    ci_lo, ci_hi = results["paired"]["bootstrap_95ci_pp"]
    results["consistent"] = bool(
        results["paired"]["mcnemar_p"] >= 0.05 and ci_lo <= 0.0 <= ci_hi)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    p = results["paired"]
    print(f"delta (raw - cleaned) = {results['delta_pp']}pp | "
          f"McNemar p={p['mcnemar_p']:.3f} | "
          f"bootstrap 95% CI {p['bootstrap_95ci_pp']}pp | "
          f"consistent (p>=0.05 & CI incl. 0): {results['consistent']}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
