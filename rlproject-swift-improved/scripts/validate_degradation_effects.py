"""Verify that each Phase V2 axis changes real-gsplat policy observations."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.degradation_utils import DEGRADATION_AXES
from envs.visual_drone_env import VisualDroneEnv


def compare(reference, candidate):
    delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {
        "mean_abs_depth_delta": float(delta.mean()),
        "max_abs_depth_delta": float(delta.max()),
        "changed_pixel_fraction": float(np.mean(delta > 1e-6)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = {"renderer": "gsplat", "ply_path": args.ply}
    clean_env = VisualDroneEnv(config=base)
    clean_obs, _ = clean_env.reset(seed=args.seed)
    clean_depth = clean_obs["depth"]
    clean_env.close()

    results = {}
    for axis, definition in DEGRADATION_AXES.items():
        baseline_level = definition["levels"][0]
        extreme_level = definition["levels"][-1]
        axis_results = {}
        for name, level in [
            ("baseline_level", baseline_level),
            ("extreme_level", extreme_level),
        ]:
            config = dict(base)
            config["degradation"] = {axis: level}
            env = VisualDroneEnv(config=config)
            obs, _ = env.reset(seed=args.seed)
            axis_results[name] = {
                "level": level,
                **compare(clean_depth, obs["depth"]),
            }
            env.close()
        results[axis] = axis_results

    payload = {
        "renderer": "gsplat",
        "ply": args.ply,
        "seed": args.seed,
        "results": results,
        "expected_negative_control": {
            "axis": "lighting",
            "reason": "The policy observation contains depth and vector state, not RGB.",
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
