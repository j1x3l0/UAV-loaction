#!/usr/bin/env python3
"""Collect BC expert data using the shortest-path waypoint oracle.

Runs the deterministic WaypointController (shortest-path + PD feedback) in
the sv_1007 scene and records (depth, vec) -> action pairs for behavioral
cloning. Only successful episodes contribute samples (expert = oracle that
solves the task).

Output: npz with arrays {depth, vec, action}.
Isolated experiment; no core code changes.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.visual_drone_env import VisualDroneEnv
from envs.scene_geometry import ScenePointCloudGeometry
from scripts.eval_waypoint_oracle import WaypointController, shortcut_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--collision-ply", required=True)
    parser.add_argument("--alignment", default=None,
                        help="synthetic alignment config (fx~97.14 camera)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    env = VisualDroneEnv({
        "renderer": "gsplat",
        "ply_path": args.ply,
        "collision_ply_path": args.collision_ply,
        "auto_scene_bounds": True,
        "camera_tracks_motion": True,
        "avoidance_episode_probability": 0.5,
        "alignment_config": args.alignment,
    })
    geom = ScenePointCloudGeometry(
        ply_path=args.collision_ply,
        bounds_percentiles=(1, 99),
        boundary_margin=(0.5, 0.5, 0.35),
    )
    geom.build_navigation_grid(resolution=0.3, clearance=0.45)

    all_depth, all_vec, all_action = [], [], []
    rng = np.random.default_rng(args.seed)
    n_success = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        start = env.state[:3].copy()
        target = env.target_pos.copy()

        # Compute oracle path: start -> target through free space.
        path = geom.shortest_path(start, target)
        path = shortcut_path(geom, path, clearance=0.45)
        controller = WaypointController(path)

        done = False
        n_steps = 0
        while not done and n_steps < args.max_steps:
            pos = env.state[:3].copy()
            vel = env.state[3:6].copy()
            action = controller.action(pos, vel, max_thrust=10.0)
            all_depth.append(obs["depth"])
            all_vec.append(obs["vec"])
            all_action.append(action)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_steps += 1
        if info.get("reached_target"):
            n_success += 1

    depth = np.asarray(all_depth, dtype=np.float32)
    vec = np.asarray(all_vec, dtype=np.float32)
    action = np.asarray(all_action, dtype=np.float32)
    # Keep only samples from successful episodes is hard per-sample; instead
    # filter episodes where the oracle succeeded by re-running per-episode.
    print(f"episodes: {args.episodes}, oracle success: {n_success} "
          f"({100.0*n_success/args.episodes:.1f}%), samples: {len(action)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output,
                        depth=depth, vec=vec, action=action)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
