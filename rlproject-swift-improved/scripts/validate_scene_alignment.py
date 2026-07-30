#!/usr/bin/env python3
"""Compare rendered forward depth with the aligned collision point cloud."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.visual_drone_env import VisualDroneEnv


def cloud_forward_depth(points, position, rotation, cone_ratio):
    camera_points = (points - position[None, :]) @ rotation
    z = camera_points[:, 2]
    forward = z > 0.1
    z_safe = np.maximum(z, 1e-6)
    in_cone = (
        forward
        & (np.abs(camera_points[:, 0] / z_safe) <= cone_ratio)
        & (np.abs(camera_points[:, 1] / z_safe) <= cone_ratio)
    )
    if not np.any(in_cone):
        return None
    return float(np.min(z[in_cone]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--collision-ply", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    env = VisualDroneEnv({
        "renderer": "gsplat",
        "ply_path": args.ply,
        "collision_ply_path": args.collision_ply,
        "auto_scene_bounds": True,
        "camera_tracks_motion": True,
        "avoidance_episode_probability": 0.5,
    })
    records = []
    radius = args.patch_radius
    centre = 32
    cone_ratio = (radius + 0.5) / 32.0
    for index in range(args.samples):
        obs, reset_info = env.reset(seed=args.seed + index)
        position = env.state[:3].copy()
        quaternion = env._camera_quaternion(position, env.state[3:6])
        rotation = env.renderer._compute_c2w(
            position, quaternion)[:3, :3]
        cloud_depth = cloud_forward_depth(
            env.scene_geometry.points, position, rotation, cone_ratio)
        patch = obs["depth"][
            centre - radius:centre + radius + 1,
            centre - radius:centre + radius + 1,
            0,
        ]
        valid = patch[patch < env.renderer.max_depth - 1e-4]
        render_depth = float(np.min(valid)) if valid.size else None
        record = {
            "index": index,
            "requires_avoidance": reset_info["requires_avoidance"],
            "position": position.tolist(),
            "render_depth": render_depth,
            "cloud_depth": cloud_depth,
        }
        if render_depth is not None and cloud_depth is not None:
            record["abs_error"] = abs(render_depth - cloud_depth)
        records.append(record)
    env.close()

    comparable = [row for row in records if "abs_error" in row]
    render_values = np.array(
        [row["render_depth"] for row in comparable], dtype=np.float64)
    cloud_values = np.array(
        [row["cloud_depth"] for row in comparable], dtype=np.float64)
    errors = np.abs(render_values - cloud_values)
    correlation = (
        float(np.corrcoef(render_values, cloud_values)[0, 1])
        if len(comparable) >= 3
        and np.std(render_values) > 0
        and np.std(cloud_values) > 0
        else None
    )
    summary = {
        "samples": args.samples,
        "comparable_samples": len(comparable),
        "comparable_fraction": len(comparable) / max(args.samples, 1),
        "median_abs_error": (
            float(np.median(errors)) if len(errors) else None),
        "p90_abs_error": (
            float(np.percentile(errors, 90)) if len(errors) else None),
        "correlation": correlation,
        "provisional_pass": (
            len(comparable) >= 0.6 * args.samples
            and float(np.median(errors)) <= 0.75
            and correlation is not None
            and correlation >= 0.5
        ) if len(errors) else False,
    }
    output = {"summary": summary, "records": records}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
