#!/usr/bin/env python3
"""Generate a synthetic dense-obstacle corridor GS scene + collision cloud.

Five vertical walls along the +X flight path, each with an alternating
horizontal gap (slalom). The gaps force the drone to weave; the walls are
dense enough that blind (depth-free) navigation fails, so depth becomes the
only obstacle-avoidance signal. This is the D3-lite visual-necessity task.

Outputs two .ply files: a gsplat scene (rendered depth) and a collision
point cloud (Gaussian centers). Isolated experiment; delete to revert.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from plyfile import PlyData, PlyElement


def build_wall_centers(x, gap_center, gap_half, z_lo, z_hi, spacing):
    """Gaussian centers for one wall plane at fixed x."""
    y = np.arange(-3.0, 3.0 + spacing * 0.5, spacing)
    z = np.arange(z_lo, z_hi + spacing * 0.5, spacing)
    centers = []
    for yy in y:
        if gap_center - gap_half <= yy <= gap_center + gap_half:
            continue  # leave the gap
        for zz in z:
            centers.append((x, yy, zz))
    return np.asarray(centers, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--spacing", type=float, default=0.25,
                        help="Gaussian spacing inside the walls (m)")
    parser.add_argument("--scale", type=float, default=0.18,
                        help="Gaussian scale (radius) per axis (m)")
    parser.add_argument("--gap-half", type=float, default=0.9,
                        help="half-width of each wall gap (m)")
    args = parser.parse_args()

    walls = [(-4.0, -1.5), (-2.0, 1.5), (0.0, -1.5), (2.0, 1.5), (4.0, -1.5)]
    z_lo, z_hi = 0.2, 3.5
    means = np.vstack([
        build_wall_centers(x, gap, args.gap_half, z_lo, z_hi, args.spacing)
        for x, gap in walls
    ]).astype(np.float32)
    n = len(means)
    print(f"wall Gaussians: {n}")

    # Floor + ceiling caps to keep the drone inside the corridor band.
    xs = np.arange(-6.0, 6.5, 0.5)
    ys = np.arange(-3.0, 3.5, 0.5)
    floor = np.stack([np.tile(xs, len(ys)), np.repeat(ys, len(xs)),
                      np.full(len(xs) * len(ys), 0.1)], axis=-1).astype(np.float32)
    ceil = np.stack([np.tile(xs, len(ys)), np.repeat(ys, len(xs)),
                     np.full(len(xs) * len(ys), 3.6)], axis=-1).astype(np.float32)
    means = np.vstack([means, floor, ceil])
    n = len(means)

    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))
    scales = np.full((n, 3), args.scale, dtype=np.float32)
    opacity_logits = np.full((n,), 5.0, dtype=np.float32)  # sigmoid ~0.99
    sh0 = np.full((n, 3), 0.6, dtype=np.float32)  # gray

    vertex = np.zeros(
        n,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
               ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
               ("opacity", "f4"),
               ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")],
    )
    vertex["x"], vertex["y"], vertex["z"] = means.T
    vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"] = quats.T
    vertex["scale_0"], vertex["scale_1"], vertex["scale_2"] = scales.T
    vertex["opacity"] = opacity_logits
    vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"] = sh0.T

    os.makedirs(args.out_dir, exist_ok=True)
    scene_path = os.path.join(args.out_dir, "dense_corridor.ply")
    PlyData([PlyElement.describe(vertex, "vertex")]).write(scene_path)
    print(f"scene saved: {scene_path}")

    # Collision cloud: densify each Gaussian into small samples so the
    # KD-tree collision matches the rendered solid volume.
    rng = np.random.default_rng(20260803)
    per_gaussian = 12
    offsets = rng.normal(0.0, args.scale, size=(n, per_gaussian, 3)).astype(np.float32)
    collision = (means[:, None, :] + offsets).reshape(-1, 3)
    collision_vertex = np.zeros(
        len(collision),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    collision_vertex["x"], collision_vertex["y"], collision_vertex["z"] = collision.T
    cloud_path = os.path.join(args.out_dir, "dense_corridor_collision.ply")
    PlyData([PlyElement.describe(collision_vertex, "vertex")]).write(cloud_path)
    print(f"collision cloud saved: {cloud_path} ({len(collision)} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
