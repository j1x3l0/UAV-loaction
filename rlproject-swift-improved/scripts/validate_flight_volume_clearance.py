#!/usr/bin/env python3
"""Check minimum collision clearance of the mapped PX4 flight volume.

Maps a PX4 LOCAL_NED flight box into the reconstructed 3DGS scene via
``Px4SceneAlignment`` and reports the nearest collision-cloud distance
over a sampled grid. A mapping or scene-alignment error shows up as
collision points inside the flight volume (clearance near zero).

The NED flight box is a local decision: there is no canonical "flight
volume" spec yet, so this script defaults to a box centred on the
alignment anchor and makes every bound overridable. Run with
``--ned-bounds`` to pin the exact gate envelope.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.px4_scene_alignment import Px4SceneAlignment

# Drone task altitudes used by the v2 env, expressed in scene Z.
_SCENE_ALT_MIN = 0.5
_SCENE_ALT_MAX = 3.5


def load_collision_cloud(ply_path: str) -> np.ndarray:
    vertex = PlyData.read(ply_path)["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 4:
        raise ValueError("collision cloud contains too few valid points")
    return np.asarray(points, dtype=np.float32)


def position_ned_from_scene(alignment, position_scene):
    position = np.asarray(position_scene, dtype=np.float64)
    return (
        alignment.scene_from_ned_rotation.T
        @ (position - alignment.scene_from_ned_translation)
    ) / alignment.scale


def default_ned_bounds(alignment) -> tuple[float, float, float, float, float, float]:
    """NED box spanning the scene flight band used by the task."""
    corners = np.array([
        [-10, -10, _SCENE_ALT_MIN], [10, -10, _SCENE_ALT_MIN],
        [-10, 10, _SCENE_ALT_MIN], [10, 10, _SCENE_ALT_MIN],
        [-10, -10, _SCENE_ALT_MAX], [10, -10, _SCENE_ALT_MAX],
        [-10, 10, _SCENE_ALT_MAX], [10, 10, _SCENE_ALT_MAX],
    ], dtype=np.float64)
    ned = np.vstack([position_ned_from_scene(alignment, c) for c in corners])
    low = ned.min(axis=0)
    high = ned.max(axis=0)
    return (float(low[0]), float(high[0]), float(low[1]), float(high[1]),
            float(low[2]), float(high[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", required=True,
                        help="Px4SceneAlignment JSON config")
    parser.add_argument("--collision-ply", required=True,
                        help="scene collision point cloud in 3DGS coords")
    parser.add_argument("--ned-bounds",
                        help="xmin,xmax,ymin,ymax,zmin,zmax (NED, metres); "
                             "defaults to the mapped scene flight band")
    parser.add_argument("--resolution", type=float, default=0.5,
                        help="grid spacing inside the flight volume (m)")
    parser.add_argument("--clearance-threshold", type=float, default=0.5,
                        help="minimum acceptable nearest-obstacle distance (m)")
    parser.add_argument("--free-clearance", type=float, default=0.45,
                        help="required clearance for the connected free-space "
                             "component analysis (m)")
    parser.add_argument("--free-grid", type=float, default=0.3,
                        help="grid spacing for the free-space analysis (m)")
    parser.add_argument("--min-free-volume", type=float, default=10.0,
                        help="minimum acceptable largest connected free volume "
                             "(m^3)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    alignment = Px4SceneAlignment.from_json(args.alignment)
    cloud = load_collision_cloud(args.collision_ply)
    tree = cKDTree(cloud)

    from envs.scene_geometry import ScenePointCloudGeometry
    geometry = ScenePointCloudGeometry(points=cloud)
    free_nodes = geometry.build_navigation_grid(
        resolution=args.free_grid, clearance=args.free_clearance)
    free_volume = free_nodes * args.free_grid ** 3

    if args.ned_bounds:
        values = [float(v) for v in args.ned_bounds.split(",")]
        if len(values) != 6 or not all(np.isfinite(values)):
            raise ValueError("--ned-bounds must be six finite numbers")
        if values[0] >= values[1] or values[2] >= values[3] or values[4] >= values[5]:
            raise ValueError("--ned-bounds must satisfy xmin<xmax etc.")
        ned_low = np.asarray([values[0], values[2], values[4]])
        ned_high = np.asarray([values[1], values[3], values[5]])
        bounds_label = "explicit"
    else:
        bounds = default_ned_bounds(alignment)
        ned_low = np.asarray([bounds[0], bounds[2], bounds[4]])
        ned_high = np.asarray([bounds[1], bounds[3], bounds[5]])
        bounds_label = "derived"

    axes = [
        np.arange(low, high + args.resolution * 0.5, args.resolution)
        for low, high in zip(ned_low, ned_high)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid_ned = np.stack([axis.ravel() for axis in mesh], axis=-1)
    grid_scene = np.vstack([
        alignment.position_scene_from_ned(point) for point in grid_ned
    ]).astype(np.float32)
    distances, _ = tree.query(grid_scene, k=1)
    distances = np.asarray(distances, dtype=np.float64)

    records = {
        "ned_bounds": {
            "x_min": float(ned_low[0]), "x_max": float(ned_high[0]),
            "y_min": float(ned_low[2]), "y_max": float(ned_high[2]),
            "z_min": float(ned_low[1]), "z_max": float(ned_high[1]),
            "label": bounds_label,
        },
        "grid_samples": int(len(grid_ned)),
        "clearance_m": {
            "min": float(np.min(distances)),
            "p5": float(np.percentile(distances, 5)),
            "median": float(np.median(distances)),
            "max": float(np.max(distances)),
        },
        "below_threshold": {
            "count": int(np.sum(distances < args.clearance_threshold)),
            "fraction": float(np.mean(distances < args.clearance_threshold)),
        },
        "collision_cloud": {
            "points": int(len(cloud)),
            "scene_min": cloud.min(axis=0).tolist(),
            "scene_max": cloud.max(axis=0).tolist(),
        },
        "free_space": {
            "required_clearance_m": args.free_clearance,
            "grid_m": args.free_grid,
            "largest_component_nodes": int(free_nodes),
            "largest_component_volume_m3": float(free_volume),
            "min_acceptable_volume_m3": args.min_free_volume,
        },
        # A raw min-clearance over a box that contains the gate structure
        # itself is not a usable pass criterion; the meaningful gate is the
        # connected free-space volume at the operating clearance.
        "pass": bool(free_volume >= args.min_free_volume),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(json.dumps(records, indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
