#!/usr/bin/env python3
"""Clean the low-fidelity sv_1007 reconstruction.

Low-fidelity 3DGS output mixes real structure with artifact "noise": tiny
disconnected Gaussian blobs (a few voxels) that are not real obstacles. They
render as spurious depth spikes and inflate the apparent obstacle count.

This script:
  1. Drops NaN / inf points (low-fidelity reconstructions occasionally emit them).
  2. Splits the scene into a ground slab (z < --ground-z) and everything above it,
     so the floor does not merge every obstacle into one connected component.
  3. Connected-component labels the non-ground points on a voxel grid.
  4. Keeps ground (unless --drop-ground) plus clusters >= --min-cluster-voxels.
  5. Drops the remaining noise clusters.
  6. Reports the honest breakdown (ground / N obstacle clusters / M noise removed),
     correcting inflated "16 obstacle clusters"-style counts from the raw cloud.
  7. Writes a cleaned gsplat .ply (renderer), a matching .ply (collision points,
     the format the env's --collision-ply expects), and a .npy (collision array).

Measured on sv_1007 (--voxel 0.5 --ground-z 0.5 --min-cluster-voxels 8):
  raw cloud at 0.5 m voxels  -> 16 connected components
  after cleaning             -> ground + 5 obstacle clusters, 13 noise removed
  Most of the raw "16" were isolated 1-3 voxel blobs, not real obstacles.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from plyfile import PlyData, PlyElement
from scipy import ndimage


def _cluster(pts: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-point cluster label, per-cluster voxel size, and cluster count.

    `pts` must be non-empty. Empty voxels are excluded from the occupancy
    grid, so the cluster size is the number of *occupied* voxels.
    """
    lo = pts.min(0)
    idx = np.floor((pts - lo) / voxel).astype(np.int64)
    shape = tuple(int(m) + 1 for m in idx.max(0))
    occ = np.zeros(shape, dtype=bool)
    occ[tuple(idx.T)] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())
    return lab[tuple(idx.T)], sizes, n


def _cluster_table(pts, plab, sizes, cluster_ids):
    """Human-readable table of the kept obstacle clusters (voxel/point counts,
    bounding box). `cluster_ids` are sorted largest-first for readability."""
    rows = []
    for cid in cluster_ids:
        m = plab == cid
        c = pts[m]
        lo, hi = c.min(0), c.max(0)
        rows.append(
            f"  cluster {cid:2d}: voxels={sizes[cid]:5d} pts={int(m.sum()):6d} "
            f"x[{lo[0]:6.2f},{hi[0]:6.2f}] y[{lo[1]:6.2f},{hi[1]:6.2f}] "
            f"z[{lo[2]:5.2f},{hi[2]:5.2f}]"
        )
    return "\n".join(rows)


def _json_report(args, pts, is_ground, obstacle_idx, plab, sizes, n,
                 kept_clusters, mask, total_points):
    """Structured summary for the cleaning run (used for reports/tests)."""
    kept_ids = sorted(kept_clusters, key=lambda c: -int(sizes[c]))
    report = {
        "source_ply": args.ply,
        "params": {
            "voxel": args.voxel,
            "min_cluster_voxels": args.min_cluster_voxels,
            "ground_z": args.ground_z,
            "drop_ground": args.drop_ground,
        },
        "total_points": int(total_points),
        "ground_points": int(is_ground.sum()),
        "raw_clusters_non_ground": int(n),
        "kept_obstacle_clusters": int(len(kept_ids)),
        "noise_clusters_removed": int(n - len(kept_ids)),
        "noise_points_removed": int((~mask).sum()),
        "kept_points": int(mask.sum()),
        "kept_clusters": [
            {
                "cluster_id": int(cid),
                "voxels": int(sizes[cid]),
                "points": int((plab == cid).sum()),
                "bbox": {
                    "min": [float(x) for x in pts[plab == cid].min(0)],
                    "max": [float(x) for x in pts[plab == cid].max(0)],
                },
            }
            for cid in kept_ids
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True, help="source gsplat .ply")
    parser.add_argument("--voxel", type=float, default=0.5)
    parser.add_argument("--min-cluster-voxels", type=int, default=8,
                        help="keep obstacle clusters with >= this many occupied voxels")
    parser.add_argument("--ground-z", type=float, default=0.5,
                        help="points with z < this are treated as the ground slab")
    parser.add_argument("--drop-ground", action="store_true",
                        help="exclude the ground slab from the outputs (default: keep)")
    parser.add_argument("--out-ply", required=True,
                        help="cleaned gsplat .ply (renderer)")
    parser.add_argument("--out-collision-ply", default=None,
                        help="cleaned collision .ply (xyz points, for --collision-ply). "
                             "Default: --out-ply with '.ply' replaced by '_collision.ply'")
    parser.add_argument("--out-npy", required=True,
                        help="cleaned collision points .npy (N,3 array)")
    parser.add_argument("--report-json", default=None,
                        help="optional structured stats report (.json)")
    args = parser.parse_args()

    if args.out_collision_ply is None:
        args.out_collision_ply = args.out_ply[:-len(".ply")] + "_collision.ply"
        print(f"note: --out-collision-ply defaults to {args.out_collision_ply}")

    v = PlyData.read(args.ply)["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)

    # 1. drop invalid points so downstream voxelization stays sane
    good = np.isfinite(pts).all(1)
    if not good.all():
        print(f"note: dropped {(~good).sum()} non-finite points")
        v.data = v.data[good]
        pts = pts[good]

    # 2. ground slab vs. everything above it
    is_ground = pts[:, 2] < args.ground_z

    # 3. cluster the non-ground points only, so the floor does not glue
    #    every obstacle into one giant connected component
    obstacle_idx = np.flatnonzero(~is_ground)
    if obstacle_idx.size == 0:
        print("no non-ground points; nothing to clean")
        return 1
    plab, sizes, n = _cluster(pts[obstacle_idx], args.voxel)

    # 4-5. keep big obstacle clusters, drop the tiny noise blobs
    keep_cluster = sizes[plab] >= args.min_cluster_voxels
    kept_clusters = np.unique(plab[keep_cluster])
    n_noise = n - kept_clusters.size

    mask = np.zeros(len(pts), dtype=bool)
    mask[obstacle_idx] = keep_cluster
    if not args.drop_ground:
        mask |= is_ground

    print(f"total points: {len(pts)}")
    print(f"ground slab (z<{args.ground_z}): {int(is_ground.sum())} pts"
          f"{' (dropped)' if args.drop_ground else ''}")
    print(f"obstacle clusters >= {args.min_cluster_voxels} voxels: {kept_clusters.size}")
    print(f"noise clusters removed: {n_noise}")
    print(f"kept: {mask.sum()} ({mask.mean()*100:.1f}%), "
          f"removed noise points: {int((~mask).sum())}")
    print(f"kept obstacle clusters:")
    kept_ids = sorted(kept_clusters, key=lambda c: -int(sizes[c]))
    print(_cluster_table(pts[obstacle_idx], plab, sizes, kept_ids))

    # 7. write cleaned gsplat ply (all original vertex fields preserved) +
    #    collision outputs in both formats the pipeline understands
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)), exist_ok=True)
    PlyData([PlyElement.describe(v.data[mask], "vertex")]).write(args.out_ply)

    clean_pts = pts[mask].astype(np.float32)
    collision_vertex = np.zeros(
        len(clean_pts),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    collision_vertex["x"], collision_vertex["y"], collision_vertex["z"] = clean_pts.T
    os.makedirs(os.path.dirname(os.path.abspath(args.out_collision_ply)),
                exist_ok=True)
    PlyData([PlyElement.describe(collision_vertex, "vertex")]).write(
        args.out_collision_ply)
    np.save(args.out_npy, clean_pts)
    print(f"wrote {args.out_ply}, {args.out_collision_ply}, {args.out_npy}")

    if args.report_json:
        report = _json_report(args, pts[obstacle_idx], is_ground, obstacle_idx,
                              plab, sizes, n, kept_clusters, mask,
                              total_points=len(pts))
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
