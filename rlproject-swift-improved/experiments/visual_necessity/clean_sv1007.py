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
  7. Writes a cleaned gsplat .ply (renderer) + a matching .npy (collision points).

Measured on sv_1007 (--voxel 0.5 --ground-z 0.5 --min-cluster-voxels 8):
  raw cloud at 0.5 m voxels  -> 16 connected components
  after cleaning             -> ground + 5 obstacle clusters, 13 noise removed
  Most of the raw "16" were isolated 1-3 voxel blobs, not real obstacles.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--out-npy", required=True)
    args = parser.parse_args()

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

    # 7. write filtered gsplat ply (all original vertex fields preserved)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)), exist_ok=True)
    PlyData([PlyElement.describe(v.data[mask], "vertex")]).write(args.out_ply)
    np.save(args.out_npy, pts[mask].astype(np.float32))
    print(f"wrote {args.out_ply} and {args.out_npy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
