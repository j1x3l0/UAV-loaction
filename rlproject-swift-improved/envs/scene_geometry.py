"""Collision geometry tied to the same reconstructed scene as the renderer."""

import numpy as np
from scipy.spatial import cKDTree


class ScenePointCloudGeometry:
    """KD-tree collision queries and free-space sampling for a scene cloud."""

    def __init__(self, ply_path=None, points=None, bounds_percentiles=(1, 99),
                 boundary_margin=0.25):
        if points is None:
            if not ply_path:
                raise ValueError("ply_path or points is required")
            from plyfile import PlyData
            vertex = PlyData.read(ply_path)["vertex"]
            points = np.stack(
                [vertex["x"], vertex["y"], vertex["z"]], axis=-1)
        self.points = np.asarray(points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("scene points must have shape (N, 3)")
        finite = np.isfinite(self.points).all(axis=1)
        self.points = self.points[finite]
        if len(self.points) < 4:
            raise ValueError("scene point cloud contains too few valid points")
        self.tree = cKDTree(self.points)
        low, high = np.percentile(
            self.points, bounds_percentiles, axis=0)
        margin = np.broadcast_to(
            np.asarray(boundary_margin, dtype=np.float32), (3,))
        self.boundary_min = low + margin
        self.boundary_max = high - margin
        if np.any(self.boundary_min >= self.boundary_max):
            raise ValueError("scene bounds collapse after applying margin")
        self.navigation_points = None
        self.navigation_tree = None
        self.navigation_graph = None

    def nearest_distance(self, position):
        distance, _ = self.tree.query(
            np.asarray(position, dtype=np.float32), k=1)
        return float(distance)

    def collides(self, position, radius):
        return self.nearest_distance(position) <= float(radius)

    def sample_free(self, rng, clearance, bounds_min=None, bounds_max=None,
                    max_attempts=2000):
        low = self.boundary_min if bounds_min is None else np.asarray(bounds_min)
        high = self.boundary_max if bounds_max is None else np.asarray(bounds_max)
        for _ in range(max_attempts):
            candidate = rng.uniform(low, high).astype(np.float32)
            if self.nearest_distance(candidate) > clearance:
                return candidate
        raise RuntimeError(
            f"failed to sample free position after {max_attempts} attempts")

    def segment_min_clearance(self, start, end, samples=64):
        fractions = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
        line = (
            np.asarray(start, dtype=np.float32)[None, :] * (1.0 - fractions)
            + np.asarray(end, dtype=np.float32)[None, :] * fractions
        )
        distances, _ = self.tree.query(line, k=1)
        return float(np.min(distances))

    def build_navigation_grid(self, resolution=0.3, clearance=0.45):
        """Build the largest connected free-space component."""
        from scipy.ndimage import generate_binary_structure, label
        from scipy.sparse import csr_matrix

        axes = [
            np.arange(low, high + resolution * 0.5, resolution)
            for low, high in zip(self.boundary_min, self.boundary_max)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        grid_points = np.stack([axis.ravel() for axis in mesh], axis=-1)
        distances, _ = self.tree.query(grid_points, k=1)
        free = (distances > clearance).reshape(
            tuple(len(axis) for axis in axes))
        labels, count = label(
            free, structure=generate_binary_structure(3, 1))
        if count == 0:
            raise RuntimeError("scene contains no connected free-space grid")
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = int(np.argmax(sizes))
        component_grid = labels == largest
        component = component_grid.ravel()
        self.navigation_points = grid_points[component].astype(np.float32)
        if len(self.navigation_points) < 2:
            raise RuntimeError("largest free-space component is too small")
        self.navigation_tree = cKDTree(self.navigation_points)

        # Six-neighbour sparse graph for target-conditioned shortest paths.
        node_ids = np.full(component_grid.shape, -1, dtype=np.int64)
        node_ids[component_grid] = np.arange(len(self.navigation_points))
        edge_starts = []
        edge_ends = []
        edge_weights = []
        for axis in range(3):
            left = [slice(None)] * 3
            right = [slice(None)] * 3
            left[axis] = slice(0, -1)
            right[axis] = slice(1, None)
            left = tuple(left)
            right = tuple(right)
            connected = component_grid[left] & component_grid[right]
            starts = node_ids[left][connected]
            ends = node_ids[right][connected]
            weight = float(axes[axis][1] - axes[axis][0])
            edge_starts.extend([starts, ends])
            edge_ends.extend([ends, starts])
            edge_weights.extend([
                np.full(len(starts), weight, dtype=np.float32),
                np.full(len(starts), weight, dtype=np.float32),
            ])
        rows = np.concatenate(edge_starts)
        cols = np.concatenate(edge_ends)
        weights = np.concatenate(edge_weights)
        self.navigation_graph = csr_matrix(
            (weights, (rows, cols)),
            shape=(len(self.navigation_points), len(self.navigation_points)),
        )
        return len(self.navigation_points)

    def geodesic_distance_field(self, target):
        """Return shortest free-space distance from every node to target."""
        if self.navigation_graph is None or self.navigation_tree is None:
            raise RuntimeError("build_navigation_grid must be called first")
        from scipy.sparse.csgraph import dijkstra

        _, target_index = self.navigation_tree.query(
            np.asarray(target, dtype=np.float32), k=1)
        return np.asarray(
            dijkstra(
                self.navigation_graph,
                directed=False,
                indices=int(target_index),
            ),
            dtype=np.float32,
        )

    def geodesic_distance(self, position, distance_field):
        """Approximate a smooth free-space distance at any position."""
        if self.navigation_tree is None:
            raise RuntimeError("build_navigation_grid must be called first")
        neighbour_count = min(8, len(self.navigation_points))
        node_offsets, node_indices = self.navigation_tree.query(
            np.asarray(position, dtype=np.float32), k=neighbour_count)
        node_offsets = np.atleast_1d(node_offsets)
        node_indices = np.atleast_1d(node_indices).astype(np.int64)
        candidates = distance_field[node_indices] + node_offsets
        return float(np.min(candidates))

    def shortest_path(self, start, target):
        """Return a collision-free grid path between arbitrary positions."""
        if self.navigation_graph is None or self.navigation_tree is None:
            raise RuntimeError("build_navigation_grid must be called first")
        from scipy.sparse.csgraph import dijkstra

        _, start_index = self.navigation_tree.query(
            np.asarray(start, dtype=np.float32), k=1)
        _, target_index = self.navigation_tree.query(
            np.asarray(target, dtype=np.float32), k=1)
        distances, predecessors = dijkstra(
            self.navigation_graph,
            directed=False,
            indices=int(start_index),
            return_predecessors=True,
        )
        if not np.isfinite(distances[int(target_index)]):
            raise RuntimeError("no navigation path exists between positions")
        node_path = [int(target_index)]
        while node_path[-1] != int(start_index):
            predecessor = int(predecessors[node_path[-1]])
            if predecessor < 0:
                raise RuntimeError("failed to reconstruct navigation path")
            node_path.append(predecessor)
        node_path.reverse()
        path = self.navigation_points[node_path]
        return np.vstack([
            np.asarray(start, dtype=np.float32),
            path,
            np.asarray(target, dtype=np.float32),
        ])

    def sample_reachable_pair(self, rng, min_distance=3.0,
                              blocked_probability=0.5,
                              collision_radius=0.25,
                              max_attempts=4000):
        """Sample a same-component pair, stratified by direct obstruction."""
        if self.navigation_points is None:
            raise RuntimeError("build_navigation_grid must be called first")
        require_blocked = (
            None if blocked_probability is None
            else bool(rng.random() < blocked_probability)
        )
        count = len(self.navigation_points)
        for _ in range(max_attempts):
            start = self.navigation_points[int(rng.integers(count))]
            target = self.navigation_points[int(rng.integers(count))]
            if np.linalg.norm(target - start) < min_distance:
                continue
            if require_blocked is not None:
                blocked = self.segment_min_clearance(
                    start, target) <= collision_radius
                if blocked != require_blocked:
                    continue
            return start.copy(), target.copy(), require_blocked
        raise RuntimeError(
            "failed to sample a reachable pair matching task stratum")
