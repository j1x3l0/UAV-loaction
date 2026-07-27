"""
validate_collision_geometry.py — 碰撞几何验证工具 v1

目的: 验证碰撞几何与渲染几何的一致性，确保实验结论有效性。
     发现视觉观测(深度图)与碰撞检测(球体)之间的空间错位。

架构位置: scripts/ (Application层)
数据流: env_config + renderer → 空间采样 → 深度图↔碰撞几何对比 → 可视化+报告
边界: 不修改环境、不评估策略、不训练模型
风险: 如果发现严重错位，需要修改环境定义而非本工具

使用方法:
  # Mock渲染器验证 (默认)
  python scripts/validate_collision_geometry.py

  # 真实3DGS渲染器验证
  python scripts/validate_collision_geometry.py --renderer gsplat --ply <path>

  # 完整验证 (含3DGS点云叠加)
  python scripts/validate_collision_geometry.py --renderer gsplat --ply <path> --output reports/collision_validation --visualize
"""

import numpy as np
import json, os, sys, argparse, logging, itertools
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field, asdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 数据模型
# ============================================================

@dataclass
class CollisionSphere:
    """一个碰撞检测球体"""
    x: float; y: float; z: float; radius: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def distance(self, pos: np.ndarray) -> float:
        return max(np.linalg.norm(pos - self.as_array()) - self.radius, 0.0)

    def contains(self, pos: np.ndarray) -> bool:
        return np.linalg.norm(pos - self.as_array()) <= self.radius + 0.5  # +drone radius


@dataclass
class ValidationSample:
    """单个采样点的验证数据"""
    position: Tuple[float, float, float]
    true_collision_dist: float              # 到最近碰撞球面的真实距离
    depth_min: float                        # 深度图最小值
    depth_mean: float                       # 深度图均值
    depth_at_obstacle_dir: float            # 障碍物方向上的深度值
    is_collision_true: bool                 # 根据碰撞几何是否碰撞
    is_collision_depth_estimated: bool      # 根据深度图是否估算为碰撞
    mismatch: float = 0.0                   # 错位程度 (负数=深度看到障碍但碰撞检测无)
    in_collision_geometry: bool = False     # 当前点在碰撞几何内


@dataclass
class ValidationReport:
    """验证报告"""
    timestamp: str = ""
    n_samples: int = 0
    mismatch_ratio: float = 0.0              # 错位采样点比例
    mean_mismatch: float = 0.0               # 平均错位距离
    max_mismatch: float = 0.0                # 最大错位距离
    phantom_collisions: int = 0              # 深度图无物但碰撞检测撞
    missed_collisions: int = 0               # 深度图有物但碰撞检测没撞
    collision_geometry_details: dict = field(default_factory=dict)
    gsplat_check: Optional[dict] = None      # 真实GS的碰撞几何对齐检查


# ============================================================
# 验证引擎
# ============================================================

class CollisionGeometryValidator:
    """碰撞几何验证器"""

    def __init__(self, config: dict):
        self.config = config
        self.renderer_type = config.get('renderer', 'mock')

        # --- 碰撞检测球体 (从环境定义提取) ---
        # source: visual_drone_env.py line 170-176
        self.collision_spheres = [
            CollisionSphere(x=2.0, y=2.0, z=3.0, radius=1.0),
            CollisionSphere(x=6.0, y=3.0, z=5.0, radius=1.0),
            CollisionSphere(x=3.0, y=7.0, z=4.0, radius=1.0),
        ]
        self.drone_radius = 0.5  # collision_threshold
        self.collision_threshold = self.drone_radius  # = 0.5

        # --- 环境边界 ---
        self.boundary_min = np.array([-10.0, -10.0, 0.0])
        self.boundary_max = np.array([10.0, 10.0, 10.0])

        # --- 创建环境 (用于获取渲染器) ---
        self._init_env()

        logger.info(f"碰撞几何验证器初始化 | renderer={self.renderer_type}")
        logger.info(f"  碰撞球体: {len(self.collision_spheres)}个")
        for i, s in enumerate(self.collision_spheres):
            logger.info(f"    S{i}: ({s.x:.1f}, {s.y:.1f}, {s.z:.1f}) r={s.radius}")

    def _init_env(self):
        """创建VisualDroneEnv实例 (仅读取渲染器, 不交互)"""
        from envs.visual_drone_env import VisualDroneEnv
        env_config = self.config.copy()
        # 用ablation const_depth确保深度图不被退化修改 (我们要原始几何)
        env_config['ablation'] = {}
        self.env = VisualDroneEnv(config=env_config)
        self.renderer = self.env.renderer

        # 提取Mock渲染器的内部障碍物 (如果是Mock)
        if self.renderer_type == 'mock':
            self.mock_obstacles = self.env._base_obstacles_for_render
            logger.info(f"  Mock渲染器障碍物: {len(self.mock_obstacles)}个")
            for i, obs in enumerate(self.mock_obstacles):
                logger.info(f"    O{i}: ({obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}) r={obs[3]}")

    def close(self):
        self.env.close()

    # ── 核心验证方法 ──

    def sample_positions(self, grid_size: int = 8) -> np.ndarray:
        """
        在环境空间生成均匀网格采样点

        Args:
            grid_size: 每个维度采样点数 (总点数 = grid_size^3)
        Returns:
            (N, 3) 位置数组
        """
        xs = np.linspace(self.boundary_min[0] + 0.5, self.boundary_max[0] - 0.5, grid_size)
        ys = np.linspace(self.boundary_min[1] + 0.5, self.boundary_max[1] - 0.5, grid_size)
        zs = np.linspace(self.boundary_min[2] + 0.5, self.boundary_max[2] - 0.5, max(3, grid_size // 2))
        positions = np.array(list(itertools.product(xs, ys, zs)))
        logger.info(f"生成采样点: {len(positions)}个 (grid {grid_size}×{grid_size}×{len(zs)})")
        return positions

    def render_depth_at(self, pos: np.ndarray) -> np.ndarray:
        """从指定位置渲染深度图"""
        from envs.visual_drone_env import MockGSRenderer

        if isinstance(self.renderer, MockGSRenderer):
            # Mock渲染器: 用内部障碍物渲染
            if self.renderer_type == 'gsplat':
                depth, _ = self.renderer.render(pos)
            else:
                depth, _ = self.renderer.render(pos, obstacles=self.mock_obstacles)
        else:
            depth, _ = self.renderer.render(pos)
        return depth

    def true_collision_distance(self, pos: np.ndarray) -> Tuple[float, int]:
        """
        计算到最近碰撞球面的真实距离
        Returns: (distance, nearest_sphere_idx)
        """
        min_dist = float('inf')
        nearest = -1
        for i, sphere in enumerate(self.collision_spheres):
            d = sphere.distance(pos)
            if d < min_dist:
                min_dist = d
                nearest = i
        return min_dist, nearest

    def depth_minimum_at_center(self, depth: np.ndarray) -> float:
        """深度图中心区域最小值 (代表正前方障碍物距离)"""
        H, W = depth.shape[:2]
        center = depth[H//2-4:H//2+4, W//2-4:W//2+4, 0]
        return float(center.min())

    def depth_obstacle_distance(self, depth: np.ndarray) -> Tuple[float, float]:
        """
        从深度图估算障碍物距离
        Returns: (min_depth_global, min_depth_center)
          - global: 全图最小值 (任意方向最近障碍)
          - center: 中心区域最小值 (前方向障碍)
        """
        global_min = float(depth[:, :, 0].min())
        center_min = self.depth_minimum_at_center(depth)
        return global_min, center_min

    def validate_point(self, pos: np.ndarray) -> ValidationSample:
        """验证单个采样点"""
        # 真实碰撞几何
        true_dist, nearest_idx = self.true_collision_distance(pos)
        is_collision_true = true_dist <= self.collision_threshold

        # 渲染深度
        depth = self.render_depth_at(pos)
        depth_global_min, depth_center_min = self.depth_obstacle_distance(depth)
        depth_mean = float(depth[:, :, 0].mean())

        # 碰撞球的方位对应深度
        sphere_dists = []
        for sphere in self.collision_spheres:
            sphere_dir = sphere.as_array() - pos
            sphere_dist = np.linalg.norm(sphere_dir)
            sphere_dists.append(sphere_dist)

        # 深度估算: 如果深度最小值 < 碰撞球距离 → 深度看到的东西比碰撞几何更近
        # is_collision_depth_estimated: 如果深度最小值 < 1.5m (阈值+球半径), 可判断为碰撞
        depth_min = depth_global_min
        is_collision_depth = depth_min <= self.collision_threshold + 1.0  # 保守阈值

        # 错位度量: 深度最小距离 vs 真正碰撞距离
        # 正值: 深度看到障碍物但碰撞检测认为安全 (深度显示2m但碰撞球距离5m)
        # 负值: 碰撞检测认为有碰撞风险但深度显示空旷
        mismatch = true_dist - depth_min

        # 点是否位于碰撞几何内部
        in_geom = any(s.contains(pos) for s in self.collision_spheres)

        return ValidationSample(
            position=tuple(pos),
            true_collision_dist=true_dist,
            depth_min=depth_min,
            depth_mean=depth_mean,
            depth_at_obstacle_dir=depth_center_min,
            is_collision_true=is_collision_true,
            is_collision_depth_estimated=is_collision_depth,
            mismatch=mismatch,
            in_collision_geometry=in_geom,
        )

    def run_grid_validation(self, grid_size: int = 8) -> ValidationReport:
        """在整个环境空间网格采样验证"""
        positions = self.sample_positions(grid_size)
        samples = []

        for i, pos in enumerate(positions):
            sample = self.validate_point(pos)
            samples.append(sample)
            if (i + 1) % 100 == 0:
                logger.info(f"  验证进度: {i+1}/{len(positions)}")

        return self._aggregate(samples)

    def _aggregate(self, samples: List[ValidationSample]) -> ValidationReport:
        """汇总验证结果"""
        n = len(samples)
        mismatches = [s.mismatch for s in samples]
        abs_mismatches = [abs(s.mismatch) for s in samples]

        # 错位: |真实距离 - 深度估算距离| > 1.0m
        threshold = 1.0
        mismatched = sum(1 for m in mismatches if abs(m) > threshold)

        # 假碰撞: 深度说没有但碰撞说有
        phantom = sum(1 for s in samples
                      if not s.is_collision_depth_estimated and s.is_collision_true)
        # 漏碰撞: 深度说有但碰撞说没有
        missed = sum(1 for s in samples
                     if s.is_collision_depth_estimated and not s.is_collision_true)

        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            n_samples=n,
            mismatch_ratio=mismatched / n * 100,
            mean_mismatch=float(np.mean(abs_mismatches)),
            max_mismatch=float(np.max(abs_mismatches)),
            phantom_collisions=phantom,
            missed_collisions=missed,
            collision_geometry_details={
                "n_spheres": len(self.collision_spheres),
                "spheres": [
                    {"x": s.x, "y": s.y, "z": s.z, "radius": s.radius}
                    for s in self.collision_spheres
                ],
                "drone_radius": self.drone_radius,
                "collision_threshold": self.collision_threshold,
            },
        )

        # Mock渲染器: 额外验证渲染障碍物 vs 碰撞球体的对应关系
        if self.renderer_type == 'mock':
            report.collision_geometry_details["render_obstacles"] = [
                {"x": o[0], "y": o[1], "z": o[2], "radius": o[3]}
                for o in self.mock_obstacles
            ]
            # 计算渲染障碍物到最近碰撞球体的距离
            render_to_collision_dists = []
            for obs in self.mock_obstacles:
                obs_pos = obs[:3]
                min_d = min(np.linalg.norm(obs_pos - s.as_array())
                           for s in self.collision_spheres)
                render_to_collision_dists.append(float(min_d))
            report.collision_geometry_details["render_to_collision_offset"] = \
                render_to_collision_dists
            report.collision_geometry_details["render_collision_aligned"] = \
                all(d < 0.5 for d in render_to_collision_dists)

        # 额外统计: 深度最小值的分布
        depth_mins = [s.depth_min for s in samples if s.depth_min < 19.0]
        report.collision_geometry_details["depth_stats"] = {
            "mean_min_depth": float(np.mean(depth_mins)) if depth_mins else 0,
            "min_depth_5pct": float(np.percentile(depth_mins, 5)) if depth_mins else 0,
            "min_depth_95pct": float(np.percentile(depth_mins, 95)) if depth_mins else 0,
        }

        # 错位分布分位数
        report.collision_geometry_details["mismatch_percentiles"] = {
            "p50": float(np.percentile(abs_mismatches, 50)),
            "p90": float(np.percentile(abs_mismatches, 90)),
            "p95": float(np.percentile(abs_mismatches, 95)),
        }

        return report

    # ── 真实3DGS特别检查 ──

    def check_gsplat_alignment(self) -> dict:
        """
        检查真实3DGS点云与碰撞球体的对齐情况

        方法: 加载3DGS的Gaussian中心点, 统计落在每个碰撞球体内的比例,
             以及球体附近是否有Gaussian支持.
        """
        if self.renderer_type != 'gsplat':
            return {"skipped": "only meaningful for gsplat renderer"}

        logger.info("执行真实3DGS碰撞几何对齐检查...")
        means = np.array(self.renderer._means)

        results = {}
        total_in_any = 0
        for i, sphere in enumerate(self.collision_spheres):
            center = sphere.as_array()
            r = sphere.radius + self.drone_radius  # 有效碰撞半径
            dists = np.linalg.norm(means - center, axis=1)
            in_sphere = (dists <= r).sum()
            near_sphere = ((dists > r) & (dists <= r + 2.0)).sum()
            total_in_any += in_sphere

            # 球体中心附近最近Gaussian的距离
            nearest_gaussian_dist = float(dists.min()) if len(dists) > 0 else -1

            results[f"sphere_{i}"] = {
                "center": [sphere.x, sphere.y, sphere.z],
                "radius": r,
                "gaussians_inside": int(in_sphere),
                "gaussians_nearby": int(near_sphere),
                "nearest_gaussian_dist": nearest_gaussian_dist,
                "has_geometric_support": in_sphere > 0,
            }
            logger.info(f"  球体S{i}({center}): {in_sphere}个Gaussian在内, "
                       f"{near_sphere}个附近, 最近={nearest_gaussian_dist:.2f}m")

        results["total_gaussians"] = len(means)
        results["gaussians_in_collision_volume"] = int(total_in_any)
        results["collision_volume_coverage_pct"] = float(
            total_in_any / len(means) * 100) if len(means) > 0 else 0

        # 总体对齐判定: 如果任一球体没有Gaussian支持, 标记警告
        all_supported = all(
            v["has_geometric_support"] for k, v in results.items()
            if k.startswith("sphere_")
        )
        results["aligned"] = bool(all_supported)
        if not all_supported:
            unsupported = [k for k, v in results.items()
                          if k.startswith("sphere_") and not v["has_geometric_support"]]
            logger.warning(f"⚠️ 碰撞球体在3DGS场景中无几何支持: {unsupported}")

        return results

    # ── 可视化 ──

    def generate_visualization(self, report: ValidationReport, output_dir: str):
        """生成验证可视化"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        os.makedirs(output_dir, exist_ok=True)

        # 从env重新采样 (量大时不需要全存)
        positions = self.sample_positions(grid_size=10)
        samples = [self.validate_point(pos) for pos in positions]

        # 图1: 3D错位图
        fig1 = plt.figure(figsize=(12, 10))
        ax = fig1.add_subplot(111, projection='3d')
        mismatches = np.array([s.mismatch for s in samples])
        pos_arr = np.array([s.position for s in samples])

        # 只显示错位 > 0.5m的点
        significant = np.abs(mismatches) > 0.5
        sc = ax.scatter(pos_arr[significant, 0], pos_arr[significant, 1],
                       pos_arr[significant, 2],
                       c=mismatches[significant], cmap='RdBu_r',
                       vmin=-3, vmax=3, s=20, alpha=0.7)
        plt.colorbar(sc, label='Mismatch (true_dist - depth_min) [m]')

        # 绘制碰撞球体
        for sphere in self.collision_spheres:
            self._draw_sphere(ax, sphere.as_array(), sphere.radius, 'red', 0.1)

        # 绘制Mock渲染障碍物 (如适用)
        if self.renderer_type == 'mock':
            for obs in self.mock_obstacles:
                self._draw_sphere(ax, obs[:3], obs[3], 'blue', 0.08)

        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
        ax.set_title('Collision Geometry Mismatch Map\n'
                    f'(Red/Blue=Collision spheres, Blue circles=Render obstacles)')
        ax.set_xlim([-10, 10]); ax.set_ylim([-10, 10]); ax.set_zlim([0, 10])
        fig1.savefig(os.path.join(output_dir, 'mismatch_3d.png'), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        logger.info(f"  3D错位图 → mismatch_3d.png")

        # 图2: XY平面切片 (在碰撞球体高度z≈3.5)
        fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ['True Collision Distance', 'Depth Min Distance', 'Mismatch']
        z_slice = 3.5  # 障碍物高度

        xy_positions = np.array([
            [s.position[0], s.position[1], s.position[2]]
            for s in samples
        ])
        near_z = np.abs(xy_positions[:, 2] - z_slice) < 1.0
        xy_near = xy_positions[near_z]
        mismatches_near = mismatches[near_z]
        true_dists = np.array([s.true_collision_dist for s in samples])[near_z]
        depth_mins = np.array([s.depth_min for s in samples])[near_z]

        for idx, (ax_i, data, cmap, label) in enumerate(zip(
            axes, [true_dists, depth_mins, mismatches_near],
            ['Greens', 'Blues', 'RdBu_r'],
            ['True Dist to Collision [m]', 'Depth Min [m]', 'Mismatch [m]']
        )):
            sc = ax_i.scatter(xy_near[:, 0], xy_near[:, 1], c=data,
                            cmap=cmap, s=15, alpha=0.7,
                            vmin=(-3 if idx == 2 else None),
                            vmax=(3 if idx == 2 else None))
            plt.colorbar(sc, ax=ax_i, label=label)
            ax_i.set_xlabel('X [m]'); ax_i.set_ylabel('Y [m]')
            ax_i.set_title(titles[idx])
            ax_i.set_xlim([-10, 10]); ax_i.set_ylim([-10, 10])
            ax_i.set_aspect('equal')
            # 标记障碍物位置
            for s in self.collision_spheres:
                circle = plt.Circle((s.x, s.y), s.radius, fill=False,
                                   color='red', linestyle='--', alpha=0.5)
                ax_i.add_patch(circle)
            if self.renderer_type == 'mock':
                for obs in self.mock_obstacles:
                    circle = plt.Circle((obs[0], obs[1]), obs[3], fill=False,
                                       color='blue', linestyle=':', alpha=0.5)
                    ax_i.add_patch(circle)

        fig2.suptitle(f'XY Slice at z≈{z_slice}m (Collision Sphere Height)')
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'mismatch_xy_slice.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        logger.info(f"  XY切片 → mismatch_xy_slice.png")

        # 图3: 错位直方图
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.hist(mismatches, bins=50, alpha=0.7, color='steelblue',
                edgecolor='white')
        ax3.axvline(x=1.0, color='red', linestyle='--', label='Warning threshold')
        ax3.axvline(x=-1.0, color='red', linestyle='--')
        ax3.set_xlabel('Mismatch (true_dist - depth_min) [m]')
        ax3.set_ylabel('Count')
        ax3.set_title('Mismatch Distribution\n'
                     '(positive = depth sees obstacle closer than collision geometry)')
        ax3.legend()
        fig3.tight_layout()
        fig3.savefig(os.path.join(output_dir, 'mismatch_histogram.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        logger.info(f"  错位直方图 → mismatch_histogram.png")

        # 图4: 真实GS渲染器的点云+球体叠加 (如果有3DGS)
        if self.renderer_type == 'gsplat':
            self._visualize_gsplat_alignment(output_dir)

    def _draw_sphere(self, ax, center, radius, color='red', alpha=0.1):
        """在3D图中绘制球体线框 (修复inhomogeneous shape问题)"""
        u = np.linspace(0, 2*np.pi, 16)
        v = np.linspace(0, np.pi, 12)
        u, v = np.meshgrid(u, v)
        x = center[0] + radius * np.sin(v) * np.cos(u)
        y = center[1] + radius * np.sin(v) * np.sin(u)
        z = center[2] + radius * np.cos(v)
        for i in range(len(u)):
            ax.plot(x[i], y[i], z[i], color=color, alpha=alpha, linewidth=0.5)
        for i in range(len(v)):
            ax.plot(x[:, i], y[:, i], z[:, i], color=color, alpha=alpha, linewidth=0.5)

    def _visualize_gsplat_alignment(self, output_dir: str):
        """可视化3DGS点云与碰撞球体叠加"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        means = np.array(self.renderer._means)

        # 随机采样Gaussian中心 (太多点会卡)
        max_points = 20000
        if len(means) > max_points:
            idx = np.random.choice(len(means), max_points, replace=False)
            means = means[idx]

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制Gaussian点云 (根据高度着色)
        sc = ax.scatter(means[:, 0], means[:, 1], means[:, 2],
                       c=means[:, 2], cmap='viridis', s=1, alpha=0.5)
        plt.colorbar(sc, label='Z [m]', ax=ax)

        # 绘制碰撞球体
        for sphere in self.collision_spheres:
            self._draw_sphere(ax, sphere.as_array(), sphere.radius + self.drone_radius,
                            'red', 0.3)
            # 球体中心标记
            ax.scatter(*sphere.as_array(), color='red', s=50, marker='x')

        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
        ax.set_title('3DGS Point Cloud + Collision Spheres\n'
                    '(Red=Collision geometry, points=3DGS Gaussians)')
        ax.set_xlim([-10, 10]); ax.set_ylim([-10, 10]); ax.set_zlim([0, 10])
        fig.savefig(os.path.join(output_dir, 'gsplat_alignment_3d.png'),
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  3DGS对齐图 → gsplat_alignment_3d.png")


# ============================================================
# 报告
# ============================================================

def print_report(report: ValidationReport):
    """打印验证报告（GBK-safe版，不用Unicode字符）"""
    print("\n" + "=" * 60)
    print("  碰撞几何验证报告")
    print("=" * 60)
    print(f"  采样点:     {report.n_samples}")
    print(f"  错位比例:   {report.mismatch_ratio:.1f}%")
    print(f"  平均错位:   {report.mean_mismatch:.2f}m")
    print(f"  最大错位:   {report.max_mismatch:.2f}m")
    print(f"  假碰撞:     {report.phantom_collisions}")
    print(f"  漏碰撞:     {report.missed_collisions}")
    print()

    geom = report.collision_geometry_details
    if "render_to_collision_offset" in geom:
        print(f"  渲染-碰撞偏移: {geom['render_to_collision_offset']}")
        ok = geom.get('render_collision_aligned')
        print(f"  几何对齐:     {'OK' if ok else 'NOT ALIGNED!'}")

    if "depth_stats" in geom:
        ds = geom["depth_stats"]
        print(f"  深度最小值均值: {ds['mean_min_depth']:.2f}m")
        print(f"  深度最小值5%%: {ds['min_depth_5pct']:.2f}m  | 95%%: {ds['min_depth_95pct']:.2f}m")

    if "mismatch_percentiles" in geom:
        mp = geom["mismatch_percentiles"]
        print(f"  错位中位数:  {mp['p50']:.2f}m")
        print(f"  错位P90:     {mp['p90']:.2f}m")
        print(f"  错位P95:     {mp['p95']:.2f}m")

    print()
    print("  判定标准:")
    print("     错位=碰撞球面距离 - 深度最小值")
    print("     >0: 深度看到的障碍物比碰撞几何近 (Agent视觉上更危险)")
    print("     <0: 碰撞几何比深度看到障碍物更近 (碰撞检测过于保守)")
    print("     |错位|>1.0m 视为显著错位")
    print()

    # 警告判定
    warnings = []
    if report.mismatch_ratio > 10:
        warnings.append(f"[WARN] 错位比例 {report.mismatch_ratio:.1f}% > 10% - 几何不一致")
    if report.phantom_collisions > report.n_samples * 0.05:
        warnings.append(f"[WARN] 假碰撞 {report.phantom_collisions} 次 (>{report.n_samples*0.05:.0f})")
    if report.missed_collisions > report.n_samples * 0.05:
        warnings.append(f"[WARN] 漏碰撞 {report.missed_collisions} 次 (>{report.n_samples*0.05:.0f})")

    geom_detail = report.collision_geometry_details
    if "render_collision_aligned" in geom_detail and not geom_detail['render_collision_aligned']:
        warnings.append("[CRITICAL] Mock渲染器障碍物与碰撞球体不一致!")

    if report.gsplat_check is not None:
        if report.gsplat_check.get("aligned") is False:
            warnings.append("[CRITICAL] 3DGS场景中碰撞球体无几何支持!")

    if warnings:
        for w in warnings:
            print(f"  {w}")
    else:
        print("  [OK] 碰撞几何验证通过")
    print("=" * 60)


def save_report(report: ValidationReport, output_dir: str):
    """保存报告为JSON (支持numpy float32)"""
    import json as _json

    class _NumpyEncoder(_json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "collision_geometry_report.json")
    with open(path, 'w') as f:
        _json.dump({
            "timestamp": report.timestamp,
            "n_samples": report.n_samples,
            "mismatch_ratio_pct": report.mismatch_ratio,
            "mean_mismatch_m": report.mean_mismatch,
            "max_mismatch_m": report.max_mismatch,
            "phantom_collisions": report.phantom_collisions,
            "missed_collisions": report.missed_collisions,
            "collision_geometry_details": report.collision_geometry_details,
            "gsplat_alignment": report.gsplat_check,
        }, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    logger.info(f"报告已保存: {path}")
    return path


# ============================================================
# CLI入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="碰撞几何验证工具 — 确保实验结论有效性")
    parser.add_argument('--renderer', choices=['mock', 'gsplat'], default='mock',
                       help='渲染器类型 (默认mock)')
    parser.add_argument('--ply', type=str, default=None,
                       help='真实3DGS PLY路径 (--renderer gsplat时必需)')
    parser.add_argument('--grid', type=int, default=10,
                       help='网格采样密度 (默认10, 总点数≈10³=1000)')
    parser.add_argument('--output', type=str, default='reports/collision_validation',
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化图表')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式 (grid=6, 无可视化)')
    args = parser.parse_args()

    if args.quick:
        args.grid = 6
        args.visualize = False

    if args.renderer == 'gsplat' and (not args.ply or not os.path.isfile(args.ply)):
        parser.error("--renderer gsplat requires --ply <path>")

    # 配置
    config = {'renderer': args.renderer}
    if args.ply:
        ply_path = os.path.abspath(args.ply)
        config['ply_path'] = ply_path

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # 初始化验证器
    validator = CollisionGeometryValidator(config)
    logger.info(f"输出目录: {output_dir}")

    try:
        # 1. 网格验证
        report = validator.run_grid_validation(grid_size=args.grid)

        # 2. 真实GS对齐检查
        if args.renderer == 'gsplat':
            report.gsplat_check = validator.check_gsplat_alignment()

        # 3. 打印并保存报告
        print_report(report)
        report_path = save_report(report, output_dir)

        # 4. 可视化
        if args.visualize:
            logger.info("生成可视化...")
            validator.generate_visualization(report, output_dir)

        # 5. 保存配置
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump({
                "renderer": args.renderer,
                "ply": args.ply,
                "grid_size": args.grid,
            }, f, indent=2)

        logger.info(f"验证完成 → {output_dir}")
        return 0 if report.mismatch_ratio < 10 else 1

    finally:
        validator.close()


if __name__ == "__main__":
    sys.exit(main())
