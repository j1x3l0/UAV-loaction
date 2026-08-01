"""
VisualDroneEnv — 3DGS 渲染 + 深度图观测的无人机导航环境 v2

架构位置: envs/ (Extension层)
WHY 这个设计:
  - Mock renderer 让管线在无3DGS时也能开发调试
  - 可插拔接口: 后面换真实3DGS渲染器只需实现相同接口
  - 复用v1的质点动力学+碰撞+7组件reward
数据流: drone_pos → renderer.render(pose) → depth(64×64) + vec(6) → CNN → action
边界: 不负责网络训练、不负责3DGS训练、不负责真机部署
风险: Mock渲染器和真3DGS的深度图分布可能不同 → 真3DGS接入后需要重新tune超参
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Dict, Any, Tuple, Optional
import logging

from envs.degradation_utils import (
    apply_gaussian_sparsification,
    apply_degradation_pipeline,
)
from envs.gs_renderer import GSplatRenderer
from envs.scene_geometry import ScenePointCloudGeometry

logger = logging.getLogger(__name__)


# ─── Mock 3DGS 渲染器 ───────────────────────────────────────────
# 位置: VisualDroneEnv 的内部组件
# WHY: 在无3DGS/Nerfstudio环境时提供可训练的合成深度图
# 边界: 只模拟深度图。不模拟RGB、不保证物理精度
# 风险: 和真3DGS深度分布不同 → 真3DGS接入后需重新验证

class MockGSRenderer:
    """合成深度图渲染器 — 用简单几何体模拟3DGS渲染输出"""

    def __init__(self, width=64, height=64, obstacles=None):
        self.width = width
        self.height = height
        self.fov = 90.0
        self.max_depth = 20.0
        if obstacles is not None:
            self.obstacles = obstacles
        else:
            self.obstacles = np.array([
                [5.0, 0.0, 3.0, 1.5],
                [8.0, 2.0, 4.0, 1.2],
                [3.0, -2.0, 2.5, 1.0],
            ])

    def render(self, camera_pos: np.ndarray,
               camera_quat: np.ndarray = None,
               obstacles: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从相机位姿渲染合成深度图

        Args:
            camera_pos: (3,) 相机世界坐标
            camera_quat: (4,) 四元数 [x,y,z,w] (暂用简化朝向)
            obstacles:  (N,4) 障碍物数组 [x,y,z,radius],
                        为 None 时使用 self.obstacles (默认场景)
        Returns:
            depth: (H, W, 1) 深度图, 范围 [0, max_depth]
            rgb: (H, W, 3) 占位RGB (全零)
        """
        if obstacles is None:
            obstacles = self.obstacles

        H, W = self.height, self.width
        depth = np.full((H, W, 1), self.max_depth, dtype=np.float32)

        # 简单透视投影: 每个像素对应一条射线
        ys, xs = np.mgrid[0:H, 0:W]
        # 像素坐标 → 相机坐标系方向
        fx = fy = (W / 2) / np.tan(np.deg2rad(self.fov / 2))
        cx, cy = W / 2, H / 2
        ray_x = (xs - cx) / fx
        ray_y = (ys - cy) / fy
        ray_z = np.ones_like(ray_x)  # 前方

        # 归一化射线方向 (相机坐标系)
        ray_norm = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
        ray_x /= ray_norm; ray_y /= ray_norm; ray_z /= ray_norm

        # 简化: 假设相机朝向 +X (世界坐标)
        # 射线在世界空间: 起点 = camera_pos, 方向 ≈ (ray_z, ray_x, ray_y)
        # (这是一个非常简化的假设, 真3DGS会用四元数做正确旋转)
        ray_dir_world = np.stack([
            ray_z,   # 世界X (前方)
            ray_x,   # 世界Y
            -ray_y   # 世界Z (上)
        ], axis=-1)  # (H, W, 3)

        # 对每个障碍物做射线-球体求交
        for obs in obstacles:
            ox, oy, oz, orad = obs
            oc = np.array([ox, oy, oz])
            # 射线-球体求交: |(cam - oc) + t*dir|² = r²
            L = camera_pos - oc
            a = np.sum(ray_dir_world * ray_dir_world, axis=-1)  # = 1
            b = 2 * np.sum(ray_dir_world * L, axis=-1)
            c_val = np.sum(L * L) - orad * orad
            disc = b * b - 4 * c_val
            hit_mask = disc > 0
            t = (-b - np.sqrt(np.maximum(disc, 0))) / 2
            t = np.where((hit_mask) & (t > 0), t, self.max_depth)
            depth = np.minimum(depth, t[..., np.newaxis])

        # 添加轻微纹理 (模拟场景几何)
        noise = 0.02 * (np.sin(xs * 0.3) * np.cos(ys * 0.3))
        depth = depth + noise[..., np.newaxis]
        depth = np.clip(depth, 0.1, self.max_depth)

        rgb = np.zeros((H, W, 3), dtype=np.float32)
        return depth, rgb


# ─── 视觉无人机环境 ─────────────────────────────────────────────

class VisualDroneEnv(gym.Env):
    """
    3DGS视觉导航环境 v2

    观测: Dict {depth: (64,64,1), vec: (6,)}
          vec = [vx, vy, vz, target_dx, target_dy, target_dz]
    动作: thrust_xyz ∈ [-1, 1]³
    退化: 通过 degradation_config 控制5条退化轴
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}

        # ── 空间参数 ──
        self.boundary_min = np.array([-10.0, -10.0, 0.0])
        self.boundary_max = np.array([10.0, 10.0, 10.0])
        self.target_min = np.array([5.0, 5.0, 2.0])
        self.target_max = np.array([8.0, 8.0, 8.0])

        # ── 物理参数 (复用v1) ──
        self.drone_mass = 1.0
        self.max_thrust = 10.0
        self.max_velocity = 5.0
        self.dt = 0.05

        # Collision geometry. Mock rendering must use these exact spheres;
        # otherwise the policy sees obstacles at different positions from
        # those used by collision detection.
        self.obstacles = np.array([
            [2.0, 2.0, 3.0],
            [6.0, 3.0, 5.0],
            [3.0, 7.0, 4.0],
        ])
        self.obstacle_radius = 1.0
        self.collision_threshold = 0.5
        self.scene_geometry = None
        collision_render_obstacles = np.column_stack([
            self.obstacles,
            np.full(len(self.obstacles), self.obstacle_radius),
        ]).astype(np.float32)

        # ── 渲染器 ──
        renderer_type = self.config.get('renderer', 'mock')
        if renderer_type == 'mock':
            self.renderer = MockGSRenderer(
                width=64, height=64, obstacles=collision_render_obstacles)
            self._base_obstacles_for_render = collision_render_obstacles.copy()
        elif renderer_type == 'gsplat':
            ply_path = self.config.get('ply_path')
            if ply_path is None:
                raise ValueError("gsplat renderer requires 'ply_path' in config")
            intrinsics = self.config.get('camera_intrinsics')
            if intrinsics is not None:
                # Explicit policy camera model (fx/fy/cx/cy), matching the
                # PX4 alignment config and the read-only observation bridge.
                # Without it the renderer falls back to fov=90 (fx=32).
                try:
                    self.renderer = GSplatRenderer(
                        ply_path, width=64, height=64,
                        fx=float(intrinsics['fx']),
                        fy=float(intrinsics['fy']),
                        cx=float(intrinsics['cx']),
                        cy=float(intrinsics['cy']),
                    )
                except KeyError as exc:
                    raise ValueError(
                        "camera_intrinsics must define fx/fy/cx/cy") from exc
            else:
                self.renderer = GSplatRenderer(ply_path, width=64, height=64)
            gaussian_level = self.config.get('degradation', {}).get('gaussian')
            if gaussian_level is not None:
                self.renderer.set_gaussian_keep_percent(gaussian_level)
            self._base_obstacles_for_render = np.empty((0, 4))
        else:
            raise ValueError(f"Unsupported renderer: {renderer_type}")

        # ── PX4 对齐相机模型 (统一训练与观测桥的相机路径) ──
        # 配置了 alignment_config 时，真实 GS 渲染改用与只读观测桥相同的
        # Px4SceneAlignment.camera_c2w 路径（从 env 状态模拟 PX4 等效位姿），
        # 使训练观测的相机朝向/内参与部署一致。不带则不启用（保持旧行为）。
        self._alignment = None
        alignment_config = self.config.get('alignment_config')
        if alignment_config:
            from integrations.px4_scene_alignment import Px4SceneAlignment
            self._alignment = Px4SceneAlignment.from_json(alignment_config)

        collision_ply_path = self.config.get('collision_ply_path')
        if collision_ply_path:
            self.scene_geometry = ScenePointCloudGeometry(
                collision_ply_path,
                bounds_percentiles=self.config.get(
                    'scene_bounds_percentiles', (1, 99)),
                boundary_margin=self.config.get(
                    'scene_boundary_margin', (0.5, 0.5, 0.35)),
            )
            if self.config.get('auto_scene_bounds', True):
                self.boundary_min = self.scene_geometry.boundary_min.copy()
                self.boundary_max = self.scene_geometry.boundary_max.copy()
            self.collision_threshold = float(
                self.config.get('drone_collision_radius', 0.25))
            self.scene_geometry.build_navigation_grid(
                resolution=float(
                    self.config.get('navigation_grid_resolution', 0.3)),
                clearance=float(
                    self.config.get('spawn_clearance',
                                    self.collision_threshold + 0.2)),
            )

        # ── 退化配置 ──
        self.deg_config = self.config.get('degradation', {})
        self.randomize_depth_scale = self.config.get(
            'randomize_depth_scale', False)
        self.depth_scale_levels = self.config.get(
            'depth_scale_levels', [1.0, 0.75, 0.5, 0.25, 0.1])
        self.depth_scale_probabilities = None
        self.set_depth_scale_probabilities(
            self.config.get('depth_scale_probabilities'))
        self.depth_scale_sample_counts = np.zeros(
            len(self.depth_scale_levels), dtype=np.int64)
        self.ablation_config = self.config.get('ablation', {})
        self.avoidance_episode_probability = self.config.get(
            'avoidance_episode_probability', 0.5)
        self.avoidance_sample_counts = np.zeros(2, dtype=np.int64)
        self.use_geodesic_reward = bool(
            self.config.get('use_geodesic_reward', False))
        self.use_waypoint_observation = bool(
            self.config.get('use_waypoint_observation', False))
        self.geodesic_progress_scale = float(
            self.config.get('geodesic_progress_scale', 10.0))
        self.geodesic_heading_weight = float(
            self.config.get('geodesic_heading_weight', 2.0))
        self.geodesic_waypoint_lookahead = float(
            self.config.get('geodesic_waypoint_lookahead', 0.9))
        if (
            self.use_geodesic_reward or self.use_waypoint_observation
        ) and self.scene_geometry is None:
            raise ValueError(
                "geodesic reward/waypoint observation requires "
                "collision_ply_path")

        # ── 观测/动作空间 ──
        self.observation_space = spaces.Dict({
            'depth': spaces.Box(0.0, 20.0, (64, 64, 1), dtype=np.float32),
            'vec': spaces.Box(-10.0, 10.0, (6,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

        self.np_random, _ = seeding.np_random(None)

        # 状态变量
        self.target_pos = None
        self.state = None       # [x, y, z, vx, vy, vz]
        self.step_count = 0
        self.max_steps = 500
        self.target_threshold = 0.5
        self._prev_action = None
        self._geodesic_distance_field = None
        self._prev_geodesic_distance = None
        self._geodesic_path = None
        self._geodesic_path_index = 0

    def set_depth_scale_probabilities(self, probabilities):
        """Validate and update per-episode scale sampling probabilities."""
        if probabilities is None:
            self.depth_scale_probabilities = None
            return
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (len(self.depth_scale_levels),):
            raise ValueError(
                "depth_scale_probabilities must match depth_scale_levels")
        if np.any(probabilities < 0) or not np.isclose(
                probabilities.sum(), 1.0):
            raise ValueError(
                "depth_scale_probabilities must be non-negative and sum to 1")
        self.depth_scale_probabilities = probabilities.copy()

    def reset_depth_scale_sample_counts(self):
        self.depth_scale_sample_counts.fill(0)

    def set_avoidance_episode_probability(self, probability):
        """Update the probability of sampling a directly blocked task."""
        if probability is not None and not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                "avoidance episode probability must be in [0, 1] or None")
        self.avoidance_episode_probability = (
            None if probability is None else float(probability))

    def reset_avoidance_sample_counts(self):
        self.avoidance_sample_counts.fill(0)

    # ── 观测构建 ──
    def _get_observation(self) -> Dict[str, np.ndarray]:
        pos = self.state[:3]
        vel = self.state[3:6]

        camera_pos = pos.copy()
        is_mock = isinstance(self.renderer, MockGSRenderer)

        if is_mock:
            # Mock: 高斯稀疏化修改障碍物 → 渲染 → 退化
            # P2-1 修复: 传 obstacles 参数而非修改 renderer 状态 (避免竞态)
            if 'gaussian' in self.deg_config:
                keep_ratio = self.deg_config['gaussian'] / 100.0
                active_obstacles = apply_gaussian_sparsification(
                    self._base_obstacles_for_render, keep_ratio)
            else:
                active_obstacles = self._base_obstacles_for_render

            depth, rgb = self.renderer.render(camera_pos,
                                              obstacles=active_obstacles)

            post_config = {k: v for k, v in self.deg_config.items()
                           if k != 'gaussian'}
            depth, rgb, _ = apply_degradation_pipeline(
                depth, rgb, active_obstacles, post_config)
        else:
            # Real GS: 直接渲染 → 后处理退化
            if self._alignment is not None:
                # 统一相机模型：模拟 PX4 等效位姿 → camera_c2w（与观测桥一致）
                c2w = self._aligned_camera_c2w(pos, vel)
                depth, rgb = self.renderer.render(camera_pos, camera_c2w=c2w)
            else:
                camera_quat = self._camera_quaternion(pos, vel)
                depth, rgb = self.renderer.render(camera_pos, camera_quat)
            post_config = {k: v for k, v in self.deg_config.items()
                           if k != 'gaussian'}
            depth, rgb, _ = apply_degradation_pipeline(
                depth, rgb, np.empty((0, 4)), post_config)

        if self.ablation_config.get('const_depth', False):
            depth = np.full_like(depth, 5.0)

        # 向量状态
        target_dir = self.target_pos - pos
        if self.use_waypoint_observation:
            target_dir = self._geodesic_waypoint_direction(pos)
        if self.ablation_config.get('no_velocity', False):
            vel = np.zeros(3, dtype=np.float32)
        if self.ablation_config.get('no_target_dir', False):
            target_dir = np.zeros(3, dtype=np.float32)
        vec = np.array([
            vel[0], vel[1], vel[2],
            target_dir[0], target_dir[1], target_dir[2]
        ], dtype=np.float32)

        return {'depth': depth.astype(np.float32), 'vec': vec}

    # ── 物理仿真 (复用v1) ──
    def _get_min_obstacle_distance(self, pos: np.ndarray) -> float:
        if self.scene_geometry is not None:
            return max(
                self.scene_geometry.nearest_distance(pos)
                - self.collision_threshold,
                0.0,
            )
        min_dist = float('inf')
        for obs_pos in self.obstacles:
            dist = np.linalg.norm(pos - obs_pos) - self.obstacle_radius
            min_dist = min(min_dist, dist)
        return max(min_dist, 0.0)

    @staticmethod
    def _rotation_matrix_to_quaternion(rotation):
        """Convert a 3x3 camera-to-world rotation to [x, y, z, w]."""
        matrix = np.asarray(rotation, dtype=np.float64)
        trace = np.trace(matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (matrix[2, 1] - matrix[1, 2]) / s
            qy = (matrix[0, 2] - matrix[2, 0]) / s
            qz = (matrix[1, 0] - matrix[0, 1]) / s
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1]
                            - matrix[2, 2]) * 2.0
                qw = (matrix[2, 1] - matrix[1, 2]) / s
                qx, qy, qz = 0.25 * s, (
                    matrix[0, 1] + matrix[1, 0]) / s, (
                    matrix[0, 2] + matrix[2, 0]) / s
            elif axis == 1:
                s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0]
                            - matrix[2, 2]) * 2.0
                qw = (matrix[0, 2] - matrix[2, 0]) / s
                qx, qy, qz = (
                    matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (
                    matrix[1, 2] + matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0]
                            - matrix[1, 1]) * 2.0
                qw = (matrix[1, 0] - matrix[0, 1]) / s
                qx, qy, qz = (
                    matrix[0, 2] + matrix[2, 0]) / s, (
                    matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s
        quaternion = np.array([qx, qy, qz, qw], dtype=np.float32)
        return quaternion / (np.linalg.norm(quaternion) + 1e-8)

    def _camera_quaternion(self, pos, velocity):
        """Point the optical +Z axis along motion, falling back to the goal."""
        if not self.config.get('camera_tracks_motion', False):
            return None
        forward = np.asarray(velocity, dtype=np.float64)
        if np.linalg.norm(forward) < 0.1:
            forward = np.asarray(self.target_pos - pos, dtype=np.float64)
        forward /= np.linalg.norm(forward) + 1e-8
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.95:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right) + 1e-8
        down = np.cross(forward, right)
        rotation = np.column_stack([right, down, forward])
        return self._rotation_matrix_to_quaternion(rotation)

    def _aligned_camera_c2w(self, pos, velocity):
        """Simulated PX4-equivalent camera, matching the read-only bridge.

        Maps the scene-space drone state to a PX4 LOCAL_NED pose (yaw follows
        the horizontal velocity, falling back to the goal heading at low
        speed; roll/pitch stay 0 for the point-mass model) and builds the
        OpenCV optical camera-to-scene matrix exactly like the observation
        bridge's ``Px4SceneAlignment.camera_c2w``.
        """
        alignment = self._alignment
        position = np.asarray(pos, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        position_ned = (
            alignment.scene_from_ned_rotation.T
            @ (position - alignment.scene_from_ned_translation)
        ) / alignment.scale
        velocity_ned = alignment.vector_ned_from_scene(velocity)
        heading = velocity_ned[:2]
        if np.linalg.norm(heading) < 0.1:
            target_dir_ned = alignment.vector_ned_from_scene(
                np.asarray(self.target_pos, dtype=np.float64) - position)
            heading = target_dir_ned[:2]
        yaw = float(np.arctan2(heading[1], heading[0]))
        return alignment.camera_c2w(position_ned, 0.0, 0.0, yaw)

    # ── Gym API ──
    def reset(self, seed: int = None,
              options: Dict[str, Any] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        if self.randomize_depth_scale:
            # Sample once per episode so the policy cannot memorize a fixed
            # calibration while each trajectory remains internally coherent.
            sampled_index = int(self.np_random.choice(
                len(self.depth_scale_levels),
                p=self.depth_scale_probabilities))
            self.depth_scale_sample_counts[sampled_index] += 1
            self.deg_config = {
                **self.deg_config,
                'depth_scale': float(
                    self.depth_scale_levels[sampled_index]),
            }

        if self.scene_geometry is not None:
            pos, self.target_pos, requires_avoidance = \
                self.scene_geometry.sample_reachable_pair(
                    self.np_random,
                    min_distance=float(
                        self.config.get('min_goal_distance', 3.0)),
                    blocked_probability=self.avoidance_episode_probability,
                    collision_radius=self.collision_threshold,
                )
            if requires_avoidance is not None:
                self.avoidance_sample_counts[int(requires_avoidance)] += 1
        else:
            start_min = self.boundary_min + 1.0
            start_max = np.array([2.0, 2.0, 2.0])
            pos = self.np_random.uniform(start_min, start_max)
            self.target_pos = self.np_random.uniform(
                self.target_min, self.target_max)
            requires_avoidance = None
        vel = np.zeros(3)

        self.state = np.concatenate([pos, vel]).astype(np.float32)
        self.step_count = 0
        self._prev_action = None
        if self.use_geodesic_reward or self.use_waypoint_observation:
            self._geodesic_distance_field = \
                self.scene_geometry.geodesic_distance_field(self.target_pos)
            self._prev_geodesic_distance = \
                self.scene_geometry.geodesic_distance(
                    pos, self._geodesic_distance_field)
            self._geodesic_path = self.scene_geometry.shortest_path(
                pos, self.target_pos)
            self._geodesic_path_index = 0
        else:
            self._geodesic_distance_field = None
            self._prev_geodesic_distance = None
            self._geodesic_path = None
            self._geodesic_path_index = 0

        return self._get_observation(), {
            'target_pos': self.target_pos,
            'depth_scale': self.deg_config.get('depth_scale', 1.0),
            'requires_avoidance': requires_avoidance,
        }

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        thrust = np.clip(action, -1.0, 1.0) * self.max_thrust
        pos = self.state[:3]
        vel = self.state[3:6]

        # 质点动力学
        accel = thrust / self.drone_mass
        new_vel = np.clip(vel + accel * self.dt, -self.max_velocity, self.max_velocity)
        new_pos = pos + vel * self.dt + 0.5 * accel * self.dt**2

        # 边界处理
        boundary_hit = False
        for i in range(3):
            if new_pos[i] <= self.boundary_min[i]:
                new_pos[i] = self.boundary_min[i]; new_vel[i] *= -0.5
                boundary_hit = True
            elif new_pos[i] >= self.boundary_max[i]:
                new_pos[i] = self.boundary_max[i]; new_vel[i] *= -0.5
                boundary_hit = True

        # 碰撞检测
        if self.scene_geometry is not None:
            collision = self.scene_geometry.collides(
                new_pos, self.collision_threshold)
        else:
            collision = False
            for obs_pos in self.obstacles:
                if np.linalg.norm(new_pos - obs_pos) <= \
                        self.collision_threshold + self.obstacle_radius:
                    collision = True
                    break

        target_dist = np.linalg.norm(new_pos - self.target_pos)
        reached = target_dist <= self.target_threshold

        self.state = np.concatenate([new_pos, new_vel]).astype(np.float32)
        self.step_count += 1

        reward, comps = self._compute_reward(new_pos, new_vel, action,
                                              collision, reached, target_dist)
        terminated = reached or collision
        truncated = self.step_count >= self.max_steps
        self._prev_action = action.copy()

        info = {'target_pos': self.target_pos, 'collision': collision,
                'reached_target': reached, 'target_distance': target_dist,
                'boundary_hit': boundary_hit, 'step_count': self.step_count,
                'reward_components': comps}

        return self._get_observation(), reward, terminated, truncated, info

    # ── 奖励: 复用v1的7组件 ──
    def _geodesic_waypoint_direction(self, pos):
        """Return a dense safe-heading signal from the reset-time path."""
        remaining = self._geodesic_path[self._geodesic_path_index:]
        nearest_offset = int(np.argmin(
            np.linalg.norm(remaining - pos[None, :], axis=1)))
        self._geodesic_path_index += nearest_offset
        waypoint_index = self._geodesic_path_index
        accumulated = 0.0
        while waypoint_index < len(self._geodesic_path) - 1:
            accumulated += float(np.linalg.norm(
                self._geodesic_path[waypoint_index + 1]
                - self._geodesic_path[waypoint_index]))
            waypoint_index += 1
            if accumulated >= self.geodesic_waypoint_lookahead:
                break
        direction = self._geodesic_path[waypoint_index] - pos
        return direction / (np.linalg.norm(direction) + 1e-8)

    def _compute_reward(self, pos, vel, action, collision, reached, target_dist):
        reward = 0.0

        speed = np.linalg.norm(vel)
        tgt_dir = (self.target_pos - pos)
        tgt_norm = tgt_dir / (np.linalg.norm(tgt_dir) + 1e-8)
        vel_norm = vel / (speed + 1e-8) if speed > 0 else np.zeros(3)
        if self.use_geodesic_reward:
            geodesic_distance = self.scene_geometry.geodesic_distance(
                pos, self._geodesic_distance_field)
            progress = self._prev_geodesic_distance - geodesic_distance
            r_dist = self.geodesic_progress_scale * progress
            self._prev_geodesic_distance = geodesic_distance
            waypoint_direction = self._geodesic_waypoint_direction(pos)
            r_heading = (
                speed
                * np.clip(
                    np.dot(vel_norm, waypoint_direction), -1, 1)
                * self.geodesic_heading_weight
            )
        else:
            r_dist = -5.0 * (1 - np.exp(-0.3 * target_dist))
            r_heading = (
                speed
                * np.clip(np.dot(vel_norm, tgt_norm), -1, 1)
                * 2.0
            )
        reward += r_dist
        reward += r_heading

        min_obs = self._get_min_obstacle_distance(pos)
        r_obs = -2.0 / (min_obs + 0.5)
        reward += r_obs

        r_smooth = (-0.5 * np.linalg.norm(action - self._prev_action)
                    if self._prev_action is not None else 0.0)
        reward += r_smooth

        r_goal = (100.0 + 50.0 * (self.max_steps - self.step_count) / self.max_steps
                  if reached else 0.0)
        reward += r_goal
        r_collision = -50.0 if collision else 0.0
        reward += r_collision
        r_timeout = (-10.0 if self.step_count >= self.max_steps
                     and target_dist >= self.target_threshold else 0.0)
        reward += r_timeout

        comps = {'r_dist': r_dist, 'r_heading': r_heading, 'r_obs': r_obs,
                 'r_smooth': r_smooth, 'r_goal': r_goal,
                 'r_collision': r_collision, 'r_timeout': r_timeout}
        return np.clip(reward, -100, 200), comps

    def render(self, mode='human'):
        pass

    def close(self):
        pass


# ── 自检 ──
if __name__ == "__main__":
    env = VisualDroneEnv()
    obs, info = env.reset()
    print("depth shape:", obs['depth'].shape, "| range: [{:.2f}, {:.2f}]".format(
        obs['depth'].min(), obs['depth'].max()))
    print("vec:", obs['vec'])

    action = np.array([0.5, 0.5, 0.5])
    obs, reward, term, trunc, info = env.step(action)
    print("reward: {:.2f} | term: {} | trunc: {}".format(reward, term, trunc))
    print("depth range after step: [{:.2f}, {:.2f}]".format(
        obs['depth'].min(), obs['depth'].max()))
    print("\nVisualDroneEnv 自检通过")
