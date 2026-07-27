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

        # ── 障碍物 (与v1一致, 必须在渲染器初始化之前定义) ──
        self.obstacles = np.array([
            [2.0, 2.0, 3.0],
            [6.0, 3.0, 5.0],
            [3.0, 7.0, 4.0]
        ])
        self.obstacle_radius = 1.0
        self.collision_threshold = 0.5

        # ── 渲染器 ──
        renderer_type = self.config.get('renderer', 'mock')

        # 构建用于渲染的障碍物数组 [x, y, z, radius] — 与碰撞几何一致
        # FIX 2026-07-27: 之前取了 renderer.obstacles (不同的位置) 导致深度图与碰撞检测不匹配
        # 现在从 self.obstacles + self.obstacle_radius 构建, 确保视觉=碰撞几何
        self._base_obstacles_for_render = np.column_stack([
            self.obstacles,
            np.full(len(self.obstacles), self.obstacle_radius)
        ]).astype(np.float32)

        if renderer_type == 'mock':
            self.renderer = MockGSRenderer(width=64, height=64)
        elif renderer_type == 'gsplat':
            ply_path = self.config.get('ply_path')
            if ply_path is None:
                raise ValueError("gsplat renderer requires 'ply_path' in config")
            self.renderer = GSplatRenderer(ply_path, width=64, height=64)
            gaussian_level = self.config.get('degradation', {}).get('gaussian')
            if gaussian_level is not None:
                self.renderer.set_gaussian_keep_percent(gaussian_level)
        else:
            raise ValueError(f"Unsupported renderer: {renderer_type}")

        # ── 退化配置 ──
        self.deg_config = self.config.get('degradation', {})
        self.ablation_config = self.config.get('ablation', {})

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
            depth, rgb = self.renderer.render(camera_pos)
            post_config = {k: v for k, v in self.deg_config.items()
                           if k != 'gaussian'}
            depth, rgb, _ = apply_degradation_pipeline(
                depth, rgb, np.empty((0, 4)), post_config)

        if self.ablation_config.get('const_depth', False):
            depth = np.full_like(depth, 5.0)

        # 向量状态
        target_dir = self.target_pos - pos
        if self.ablation_config.get('no_target_dir', False):
            target_dir = np.zeros(3, dtype=np.float32)
        vec = np.array([
            vel[0], vel[1], vel[2],
            target_dir[0], target_dir[1], target_dir[2]
        ], dtype=np.float32)

        return {'depth': depth.astype(np.float32), 'vec': vec}

    # ── 物理仿真 (复用v1) ──
    def _get_min_obstacle_distance(self, pos: np.ndarray) -> float:
        min_dist = float('inf')
        for obs_pos in self.obstacles:
            dist = np.linalg.norm(pos - obs_pos) - self.obstacle_radius
            min_dist = min(min_dist, dist)
        return max(min_dist, 0.0)

    # ── Gym API ──
    def reset(self, seed: int = None,
              options: Dict[str, Any] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.np_random, _ = seeding.np_random(seed)

        start_min = self.boundary_min + 1.0
        start_max = np.array([2.0, 2.0, 2.0])
        pos = self.np_random.uniform(start_min, start_max)
        vel = np.zeros(3)

        self.target_pos = self.np_random.uniform(self.target_min, self.target_max)
        self.state = np.concatenate([pos, vel]).astype(np.float32)
        self.step_count = 0
        self._prev_action = None

        return self._get_observation(), {'target_pos': self.target_pos}

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
        # FIX 2026-07-27: 根据渲染器类型使用不同的碰撞检测策略
        # - Mock: 球体碰撞 (与深度图渲染的障碍物位置一致)
        # - Real GS: 深度图碰撞 (使用实际场景几何, 而非硬编码球体)
        collision = False
        if isinstance(self.renderer, MockGSRenderer):
            # 球体碰撞检测 — 障碍物位置与 _base_obstacles_for_render 一致
            for obs_pos in self.obstacles:
                if np.linalg.norm(new_pos - obs_pos) <= self.collision_threshold + self.obstacle_radius:
                    collision = True; break
        elif isinstance(self.renderer, GSplatRenderer):
            # 深度图碰撞 — 使用真实3DGS场景几何
            # 渲染当前位姿的深度图, 取最小值检查是否太近
            depth_check, _ = self.renderer.render(new_pos)
            depth_min = float(depth_check[:, :, 0].min())
            # 深度 < 0.8m = 障碍物表面距相机中心不足0.8m
            # 加上无人机半径0.5m → 相当于碰撞
            if depth_min <= self.collision_threshold + 0.3:  # ~0.8m threshold
                collision = True

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
    def _compute_reward(self, pos, vel, action, collision, reached, target_dist):
        reward = 0.0
        r_dist = -5.0 * (1 - np.exp(-0.3 * target_dist))
        reward += r_dist

        speed = np.linalg.norm(vel)
        tgt_dir = (self.target_pos - pos)
        tgt_norm = tgt_dir / (np.linalg.norm(tgt_dir) + 1e-8)
        vel_norm = vel / (speed + 1e-8) if speed > 0 else np.zeros(3)
        r_heading = speed * np.clip(np.dot(vel_norm, tgt_norm), -1, 1) * 2.0
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
