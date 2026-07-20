import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DroneEnv(gym.Env):
    """
    无人机路径规划环境（三维）- Swift改进版
    状态空间：14维向量 [x, y, z, vx, vy, vz, dx, dy, dz, dist_to_target, obs_dir_x, obs_dir_y, obs_dir_z, dist_to_obs]
    动作空间：[thrust_x, thrust_y, thrust_z]（3维连续动作，范围[-1,1]）
    奖励函数：7组件分层奖励（Swift风格增强版）
    """

    def __init__(self, config: Dict[str, Any] = None):
        super(DroneEnv, self).__init__()

        self.config = config or {}

        # 环境边界（单位：米）
        self.boundary_min = np.array([-10.0, -10.0, 0.0])
        self.boundary_max = np.array([10.0, 10.0, 10.0])

        # 目标位置（随机生成范围）
        self.target_min = np.array([5.0, 5.0, 2.0])
        self.target_max = np.array([8.0, 8.0, 8.0])

        # 无人机物理参数
        self.drone_mass = 1.0
        self.max_thrust = 10.0
        self.dt = 0.05

        # 障碍物配置
        self.obstacles = self._generate_obstacles()
        self.obstacle_radius = 1.0

        # 随机数生成器
        self.np_random, _ = seeding.np_random(None)

        # 构建14维状态空间
        self._build_observation_space()

        # 动作空间
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        self.target_pos = None
        self.state = None
        self.step_count = 0
        self.max_steps = 500
        self.collision_threshold = 0.5
        self.target_threshold = 0.5

        # 用于动作平滑奖励
        self._prev_action = None

        # 状态归一化统计（运行时均值和标准差）
        self.state_mean = np.zeros(14, dtype=np.float32)
        self.state_std = np.ones(14, dtype=np.float32)
        self.state_count = 0

        logger.info("DroneEnv (Swift改进版) 初始化完成 | 状态维度: 14 | 奖励组件: 7")

    def _build_observation_space(self):
        """构建14维紧凑状态空间"""
        pos_low = self.boundary_min
        pos_high = self.boundary_max
        vel_low = np.array([-10.0, -10.0, -10.0])
        vel_high = np.array([10.0, 10.0, 10.0])
        rel_pos_low = np.array([-20.0, -20.0, -10.0])
        rel_pos_high = np.array([20.0, 20.0, 10.0])
        dist_low = np.array([0.0])
        dist_high = np.array([30.0])
        obs_dir_low = np.array([-1.0, -1.0, -1.0])
        obs_dir_high = np.array([1.0, 1.0, 1.0])

        low = np.concatenate([pos_low, vel_low, rel_pos_low, dist_low, obs_dir_low, dist_low])
        high = np.concatenate([pos_high, vel_high, rel_pos_high, dist_high, obs_dir_high, dist_high])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.vec_state_dim = 14

    def _generate_obstacles(self) -> np.ndarray:
        """生成静态障碍物位置"""
        return np.array([
            [2.0, 2.0, 3.0],
            [6.0, 3.0, 5.0],
            [3.0, 7.0, 4.0]
        ])

    def _get_min_obstacle_distance(self, pos: np.ndarray) -> float:
        """计算当前位置到所有障碍物的最小距离（减去半径）"""
        min_dist = float('inf')
        for obs_pos in self.obstacles:
            dist = np.linalg.norm(pos - obs_pos) - self.obstacle_radius
            if dist < min_dist:
                min_dist = dist
        return max(min_dist, 0.0)

    def _get_obstacle_info(self, pos: np.ndarray) -> Tuple[np.ndarray, float]:
        """获取最近障碍物的方向和距离"""
        min_dist = float('inf')
        closest_obs = None
        for obs_pos in self.obstacles:
            dist = np.linalg.norm(pos - obs_pos) - self.obstacle_radius
            if dist < min_dist:
                min_dist = dist
                closest_obs = obs_pos

        if closest_obs is not None:
            obs_dir = closest_obs - pos
            obs_dir_norm = obs_dir / (np.linalg.norm(obs_dir) + 1e-8)
        else:
            obs_dir_norm = np.zeros(3)

        return obs_dir_norm, max(min_dist, 0.0)

    def _build_state_vector(self) -> np.ndarray:
        """构建14维紧凑状态向量"""
        pos = self.state[:3]
        vel = self.state[3:6]

        target_dir = self.target_pos - pos
        dist_to_target = np.linalg.norm(target_dir)

        obs_dir, dist_to_obs = self._get_obstacle_info(pos)

        state_vector = np.array([
            pos[0], pos[1], pos[2],           # 位置 (3)
            vel[0], vel[1], vel[2],           # 速度 (3)
            target_dir[0], target_dir[1], target_dir[2],  # 目标相对位置 (3)
            dist_to_target,                    # 目标距离 (1)
            obs_dir[0], obs_dir[1], obs_dir[2],           # 障碍物方向 (3)
            dist_to_obs                        # 障碍物距离 (1)
        ], dtype=np.float32)  # 共14维

        return state_vector

    def _get_observation(self):
        """构建观测并更新状态归一化统计"""
        state_vector = self._build_state_vector()

        # 运行时统计均值和标准差
        self.state_count += 1
        self.state_mean = ((self.state_count - 1) * self.state_mean + state_vector) / self.state_count
        diff = state_vector - self.state_mean
        self.state_std = np.sqrt(((self.state_count - 1) * self.state_std**2 + diff**2) / self.state_count)

        return state_vector.copy()

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple:
        super().reset(seed=seed)

        self.np_random, _ = seeding.np_random(seed)

        start_min = self.boundary_min + 1.0
        start_max = np.array([2.0, 2.0, 2.0])
        pos = self.np_random.uniform(start_min, start_max)

        vel = np.zeros(3)

        self.target_pos = self.np_random.uniform(self.target_min, self.target_max)

        self.state = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]], dtype=np.float32)

        self.step_count = 0
        self._prev_action = None

        info = {
            "target_pos": self.target_pos,
            "obstacles": self.obstacles,
            "state_mean": self.state_mean.copy(),
            "state_std": self.state_std.copy()
        }

        return self._get_observation(), info

    def step(self, action: np.ndarray) -> Tuple:
        thrust = action * self.max_thrust

        pos = self.state[:3]
        vel = self.state[3:6]

        accel = thrust / self.drone_mass

        new_vel = vel + accel * self.dt

        max_velocity = 5.0
        new_vel = np.clip(new_vel, -max_velocity, max_velocity)

        new_pos = pos + vel * self.dt + 0.5 * accel * self.dt**2

        boundary_hit = False
        for i in range(3):
            if new_pos[i] <= self.boundary_min[i]:
                new_pos[i] = self.boundary_min[i]
                new_vel[i] = -new_vel[i] * 0.5
                boundary_hit = True
            elif new_pos[i] >= self.boundary_max[i]:
                new_pos[i] = self.boundary_max[i]
                new_vel[i] = -new_vel[i] * 0.5
                boundary_hit = True

        collision = False
        for obstacle_pos in self.obstacles:
            distance = np.linalg.norm(new_pos - obstacle_pos)
            if distance <= self.collision_threshold + self.obstacle_radius:
                collision = True
                break

        target_distance = np.linalg.norm(new_pos - self.target_pos)
        reached_target = target_distance <= self.target_threshold

        new_state = np.array([
            new_pos[0], new_pos[1], new_pos[2],
            new_vel[0], new_vel[1], new_vel[2]
        ], dtype=np.float32)

        self.state = new_state
        self.step_count += 1

        reward, reward_components = self._compute_reward(new_state, action, collision, reached_target, target_distance)

        terminated = reached_target or collision
        truncated = self.step_count >= self.max_steps

        self._prev_action = action.copy()

        info = {
            "target_pos": self.target_pos,
            "current_pos": new_pos,
            "target_distance": target_distance,
            "collision": collision,
            "reached_target": reached_target,
            "boundary_hit": boundary_hit,
            "step_count": self.step_count,
            "reward_components": reward_components
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _compute_reward(self, state, action, collision, reached_target, target_distance):
        """
        7组件分层奖励函数（Swift风格增强版）
        组件：距离引导、速度方向、障碍物惩罚、动作平滑、到达奖励、碰撞惩罚、超时惩罚
        """
        reward = 0.0
        components = {}

        pos = state[:3]
        vel = state[3:6]

        # 1. 距离引导奖励（指数衰减，借鉴 Swift 的 progress 奖励思想）
        r_dist = -5.0 * (1 - np.exp(-0.3 * target_distance))
        reward += r_dist
        components['r_dist'] = r_dist

        # 2. 速度方向奖励（借鉴 Swift 的 progress 奖励）
        speed = np.linalg.norm(vel)
        target_dir = (self.target_pos - pos)
        target_dir_norm = target_dir / (np.linalg.norm(target_dir) + 1e-8)
        vel_norm = vel / (speed + 1e-8) if speed > 0 else np.zeros(3)
        cos_angle = np.clip(np.dot(vel_norm, target_dir_norm), -1, 1)
        r_heading = speed * cos_angle * 2.0
        reward += r_heading
        components['r_heading'] = r_heading

        # 3. 障碍物接近惩罚（势场式，渐进式 — Swift 未涉及的改进）
        min_obs_dist = self._get_min_obstacle_distance(pos)
        r_obs = -2.0 / (min_obs_dist + 0.5)
        reward += r_obs
        components['r_obs'] = r_obs

        # 4. 动作平滑奖励（借鉴 Swift 的控制平滑性奖励）
        if self._prev_action is not None:
            r_smooth = -0.5 * np.linalg.norm(action - self._prev_action)
        else:
            r_smooth = 0.0
        reward += r_smooth
        components['r_smooth'] = r_smooth

        # 5. 到达奖励（基础到达奖励 + 时间效率奖励）
        if reached_target:
            remaining_steps_ratio = (self.max_steps - self.step_count) / self.max_steps
            r_goal = 100.0 + 50.0 * remaining_steps_ratio
        else:
            r_goal = 0.0
        reward += r_goal
        components['r_goal'] = r_goal

        # 6. 碰撞惩罚
        if collision:
            r_collision = -50.0
        else:
            r_collision = 0.0
        reward += r_collision
        components['r_collision'] = r_collision

        # 7. 超时惩罚
        if self.step_count >= self.max_steps and target_distance >= self.target_threshold:
            r_timeout = -10.0
        else:
            r_timeout = 0.0
        reward += r_timeout
        components['r_timeout'] = r_timeout

        reward = np.clip(reward, -100, 200)
        components['total'] = reward

        return reward, components

    def render(self, mode: str = "human"):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env = DroneEnv()

    state, info = env.reset()
    print("初始状态维度:", state.shape)
    print("初始状态:", state)
    print("目标位置:", info["target_pos"])

    action = np.array([0.5, 0.5, 0.1])
    next_state, reward, terminated, truncated, info = env.step(action)
    print("\n执行动作:", action)
    print("下一状态维度:", next_state.shape)
    print("奖励:", reward)
    print("奖励分解:", info["reward_components"])
    print("终止:", terminated)
    print("截断:", truncated)

    print("\n环境测试通过！")
