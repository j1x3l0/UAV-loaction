import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Dict, Any, Tuple, Optional
from scipy.spatial.transform import Rotation as R


class DroneEnv(gym.Env):
    """
    无人机路径规划环境（三维）
    状态空间：[x, y, z, vx, vy, vz, qw, qx, qy, qz] + 深度图像 (H, W)
    动作空间：[thrust_x, thrust_y, thrust_z]（3维连续动作，范围[-1,1]）
    奖励函数符合V1评估指标
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super(DroneEnv, self).__init__()
        
        # 默认配置
        self.config = config or {}
        
        # 传感器配置
        self.use_depth_sensor = self.config.get('use_depth_sensor', True)
        self.depth_image_size = self.config.get('depth_image_size', 16)  # 深度图像尺寸
        self.depth_fov = self.config.get('depth_fov', 60)  # 深度相机视场角（度）
        self.depth_max_range = self.config.get('depth_max_range', 20.0)  # 最大深度范围（米）
        
        # 奖励函数版本: 'v1' 或 'v2'
        self.reward_version = self.config.get('reward_version', 'v2')
        
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
        
        # 构建状态空间
        if self.use_depth_sensor:
            self.vec_state_dim = 10  # 原始向量状态维度
            self.depth_h = self.depth_image_size
            self.depth_w = self.depth_image_size
            
            # 向量状态空间
            vec_low = np.array([self.boundary_min[0], self.boundary_min[1], self.boundary_min[2],
                               -10.0, -10.0, -10.0,
                               -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            vec_high = np.array([self.boundary_max[0], self.boundary_max[1], self.boundary_max[2],
                                10.0, 10.0, 10.0,
                                1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            
            # 深度图像空间
            depth_low = np.zeros((self.depth_h, self.depth_w), dtype=np.float32)
            depth_high = np.full((self.depth_h, self.depth_w), self.depth_max_range, dtype=np.float32)
            
            # 复合状态空间：Dict类型，包含向量状态和深度图像
            self.observation_space = spaces.Dict({
                'vector': spaces.Box(low=vec_low, high=vec_high, dtype=np.float32),
                'depth': spaces.Box(low=depth_low, high=depth_high, dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Box(
                low=np.array([self.boundary_min[0], self.boundary_min[1], self.boundary_min[2],
                             -10.0, -10.0, -10.0,
                             -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([self.boundary_max[0], self.boundary_max[1], self.boundary_max[2],
                              10.0, 10.0, 10.0,
                              1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        
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
        
    def _generate_obstacles(self) -> np.ndarray:
        """
        生成静态障碍物位置
        """
        obstacles = np.array([
            [2.0, 2.0, 3.0],
            [6.0, 3.0, 5.0],
            [3.0, 7.0, 4.0]
        ])
        return obstacles
    
    def _generate_depth_image(self) -> np.ndarray:
        """
        生成模拟深度图像
        基于针孔相机模型，从无人机视角生成深度图
        
        Returns:
            depth_image: (H, W) 的深度图像，单位为米
        """
        H, W = self.depth_h, self.depth_w
        depth_image = np.full((H, W), self.depth_max_range, dtype=np.float32)
        
        # 获取无人机位置和朝向
        pos = self.state[:3]
        quat = self.state[6:10]
        
        # 提取四元数中的旋转
        rotation = R.from_quat(quat)
        
        # 相机内参
        fov_rad = np.radians(self.depth_fov)
        focal_length = (W / 2) / np.tan(fov_rad / 2)
        cx, cy = W / 2, H / 2
        
        # 障碍物和目标的位置（用于检测）
        objects = []
        for obstacle in self.obstacles:
            objects.append({
                'position': obstacle,
                'radius': self.obstacle_radius,
                'is_target': False
            })
        objects.append({
            'position': self.target_pos,
            'radius': self.target_threshold,
            'is_target': True
        })
        
        # 边界检测（简化版：检测墙壁）
        boundary_objects = []
        for i, axis in enumerate(['x', 'y', 'z']):
            min_val = self.boundary_min[i]
            max_val = self.boundary_max[i]
            if min_val != -np.inf:
                point_min = np.array(pos)
                point_min[i] = min_val
                boundary_objects.append({'position': point_min, 'radius': 0.1, 'is_target': False})
            if max_val != np.inf:
                point_max = np.array(pos)
                point_max[i] = max_val
                boundary_objects.append({'position': point_max, 'radius': 0.1, 'is_target': False})
        
        all_objects = objects + boundary_objects
        
        # 遍历图像平面每个像素
        for v in range(H):
            for u in range(W):
                # 像素坐标转换到归一化相机坐标
                x = (u - cx) / focal_length
                y = (v - cy) / focal_length
                z = 1.0
                
                # 归一化方向向量
                ray_dir = np.array([x, y, z])
                ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-8)
                
                # 将射线方向转换到世界坐标系
                ray_dir_world = rotation.apply(ray_dir)
                
                # 射线-球体相交检测（障碍物和目标）
                min_dist = self.depth_max_range
                hit_object = None
                
                for obj in all_objects:
                    # 计算射线到球心的最近点
                    oc = obj['position'] - pos
                    t = np.dot(oc, ray_dir_world)
                    
                    if t < 0:
                        continue
                    
                    closest_point = pos + t * ray_dir_world
                    dist_to_center = np.linalg.norm(closest_point - obj['position'])
                    
                    # 如果射线穿过球体
                    if dist_to_center <= obj['radius']:
                        # 计算实际交点距离
                        dist_to_hit = t - np.sqrt(obj['radius']**2 - dist_to_center**2)
                        if 0 < dist_to_hit < min_dist:
                            min_dist = dist_to_hit
                            hit_object = obj
                
                if min_dist < self.depth_max_range:
                    depth_image[v, u] = min_dist
        
        # 深度图像后处理：添加一些噪声模拟真实传感器
        noise_level = 0.02  # 2% 的相对误差
        depth_image = depth_image + np.random.normal(0, depth_image * noise_level + 0.01, depth_image.shape)
        depth_image = np.clip(depth_image, 0, self.depth_max_range)
        
        return depth_image
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        构建观测
        
        Returns:
            如果使用深度传感器：Dict{'vector': ..., 'depth': ...}
            否则：原始状态向量
        """
        if self.use_depth_sensor:
            depth_image = self._generate_depth_image()
            return {
                'vector': self.state.copy(),
                'depth': depth_image
            }
        else:
            return self.state.copy()
    
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境
        :param seed: 随机种子
        :param options: 额外选项
        :return: 初始状态和信息字典
        """
        super().reset(seed=seed)

        self.np_random, _ = seeding.np_random(seed)

        start_min = self.boundary_min + 1.0
        start_max = np.array([2.0, 2.0, 2.0])
        pos = self.np_random.uniform(start_min, start_max)
        
        vel = np.zeros(3)
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        
        self.target_pos = self.np_random.uniform(self.target_min, self.target_max)
        
        self.state = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], qw, qx, qy, qz], dtype=np.float32)
        
        self.step_count = 0
        
        info = {
            "target_pos": self.target_pos,
            "obstacles": self.obstacles
        }
        
        return self._get_observation(), info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行一步环境
        :param action: 动作 [thrust_x, thrust_y, thrust_z]，范围[-1,1]
        :return: 下一状态、奖励、是否终止、是否截断、信息
        """
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
            new_vel[0], new_vel[1], new_vel[2],
            self.state[6], self.state[7], self.state[8], self.state[9]
        ], dtype=np.float32)
        
        self.state = new_state
        self.step_count += 1
        
        target_distance = np.linalg.norm(new_pos - self.target_pos)
        old_target_distance = np.linalg.norm(pos - self.target_pos)
        distance_improvement = old_target_distance - target_distance
        
        if self.reward_version == 'v1':
            reward = self._compute_reward_v1(
                boundary_hit, collision, reached_target, distance_improvement, target_distance
            )
        else:
            reward = self._compute_reward_v2(
                boundary_hit, collision, reached_target, distance_improvement, target_distance
            )
        
        terminated = reached_target or collision
        truncated = self.step_count >= self.max_steps
        
        info = {
            "target_pos": self.target_pos,
            "current_pos": new_pos,
            "target_distance": target_distance,
            "collision": collision,
            "reached_target": reached_target,
            "boundary_hit": boundary_hit,
            "step_count": self.step_count,
            "reward_version": self.reward_version
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _compute_reward_v1(self, boundary_hit, collision, reached_target, distance_improvement, target_distance):
        """
        V1奖励函数：基于距离的负惩罚 + 事件奖励
        特点：鼓励快速到达目标，减少不必要的探索
        """
        reward = 0.0
        reward += distance_improvement * 20.0
        if boundary_hit:
            reward += -10.0
        if collision:
            reward += -50.0
        if reached_target:
            reward += 200.0
        reward = np.clip(reward, -100, 200)
        return reward
    
    def _compute_reward_v2(self, boundary_hit, collision, reached_target, distance_improvement, target_distance):
        """
        V2奖励函数：稀疏奖励 + 生存奖励
        特点：更平滑的奖励曲线，鼓励稳定飞行
        """
        reward = 0.0
        reward += distance_improvement * 10.0
        if boundary_hit:
            reward += -5.0
        if collision:
            reward += -20.0
        if reached_target:
            reward += 100.0
        reward = np.clip(reward, -50, 100)
        return reward
    
    def render(self, mode: str = "human") -> Any:
        """
        渲染环境
        预留接口：未来可实现可视化
        """
        pass
    
    def close(self) -> None:
        """
        关闭环境
        """
        pass

# 测试环境是否正常工作
if __name__ == "__main__":
    env = DroneEnv()
    
    # 测试重置
    state, info = env.reset()
    print("初始状态:", state)
    print("目标位置:", info["target_pos"])
    
    # 测试一步
    action = np.array([0.5, 0.5, 0.1])
    next_state, reward, terminated, truncated, info = env.step(action)
    print("\n执行动作:", action)
    print("下一状态:", next_state)
    print("奖励:", reward)
    print("终止:", terminated)
    print("截断:", truncated)
    print("信息:", info)
    
    print("\n环境测试通过！")