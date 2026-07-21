import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List, Dict, Union, Optional
import random
from collections import deque


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DepthImageEncoder(nn.Module):
    """深度图像编码器：轻量级CNN专为16x16图像优化"""
  
    def __init__(self, in_channels: int = 1, feature_dim: int = 128, image_size: int = 16):
        super(DepthImageEncoder, self).__init__()

        # 轻量级2层卷积架构，适合16x16图像
        self.conv_layers = nn.Sequential(
            # 第一层：保持尺寸 16x16 → 16x16 (特征提取)
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            # 第二层：降采样 16x16 → 8x8 (空间压缩)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        # 全局平均池化 + 全连接层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
            nn.ReLU(),
        )
    
    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        if depth_image.dim() == 3:
            depth_image = depth_image.unsqueeze(1)

        x = self.conv_layers(depth_image)
        x = self.classifier(x)

        return x


class VectorEncoder(nn.Module):
    """向量状态编码器：MLP提取向量特征"""
    
    def __init__(self, input_dim: int = 10, feature_dim: int = 128):
        super(VectorEncoder, self).__init__()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )
    
    def forward(self, vector_state: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(vector_state)


class FusionNetwork(nn.Module):
    """多模态融合网络：拼接图像和向量特征"""
    
    def __init__(self, image_feature_dim: int = 128, vector_feature_dim: int = 128, output_dim: int = 256):
        super(FusionNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(image_feature_dim + vector_feature_dim, output_dim),
            nn.ReLU(),
        )
    
    def forward(self, image_features: torch.Tensor, vector_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([image_features, vector_features], dim=-1)
        return self.fc(combined)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return obs, action, reward, next_obs, done
    
    def __len__(self):
        return len(self.buffer)


class SACActor(nn.Module):
    """SAC策略网络"""
    
    def __init__(
        self, 
        vec_state_dim: int = 10,
        depth_image_size: int = 16,
        action_dim: int = 3, 
        hidden_dim: int = 256,
        use_depth_sensor: bool = True
    ):
        super(SACActor, self).__init__()
        
        self.use_depth_sensor = use_depth_sensor
        self.vec_state_dim = vec_state_dim
        self.depth_image_size = depth_image_size
        self.action_dim = action_dim
        
        if self.use_depth_sensor:
            self.image_encoder = DepthImageEncoder(in_channels=1, feature_dim=128, image_size=depth_image_size)
            self.vector_encoder = VectorEncoder(input_dim=vec_state_dim, feature_dim=128)
            self.fusion = FusionNetwork(image_feature_dim=128, vector_feature_dim=128, output_dim=hidden_dim)
        else:
            self.vector_encoder = VectorEncoder(input_dim=vec_state_dim, feature_dim=hidden_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.action_max = 1.0
        
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(
        self, 
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
        with_log_prob: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_depth_sensor:
            vector_state = observation['vector']
            depth_image = observation['depth']
            
            image_features = self.image_encoder(depth_image)
            vector_features = self.vector_encoder(vector_state)
            fused_features = self.fusion(image_features, vector_features)
            hidden = fused_features
        else:
            if isinstance(observation, dict):
                vector_state = observation['vector']
            else:
                vector_state = observation
            hidden = self.vector_encoder(vector_state)
        
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        
        if deterministic:
            action = torch.tanh(mean) * self.action_max
            # 在deterministic模式下，log_prob应为每个样本一个标量0
            log_prob = torch.zeros(action.shape[0])
        else:
            u = dist.rsample()
            action = torch.tanh(u) * self.action_max
            
            if with_log_prob:
                log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1)
            else:
                log_prob = None
        
        return action, log_prob
    
    def get_action(
        self, 
        observation: Union[np.ndarray, Dict[str, np.ndarray]], 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float]:
        if self.use_depth_sensor:
            vector_state = torch.tensor(observation['vector'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            depth_image = torch.tensor(observation['depth'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            obs_dict = {'vector': vector_state, 'depth': depth_image}
        else:
            if isinstance(observation, dict):
                vector_state = torch.tensor(observation['vector'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            else:
                vector_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            obs_dict = vector_state
        
        with torch.no_grad():
            action, log_prob = self.forward(obs_dict, deterministic=deterministic, with_log_prob=True)
            action = action.squeeze(0).cpu().numpy()
            log_prob = log_prob.squeeze(0).cpu().item()
            action = np.clip(action, -self.action_max, self.action_max)
        
        return action, log_prob


class SACCritic(nn.Module):
    """SAC双Q网络"""
    
    def __init__(
        self, 
        vec_state_dim: int = 10,
        depth_image_size: int = 16,
        hidden_dim: int = 256,
        use_depth_sensor: bool = True,
        action_dim: int = 3
    ):
        super(SACCritic, self).__init__()
        
        self.use_depth_sensor = use_depth_sensor
        
        if self.use_depth_sensor:
            self.image_encoder = DepthImageEncoder(in_channels=1, feature_dim=128, image_size=depth_image_size)
            self.vector_encoder = VectorEncoder(input_dim=vec_state_dim, feature_dim=128)
            self.fusion = FusionNetwork(image_feature_dim=128, vector_feature_dim=128, output_dim=hidden_dim)
        else:
            self.vector_encoder = VectorEncoder(input_dim=vec_state_dim, feature_dim=hidden_dim)
        
        # 计算特征维度：融合后的特征维度或向量特征维度
        self.feature_dim = hidden_dim
        
        # Q网络的输入是特征和动作的拼接
        self.q1 = nn.Sequential(
            nn.Linear(self.feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(self.feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self, 
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_depth_sensor:
            vector_state = observation['vector']
            depth_image = observation['depth']
            
            image_features = self.image_encoder(depth_image)
            vector_features = self.vector_encoder(vector_state)
            fused_features = self.fusion(image_features, vector_features)
            hidden = fused_features
        else:
            if isinstance(observation, dict):
                vector_state = observation['vector']
            else:
                vector_state = observation
            hidden = self.vector_encoder(vector_state)
        
        # 将特征与动作拼接作为Q网络的输入
        q_input = torch.cat([hidden, action], dim=-1)
        
        q1 = self.q1(q_input).squeeze(dim=-1)
        q2 = self.q2(q_input).squeeze(dim=-1)
        
        return q1, q2


class SAC:
    """
    SAC（Soft Actor-Critic）算法：支持传感器数据（深度图像）+ 向量状态的混合状态空间
    """
    
    def __init__(
            self,
            vec_state_dim: int = 10,
            action_dim: int = 3,
            action_max: float = 1.0,
            lr: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            buffer_size: int = 100000,
            batch_size: int = 256,
            hidden_dim: int = 256,
            use_depth_sensor: bool = True,
            depth_image_size: int = 16,
            initial_alpha: float = 0.2,
            target_entropy: Optional[float] = None
    ):
        self.use_depth_sensor = use_depth_sensor
        self.vec_state_dim = vec_state_dim
        self.depth_image_size = depth_image_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        self.actor = SACActor(
            vec_state_dim=vec_state_dim,
            depth_image_size=depth_image_size,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            use_depth_sensor=use_depth_sensor
        ).to(DEVICE)
        self.actor.action_max = action_max
        
        self.critic = SACCritic(
            vec_state_dim=vec_state_dim,
            depth_image_size=depth_image_size,
            hidden_dim=hidden_dim,
            use_depth_sensor=use_depth_sensor,
            action_dim=action_dim
        ).to(DEVICE)
        
        self.critic_target = SACCritic(
            vec_state_dim=vec_state_dim,
            depth_image_size=depth_image_size,
            hidden_dim=hidden_dim,
            use_depth_sensor=use_depth_sensor,
            action_dim=action_dim
        ).to(DEVICE)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.log_alpha = torch.tensor(np.log(initial_alpha), dtype=torch.float32, requires_grad=True, device=DEVICE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy
        
        self.alpha = self.log_alpha.exp()
        
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: torch.Tensor):
        self._alpha = value
    
    def get_action(self, observation: Union[np.ndarray, Dict[str, np.ndarray]], deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """获取动作"""
        return self.actor.get_action(observation, deterministic=deterministic)
    
    def _convert_observations_to_tensors(
        self, 
        observations: List[Dict[str, np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """将观测列表转换为张量"""
        vectors = []
        depths = []
        
        for obs in observations:
            vectors.append(obs['vector'])
            depths.append(obs['depth'])
        
        vectors_tensor = torch.tensor(np.array(vectors), dtype=torch.float32).to(DEVICE)
        depths_tensor = torch.tensor(np.array(depths), dtype=torch.float32).to(DEVICE)
        
        return {
            'vector': vectors_tensor,
            'depth': depths_tensor
        }
    
    def update(self, updates: int = 1) -> Dict[str, float]:
        """
        SAC更新
        :param updates: 更新次数
        :return: 损失字典
        """
        if len(self.replay_buffer) < self.batch_size:
            return {'actor_loss': 0, 'critic_loss': 0, 'alpha_loss': 0, 'alpha': 0}
        
        actor_losses = []
        critic_losses = []
        alpha_losses = []
        alphas = []
        
        for _ in range(updates):
            obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)
            
            obs_tensor = self._convert_observations_to_tensors(obs)
            next_obs_tensor = self._convert_observations_to_tensors(next_obs)
            action_tensor = torch.tensor(np.array(action), dtype=torch.float32).to(DEVICE)
            reward_tensor = torch.tensor(np.array(reward), dtype=torch.float32).to(DEVICE)
            done_tensor = torch.tensor(np.array(done), dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                next_action, next_log_prob = self.actor(next_obs_tensor, with_log_prob=True)
                q1_target, q2_target = self.critic_target(next_obs_tensor, next_action)
                q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
                q_target = reward_tensor + (1 - done_tensor) * self.gamma * q_target
            
            q1, q2 = self.critic(obs_tensor, action_tensor)
            critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            critic_losses.append(critic_loss.item())
            
            new_action, log_prob = self.actor(obs_tensor, with_log_prob=True)
            q1_new, q2_new = self.critic(obs_tensor, new_action)
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.alpha * log_prob - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_losses.append(actor_loss.item())
            
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_losses.append(alpha_loss.item())
            
            self.alpha = self.log_alpha.exp()
            alphas.append(self.alpha.item())
            
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'alpha_loss': np.mean(alpha_losses),
            'alpha': np.mean(alphas)
        }
    
    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)
    
    def save_model(self, path: str = "drone_sac_model.pth") -> None:
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'use_depth_sensor': self.use_depth_sensor,
            'vec_state_dim': self.vec_state_dim,
            'depth_image_size': self.depth_image_size
        }, path)
        print(f"✅ SAC模型已保存至：{path}")
    
    def load_model(self, path: str = "drone_sac_model.pth") -> None:
        """加载模型"""
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.log_alpha.data.fill_(checkpoint['log_alpha'])
            self.alpha = self.log_alpha.exp()
            print(f"✅ SAC模型加载成功")
        except FileNotFoundError:
            print(f"❌ 未找到模型文件：{path}，从头训练")
        except Exception as e:
            print(f"❌ 加载模型失败：{e}，从头训练")
