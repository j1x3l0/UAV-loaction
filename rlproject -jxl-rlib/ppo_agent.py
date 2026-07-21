import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List, Dict, Union


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


class Actor(nn.Module):
    """策略网络：处理图像+向量混合输入"""
    
    def __init__(
        self, 
        vec_state_dim: int = 10,
        depth_image_size: int = 16,
        action_dim: int = 3, 
        hidden_dim: int = 256,
        use_depth_sensor: bool = True
    ):
        super(Actor, self).__init__()
        
        self.use_depth_sensor = use_depth_sensor
        self.vec_state_dim = vec_state_dim
        self.depth_image_size = depth_image_size

        if self.use_depth_sensor:
            self.image_encoder = DepthImageEncoder(in_channels=1, feature_dim=128, image_size=depth_image_size)
            self.vector_encoder = VectorEncoder(input_dim=vec_state_dim, feature_dim=128)
            self.fusion = FusionNetwork(image_feature_dim=128, vector_feature_dim=128, output_dim=hidden_dim)
        else:
            self.vector_encoder = VectorEncoder(input_dim=vec_state_dim, feature_dim=hidden_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0, requires_grad=True)
        self.action_max = 1.0
    
    def forward(
        self, 
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]]
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
        
        mean = torch.tanh(self.mean_layer(hidden)) * self.action_max
        std = torch.clamp(self.log_std.exp() * self.action_max, min=1e-3, max=max(1e-3, self.action_max))
        
        return mean, std


class Critic(nn.Module):
    """价值网络：处理图像+向量混合输入"""
    
    def __init__(
        self, 
        vec_state_dim: int = 10,
        depth_image_size: int = 16,
        hidden_dim: int = 256,
        use_depth_sensor: bool = True
    ):
        super(Critic, self).__init__()

        self.use_depth_sensor = use_depth_sensor

        if self.use_depth_sensor:
            self.image_encoder = DepthImageEncoder(in_channels=1, feature_dim=128, image_size=depth_image_size)
            self.vector_encoder = VectorEncoder(input_dim=vec_state_dim, feature_dim=128)
            self.fusion = FusionNetwork(image_feature_dim=128, vector_feature_dim=128, output_dim=hidden_dim)
        else:
            self.vector_encoder = VectorEncoder(input_dim=vec_state_dim, feature_dim=hidden_dim)
        
        self.value_out = nn.Linear(hidden_dim, 1)
    
    def forward(
        self, 
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
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
        
        return self.value_out(hidden).squeeze(dim=-1)


class PPO:
    """
    PPO核心类：支持传感器数据（深度图像）+ 向量状态的混合状态空间
    """
    
    def __init__(
            self,
            vec_state_dim: int = 10,
            action_dim: int = 3,
            action_max: float = 1.0,
            lr: float = 3e-4,
            gamma: float = 0.99,
            clip_eps: float = 0.2,
            epochs: int = 10,
            hidden_dim: int = 256,
            use_depth_sensor: bool = True,
            depth_image_size: int = 16
    ):
        self.use_depth_sensor = use_depth_sensor
        self.vec_state_dim = vec_state_dim
        self.depth_image_size = depth_image_size
        
        self.model = Actor(
            vec_state_dim=vec_state_dim,
            depth_image_size=depth_image_size,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            use_depth_sensor=use_depth_sensor
        ).to(DEVICE)
        
        self.model.action_max = action_max
        self.critic = Critic(
            vec_state_dim=vec_state_dim,
            depth_image_size=depth_image_size,
            hidden_dim=hidden_dim,
            use_depth_sensor=use_depth_sensor
        ).to(DEVICE)
        
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
        
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
    
    def _convert_observations_to_tensors(
        self, 
        observations: List[Dict[str, np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """
        将观测列表转换为张量
        
        Args:
            observations: 观测列表，每个元素是Dict{'vector': ..., 'depth': ...}
        
        Returns:
            包含向量和深度图像张量的字典
        """
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
    
    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """计算折扣回报"""
        returns = torch.zeros_like(rewards).to(DEVICE)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def get_action(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float]:
        """选择动作并返回动作及对数概率"""
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
            mean, std = self.model.forward(obs_dict)
            dist = Normal(mean, std)

            if deterministic:
                action_tensor = mean
            else:
                action_tensor = dist.sample()

            action_tensor = torch.clamp(action_tensor, -self.model.action_max, self.model.action_max)
            log_prob_tensor = dist.log_prob(action_tensor).sum(dim=-1)

        action = action_tensor.squeeze(0).cpu().numpy()
        log_prob = log_prob_tensor.squeeze(0).cpu().item()
        action = np.clip(action, -self.model.action_max, self.model.action_max)

        return action, log_prob

    def select_action(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = True
    ) -> np.ndarray:
        """选择动作（兼容评估脚本的接口）"""
        action, _ = self.get_action(observation, deterministic=deterministic)
        return action

    def store_transition(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_state: Dict[str, np.ndarray],
        done: bool,
        log_prob: float = 0.0
    ) -> None:
        """存储训练样本"""
        if not hasattr(self, 'memory'):
            self.memory = []
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })

    def update(self) -> float:
        """
        PPO更新：支持混合状态空间
        :return: 平均总损失
        """
        if not hasattr(self, 'memory') or len(self.memory) == 0:
            return 0.0

        observations = [t['state'] for t in self.memory]
        actions = [t['action'] for t in self.memory]
        rewards = [t['reward'] for t in self.memory]
        next_states = [t['next_state'] for t in self.memory]
        dones = [t['done'] for t in self.memory]
        log_probs = [t['log_prob'] for t in self.memory]

        observations_tensor = self._convert_observations_to_tensors(observations)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE)
        old_log_probs_tensor = torch.tensor(np.array(log_probs), dtype=torch.float32).to(DEVICE)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(DEVICE)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32).to(DEVICE)
        
        returns = self._compute_returns(rewards_tensor, dones_tensor)
        
        total_loss = 0.0
        for _ in range(self.epochs):
            mean, std = self.model.forward(observations_tensor)
            dist = Normal(mean, std)
            current_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
            values = self.critic.forward(observations_tensor)
            
            ratio = torch.exp(current_log_probs - old_log_probs_tensor)
            surr1 = ratio * (returns - values.detach())
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * (returns - values.detach())
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(values, returns)
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.memory = []
        return total_loss / self.epochs
    
    def save_model(self, path: str = "drone_ppo_model.pth") -> None:
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'use_depth_sensor': self.use_depth_sensor,
            'vec_state_dim': self.vec_state_dim,
            'depth_image_size': self.depth_image_size
        }, path)
        print(f"✅ 模型已保存至：{path}")
    
    def load_model(self, path: str = "drone_ppo_model.pth") -> None:
        """
        加载模型
        """
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
                print(f"✅ 成功加载模型参数")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"✅ 成功加载模型参数（旧格式）")
        except FileNotFoundError:
            print(f"❌ 未找到模型文件：{path}，从头训练")
        except Exception as e:
            print(f"❌ 加载模型失败：{e}，从头训练")