import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class AdaptiveEntropyCoeff:
    """自适应熵正则化系数，根据策略熵动态调整探索强度（P0修复版）
    
    P0修复：
    - 目标改为正值 (2.0-3.0)，不再是负的 -action_dim
    - 加 log_alpha 上下界 [-9.2, -1.6]（对应 alpha ≈ 1e-4 ~ 0.2）
    - 防止 log_alpha 单调爆炸
    """

    def __init__(self, initial_coeff=0.01, target_entropy=None, lr=3e-4, action_dim=3):
        # P0修复：目标改为正熵值而不是负数
        if target_entropy is None:
            self.target_entropy = 2.5  # 经验值，不再用 -action_dim
        else:
            self.target_entropy = max(0.5, target_entropy)  # 确保为正
        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.log_alpha.data.fill_(np.log(initial_coeff))
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.initial_coeff = initial_coeff
        
        # P0修复：加上下界防止爆炸
        self.log_alpha_min = np.log(1e-4)    # alpha_min ≈ 1e-4
        self.log_alpha_max = np.log(0.2)     # alpha_max ≈ 0.2
        
        self.update_count = 0  # 用于监控

    def get_coeff(self):
        return torch.clamp(self.log_alpha, self.log_alpha_min, self.log_alpha_max).exp().item()

    def update(self, entropy):
        """根据当前策略熵更新 alpha
        
        熵高于目标时降低 alpha，熵低于目标时提高 alpha。
        """
        if self.target_entropy is None:
            return

        # 梯度下降下必须使用正号：entropy > target 时 log_alpha 减小。
        loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # P0修复：手动裁剪 log_alpha，防止梯度突破上下界
        with torch.no_grad():
            self.log_alpha.data.clamp_(self.log_alpha_min, self.log_alpha_max)
        
        self.update_count += 1


class ActorCritic(nn.Module):
    """
    Swift风格 Actor-Critic 网络（两层 MLP，128x128）
    Actor 和 Critic 共享第一层特征提取器
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # 共享第一层特征提取器（Swift 实践）
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor 网络: Linear(128, 128) -> ReLU -> Linear(128, action_dim)
        self.actor_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic 网络: Linear(128, 128) -> ReLU -> Linear(128, 1)
        self.critic_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # log_std 作为可学习参数
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal 初始化（Swift 实践）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        # 输出层使用较小的 gain（Swift 实践）
        nn.init.orthogonal_(self.actor_layers[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_layers[-1].weight, gain=1.0)

    def forward(self, state):
        features = self.shared_layer(state)

        mean = torch.tanh(self.actor_layers(features))
        value = self.critic_layers(features).squeeze(dim=-1)

        std = torch.clamp(self.log_std.exp(), min=1e-3, max=1.0)

        return mean, std, value


class PPO:
    """
    PPO 智能体（Swift 改进版）
    特性：Swift风格两层MLP、GAE(λ=0.95)、自适应熵系数、PPO-Clip
    """

    def __init__(
            self,
            state_dim: int = 14,
            action_dim: int = 3,
            action_max: float = 1.0,
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_eps: float = 0.2,
            epochs: int = 10,
            minibatch_size: int = 64,
            hidden_dim: int = 128,
            use_adaptive_entropy: bool = True,
            num_envs: int = 8
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = action_max
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.num_envs = num_envs

        # Swift 风格 Actor-Critic 网络
        self.model = ActorCritic(state_dim, action_dim, hidden_dim).to(DEVICE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 自适应熵系数（P0修复版）
        if use_adaptive_entropy:
            self.entropy_coeff = AdaptiveEntropyCoeff(
                initial_coeff=0.01,
                target_entropy=2.5,  # P0修复：改为正值
                lr=lr,
                action_dim=action_dim
            )
        else:
            self.entropy_coeff = None
            self.fixed_entropy_coeff = 0.01

        self.current_lr = lr
        self.initial_lr = lr

        logger.info(f"PPO (Swift改进版) 初始化 | state_dim={state_dim}, action_dim={action_dim}")
        logger.info(f"超参数: gamma={gamma}, gae_lambda={gae_lambda}, clip_eps={clip_eps}")
        logger.info(f"训练: epochs={epochs}, minibatch_size={minibatch_size}")
        logger.info(f"自适应熵: {use_adaptive_entropy}")

    def set_lr(self, lr):
        """设置学习率（用于线性衰减）"""
        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_gae(self, rewards, values, dones, next_value):
        """
        计算广义优势估计 (GAE) - P0修复版
        参数:
            rewards: list of rewards, length T
            values: list of value predictions, length T
            dones: list of done flags, length T (1 if episode terminated, 0 otherwise)
            next_value: value prediction for next state (after last step)
        返回:
            advantages: GAE 优势估计, length T
            returns: GAE 折扣回报, length T
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        # 从后向前计算 GAE
        for t in reversed(range(T)):
            if t == T - 1:
                # 最后一步：使用传入的 next_value（P0修复：明确获取正确的 next_value）
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                # 中间步骤：使用下一步的 value
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values)

        # 优势函数标准化（降低方差，Swift 实践）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """选择动作并返回动作、对数概率、价值、熵（P0修复版）
        
        P0修复：使用 tanh-squashed Gaussian，加入 Jacobian 修正
        - 采样 pre_tanh_action ~ N(mean, std)
        - action = tanh(pre_tanh_action)
        - log_prob 加上 tanh Jacobian 修正项
        """
        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mean, std, value = self.model(state_tensor)
            dist = Normal(mean, std)

            if deterministic:
                pre_tanh_action_tensor = mean
            else:
                pre_tanh_action_tensor = dist.sample()

            # P0修复：tanh-squashed action
            action_tensor = torch.tanh(pre_tanh_action_tensor) * self.action_max
            
            # P0修复：计算 tanh Jacobian 修正
            # log_prob(tanh(x)) = log_prob(x) - log_det_jacobian
            # log_det_jacobian = sum(log(1 - tanh(x)^2))
            log_prob_tensor = dist.log_prob(pre_tanh_action_tensor).sum(dim=-1)
            tanh_jacobian = torch.log(1 - action_tensor.pow(2) / (self.action_max ** 2) + 1e-6).sum(dim=-1)
            log_prob_tensor = log_prob_tensor - tanh_jacobian
            
            entropy = dist.entropy().sum(dim=-1)

        action = action_tensor.squeeze(0).cpu().numpy()
        log_prob = log_prob_tensor.squeeze(0).cpu().item()
        entropy_val = entropy.squeeze(0).cpu().item()
        value_val = value.squeeze(0).cpu().item()

        action = np.clip(action, -self.action_max, self.action_max)

        return action, log_prob, value_val, entropy_val

    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """选择动作（兼容评估脚本的接口）"""
        action, _, _, _ = self.get_action(observation, deterministic=deterministic)
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0,
        entropy: float = 0.0
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
            'log_prob': log_prob,
            'value': value,
            'entropy': entropy
        })

    def update(self) -> Dict[str, float]:
        """
        PPO 更新（Swift 改进版）
        包含：GAE（per-environment）、minibatch 拆分、PPO-Clip、梯度裁剪、自适应熵
        """
        if not hasattr(self, 'memory') or len(self.memory) == 0:
            return {'total_loss': 0.0, 'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0, 'entropy_coeff': 0.0}

        states = np.array([t['state'] for t in self.memory], dtype=np.float32)
        actions = np.array([t['action'] for t in self.memory], dtype=np.float32)
        rewards = np.array([t['reward'] for t in self.memory], dtype=np.float32)
        dones = np.array([t['done'] for t in self.memory], dtype=np.float32)
        old_log_probs = np.array([t['log_prob'] for t in self.memory], dtype=np.float32)
        values = np.array([t['value'] for t in self.memory], dtype=np.float32)
        next_states = np.array([t['next_state'] for t in self.memory], dtype=np.float32)  # P0修复：获取 next_state

        total_samples = len(states)
        rollout_steps = total_samples // self.num_envs

        # Reshape to (rollout_steps, num_envs) then transpose to (num_envs, rollout_steps)
        # This separates each environment's trajectory for correct GAE computation
        rewards_reshaped = rewards.reshape(rollout_steps, self.num_envs).T  # (num_envs, rollout_steps)
        values_reshaped = values.reshape(rollout_steps, self.num_envs).T
        dones_reshaped = dones.reshape(rollout_steps, self.num_envs).T
        next_states_reshaped = next_states.reshape(rollout_steps, self.num_envs).T  # P0修复：reshape next_states

        # Compute GAE per environment
        all_advantages = []
        all_returns = []
        for env_idx in range(self.num_envs):
            env_rewards = rewards_reshaped[env_idx]
            env_values = values_reshaped[env_idx]
            env_dones = dones_reshaped[env_idx]

            # P0修复：使用 next_state（最后一步的 next_state）来计算 next_value
            last_next_state_idx = (rollout_steps - 1) * self.num_envs + env_idx
            next_state_tensor = torch.tensor(next_states[last_next_state_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, _, next_value = self.model(next_state_tensor)
            next_value = next_value.cpu().item()

            env_advantages, env_returns = self.compute_gae(env_rewards, env_values, env_dones, next_value)
            all_advantages.append(env_advantages)
            all_returns.append(env_returns)

        # Flatten back to (rollout_steps * num_envs) in original order
        advantages = np.array(all_advantages).T.flatten()  # (num_envs, rollout_steps) -> (rollout_steps, num_envs) -> flat
        returns = np.array(all_returns).T.flatten()

        # 转换为张量
        states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(DEVICE)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(DEVICE)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(DEVICE)

        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        num_samples = len(states)
        num_minibatches = num_samples // self.minibatch_size

        if num_minibatches == 0:
            num_minibatches = 1
            self.minibatch_size = num_samples

        for epoch in range(self.epochs):
            indices = torch.randperm(num_samples)

            for i in range(num_minibatches):
                start = i * self.minibatch_size
                end = start + self.minibatch_size
                batch_indices = indices[start:end]

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                mean, std, value = self.model(batch_states)
                dist = Normal(mean, std)
                
                # P0修复：计算 tanh-squashed action 的 log_prob（添加 Jacobian 修正）
                pre_tanh_action = torch.atanh(torch.clamp(batch_actions / self.action_max, -0.999, 0.999))
                log_prob_pre_tanh = dist.log_prob(pre_tanh_action).sum(dim=-1)
                tanh_jacobian = torch.log(1 - (batch_actions / self.action_max).pow(2) + 1e-6).sum(dim=-1)
                current_log_probs = log_prob_pre_tanh - tanh_jacobian
                
                entropy = dist.entropy().sum(dim=-1)

                # PPO-Clip 损失
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic 损失 (MSE)
                critic_loss = nn.MSELoss()(value, batch_returns)

                # 自适应熵系数
                if self.entropy_coeff is not None:
                    entropy_coeff_val = self.entropy_coeff.get_coeff()
                else:
                    entropy_coeff_val = self.fixed_entropy_coeff

                # 总损失 = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy
                loss = actor_loss + 0.5 * critic_loss - entropy_coeff_val * entropy.mean()

                # 梯度裁剪
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()

        # 更新自适应熵系数
        if self.entropy_coeff is not None:
            self.entropy_coeff.update(torch.tensor(total_entropy / (self.epochs * num_minibatches)))

        self.memory = []

        n_updates = self.epochs * num_minibatches
        result = {
            'total_loss': total_loss / n_updates,
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'entropy_coeff': entropy_coeff_val
        }

        logger.debug(f"PPO update - total_loss: {result['total_loss']:.4f}, "
                     f"actor_loss: {result['actor_loss']:.4f}, "
                     f"critic_loss: {result['critic_loss']:.4f}, "
                     f"entropy: {result['entropy']:.4f}, "
                     f"entropy_coeff: {result['entropy_coeff']:.4f}")

        return result

    def save_model(self, path: str = "drone_ppo_model.pth") -> None:
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'action_max': self.action_max
        }, path)
        logger.info(f"模型已保存至：{path}")

    def load_model(self, path: str = "drone_ppo_model.pth") -> None:
        """加载模型"""
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"成功加载模型参数：{path}")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info(f"成功加载模型参数（旧格式）：{path}")
        except FileNotFoundError:
            logger.warning(f"未找到模型文件：{path}，从头训练")
        except Exception as e:
            logger.warning(f"加载模型失败：{e}，从头训练")
