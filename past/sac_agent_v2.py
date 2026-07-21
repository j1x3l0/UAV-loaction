"""
SAC Agent V2 — 简化版 (纯向量状态, 无深度图像)
================================================
适配当前 14D DroneEnv, 与 PPO agent 共用同一环境接口。

特性:
- Actor: 2层MLP + tanh squashing (输出动作 ∈ [-1,1])
- Critic: 双Q网络 + Target网络
- 自动熵调节 (target_entropy = -action_dim)
- Replay Buffer (off-policy)
- 接口兼容 PPO: select_action(state, deterministic=True)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Dict
from collections import deque
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Actor Network
# ============================================================
class SACActor(nn.Module):
    """策略网络: 输出高斯分布的均值和标准差"""

    def __init__(self, state_dim=14, action_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)

    def forward(self, state, deterministic=False, with_log_prob=True):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            u = dist.rsample()
            action = torch.tanh(u)
            if with_log_prob:
                # tanh squashing correction: log pi(a|s) = log N(u|mean,std) - sum(log(1-tanh^2(u)))
                log_prob = dist.log_prob(u).sum(dim=-1)
                log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _ = self.forward(state_t, deterministic=deterministic, with_log_prob=False)
        return action.squeeze(0).cpu().numpy()


# ============================================================
# Critic Network (Double Q)
# ============================================================
class SACCritic(nn.Module):
    """双Q网络: Q1(s,a), Q2(s,a)"""

    def __init__(self, state_dim=14, action_dim=3, hidden_dim=128):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.q1[-1].weight, gain=1.0)
        nn.init.orthogonal_(self.q2[-1].weight, gain=1.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# ============================================================
# SAC Agent
# ============================================================
class SAC:
    """
    Soft Actor-Critic (SAC) — 14D 纯向量版本

    关键超参数:
        lr=1e-4 (SAC需要比PPO更小的学习率)
        gamma=0.99
        tau=0.005 (target网络软更新)
        buffer_size=100000
        batch_size=256
    """

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 3,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        initial_alpha: float = 0.2,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Networks
        self.actor = SACActor(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic = SACCritic(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic_target = SACCritic(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Entropy tuning
        self.target_entropy = -action_dim  # = -3
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.log_alpha.data.fill_(np.log(initial_alpha))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.total_updates = 0
        self.current_lr = lr

        logger.info(f"SAC V2 初始化 | state_dim={state_dim}, action_dim={action_dim}")
        logger.info(f"超参数: lr={lr}, gamma={gamma}, tau={tau}, batch={batch_size}")
        logger.info(f"目标熵: {self.target_entropy}")

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """兼容 PPO 的 select_action 接口"""
        return self.actor.get_action(state, deterministic=deterministic)

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """训练时采样动作 (返回 action + log_prob)"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, log_prob = self.actor.forward(state_t, deterministic=deterministic, with_log_prob=True)
        action_np = action.squeeze(0).cpu().numpy()
        log_prob_val = log_prob.item() if log_prob is not None else 0.0
        return action_np, log_prob_val

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """单次SAC更新 (从buffer采样一个batch)"""
        if len(self.replay_buffer) < self.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'alpha_loss': 0.0, 'alpha': self.alpha}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(DEVICE)
        actions_t = torch.FloatTensor(actions).to(DEVICE)
        rewards_t = torch.FloatTensor(rewards).to(DEVICE)
        next_states_t = torch.FloatTensor(next_states).to(DEVICE)
        dones_t = torch.FloatTensor(dones).to(DEVICE)

        # --- Critic Update ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.forward(next_states_t, with_log_prob=True)
            q1_target, q2_target = self.critic_target.forward(next_states_t, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            q_target = rewards_t + self.gamma * (1 - dones_t) * q_target

        q1, q2 = self.critic.forward(states_t, actions_t)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Actor Update ---
        new_actions, log_probs = self.actor.forward(states_t, with_log_prob=True)
        q1_new, q2_new = self.critic.forward(states_t, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # --- Alpha Update ---
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Target Network Soft Update ---
        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self.total_updates += 1

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
        }

    def set_lr(self, lr):
        """学习率衰减"""
        self.current_lr = lr
        for pg in self.actor_optimizer.param_groups:
            pg['lr'] = lr
        for pg in self.critic_optimizer.param_groups:
            pg['lr'] = lr

    def save_model(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, path)
        logger.info(f"SAC模型已保存: {path}")

    def load_model(self, path: str):
        try:
            ckpt = torch.load(path, map_location=DEVICE)
            self.actor.load_state_dict(ckpt['actor'])
            self.critic.load_state_dict(ckpt['critic'])
            self.critic_target.load_state_dict(ckpt['critic_target'])
            self.log_alpha.data.fill_(ckpt['log_alpha'])
            logger.info(f"SAC模型已加载: {path}")
        except Exception as e:
            logger.warning(f"加载失败: {e}")
