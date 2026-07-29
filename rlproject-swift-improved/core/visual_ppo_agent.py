"""
Visual PPO Agent — CNN编码器 + PPO 视觉导航智能体 v2

架构位置: core/ (Foundation层)
WHY 这个设计:
  - 端到端可训练CNN (vs GRaD-Nav的冻结SqueezeNet) → 退化GS下可能更鲁棒
  - CNN输出128D + 向量6D = 134D → 共享MLP → Actor/Critic (Swift实践)
  - 复用v1的PPO核心逻辑 (GAE/Clip/自适应熵)
数据流: {depth:(64,64,1), vec:(6,)} → CNN→128D + vec→6D → MLP[128,128] → action(3D)+value(1D)
边界: 不负责环境交互、不负责训练循环、不负责日志
风险: CNN可能过拟合mock渲染器的纹理 → 真3DGS后需重新验证
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Visual PPO Device: {DEVICE}")


# ─── CNN 视觉编码器 ─────────────────────────────────────────────
# WHY 3层Conv stride=2: 64→32→16→8 是标准CNN下采样范式
# 128D输出 vs GRaD-Nav的16D: 我们的信息瓶颈更宽，端到端训练可充分利用

class VisualEncoder(nn.Module):
    """深度图 → 128D 视觉特征"""
    def __init__(self, in_channels=1, feature_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),                                      # 64→32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),                                      # 32→16
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),                                      # 16→8
            nn.Flatten(),
        )
        # 计算展平维度: 64ch × 8 × 8 = 4096
        self.fc = nn.Linear(64 * 8 * 8, feature_dim)

    def forward(self, depth):
        """depth: (B, 1, 64, 64) → (B, 128)"""
        x = self.conv(depth)
        return self.fc(x)


# ── Visual Actor-Critic ─────────────────────────────────────────
# WHY 共享特征提取器: Actor/Critic共享CNN+MLP第一层, 减少参数, 加速训练(Swift实践)
# tanh(mean)将动作限制在(-1,1), 匹配归一化动作空间

class VisualActorCritic(nn.Module):
    """CNN编码器 + 共享MLP → Actor(3D动作) + Critic(1D价值)"""

    def __init__(self, vec_dim=6, visual_feature_dim=128,
                 hidden_dim=128, action_dim=3):
        super().__init__()
        self.visual_encoder = VisualEncoder(in_channels=1,
                                             feature_dim=visual_feature_dim)
        combined_dim = visual_feature_dim + vec_dim  # 128 + 6 = 134

        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 可学习 log_std
        # WHY init=-0.5 (std≈0.6): 原始std=1.0在视觉观测下entropy(4.26)太高,
        # 100ep内entropy纹丝不动。从std=0.6开始,entropy≈3.5,降低随机碰撞概率。
        self.log_std = nn.Parameter(torch.ones(action_dim) * (-0.5))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Actor输出层用小gain (Swift实践)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, depth, vec):
        """
        depth: (B, 1, 64, 64)
        vec:   (B, 6)
        → mean: (B, 3), std: (B, 3), value: (B,)
        """
        vis = self.visual_encoder(depth)           # (B, 128)
        combined = torch.cat([vis, vec], dim=-1)   # (B, 134)
        shared = self.shared(combined)              # (B, 128)

        mean = torch.tanh(self.actor(shared))       # (B, 3) ∈ (-1,1)
        std = torch.clamp(self.log_std.exp(), 1e-3, 1.0)
        value = self.critic(shared).squeeze(-1)     # (B,)

        return mean, std, value


# ── 自适应熵系数 ─────────────────────────────────────────────────
# P0修复版：目标改为正值，加上下界

class AdaptiveEntropyCoeff:
    """自适应熵系数（P0修复版）
    
    P0修复：
    - 目标改为正值 (2.0-3.0)，不再是负的 -action_dim
    - 加 log_alpha 上下界 [-9.2, -1.6]（对应 alpha ≈ 1e-4 ~ 0.2）
    """
    
    def __init__(self, initial_coeff=0.01, target_entropy=None, lr=3e-4, action_dim=3):
        # P0修复：目标改为正熵值
        if target_entropy is None:
            self.target_entropy = 2.5
        else:
            self.target_entropy = max(0.5, target_entropy)
        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.log_alpha.data.fill_(np.log(initial_coeff))
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # P0修复：上下界
        self.log_alpha_min = np.log(1e-4)
        self.log_alpha_max = np.log(0.2)
        self.update_count = 0

    def get_coeff(self):
        return torch.clamp(self.log_alpha, self.log_alpha_min, self.log_alpha_max).exp().item()

    def update(self, entropy):
        if self.target_entropy is None:
            return
        alpha = torch.clamp(self.log_alpha, self.log_alpha_min, self.log_alpha_max).exp()
        loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # P0修复：手动裁剪
        with torch.no_grad():
            self.log_alpha.data.clamp_(self.log_alpha_min, self.log_alpha_max)
        
        self.update_count += 1


# ── Visual PPO ──────────────────────────────────────────────────
# WHY 继承v1 PPO的GAE/Clip逻辑, 仅修改网络和观测处理

class VisualPPO:
    """视觉PPO智能体"""

    def __init__(self, vec_dim=6, action_dim=3, action_max=1.0,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 epochs=10, minibatch_size=64, hidden_dim=128,
                 use_adaptive_entropy=True, num_envs=8,
                 reward_scale=0.1):
        """
        Args:
            reward_scale: 奖励缩放因子。原始reward量级[-10, 100],
                          256步returns ~[-2500, 100]。Critic初始输出~0-1,
                          不做缩放会导致MSE loss ~10⁶, 梯度淹没actor信号。
                          缩放后returns ~[-25, 1], 与critic初始化匹配。
        """
        self.action_dim = action_dim
        self.action_max = action_max
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.num_envs = num_envs
        self.reward_scale = reward_scale

        self.model = VisualActorCritic(
            vec_dim=vec_dim, hidden_dim=hidden_dim, action_dim=action_dim
        ).to(DEVICE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if use_adaptive_entropy:
            # P0修复：改为正值 target_entropy，加上下界防止发散
            self.entropy_coeff = AdaptiveEntropyCoeff(
                initial_coeff=0.1, 
                target_entropy=2.5,  # P0修复：改为正值
                lr=lr * 10,
                action_dim=action_dim)
        else:
            self.entropy_coeff = None
            self.fixed_entropy_coeff = 0.01

        self.current_lr = lr
        logger.info(f"VisualPPO init | CNN+MLP | action={action_dim}")

    def set_lr(self, lr):
        self.current_lr = lr
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def get_actions_batch(self, observations: List[Dict]
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """批量推理: 多个环境的观测 → 批处理 actions/log_probs/values/entropies

        WHY: train_visual 中用单次 CNN forward 处理所有环境,
             避免逐环境推理的串行开销
        数据流: [obs1, obs2, ...] → stack → CNN → dist → sample → actions
        """
        depths = []
        vecs = []
        for obs in observations:
            depth = torch.tensor(obs['depth'], dtype=torch.float32
                                ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1,1,64,64)
            vec = torch.tensor(obs['vec'], dtype=torch.float32
                              ).unsqueeze(0).to(DEVICE)
            depths.append(depth)
            vecs.append(vec)

        depth_batch = torch.cat(depths, dim=0)  # (B, 1, 64, 64)
        vec_batch = torch.cat(vecs, dim=0)       # (B, 6)

        with torch.no_grad():
            mean, std, value = self.model(depth_batch, vec_batch)
            dist = Normal(mean, std)

            pre_tanh_action = dist.sample()
            action_tensor = torch.tanh(pre_tanh_action) * self.action_max

            log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
            tanh_jacobian = torch.log(1 - (action_tensor / self.action_max).pow(2) + 1e-6).sum(dim=-1)
            log_prob = log_prob - tanh_jacobian

            entropy = dist.entropy().sum(dim=-1)

        return (action_tensor.cpu().numpy(),
                log_prob.cpu().numpy(),
                value.cpu().numpy(),
                entropy.cpu().numpy())

    def get_action(self, observation: Dict, deterministic=False
                   ) -> Tuple[np.ndarray, float, float, float]:
        """observation = {'depth': (64,64,1), 'vec': (6,)}
        
        P0修复：使用 tanh-squashed Gaussian
        """
        depth = torch.tensor(observation['depth'], dtype=torch.float32
                            ).unsqueeze(0).to(DEVICE)  # (1,64,64,1) → (1,1,64,64)
        depth = depth.permute(0, 3, 1, 2)              # NHWC → NCHW
        vec = torch.tensor(observation['vec'], dtype=torch.float32
                          ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mean, std, value = self.model(depth, vec)
            dist = Normal(mean, std)
            
            # P0修复：tanh-squashed action
            pre_tanh_action = mean if deterministic else dist.sample()
            action_tensor = torch.tanh(pre_tanh_action) * self.action_max
            
            # P0修复：计算 tanh Jacobian 修正
            log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
            tanh_jacobian = torch.log(1 - (action_tensor / self.action_max).pow(2) + 1e-6).sum(dim=-1)
            log_prob = log_prob - tanh_jacobian
            
            entropy = dist.entropy().sum(dim=-1)

        return (action_tensor.squeeze(0).cpu().numpy(),
                log_prob.squeeze(0).cpu().item(),
                value.squeeze(0).cpu().item(),
                entropy.squeeze(0).cpu().item())

    def select_action(self, observation: Dict, deterministic=True
                      ) -> np.ndarray:
        action, _, _, _ = self.get_action(observation, deterministic)
        return action

    def store_transition(self, state, action, reward, next_state,
                         done, log_prob=0.0, value=0.0, entropy=0.0):
        if not hasattr(self, 'memory'):
            self.memory = []
        self.memory.append({'state': state, 'action': action,
                           'reward': reward * self.reward_scale,
                           'next_state': next_state,
                           'done': done, 'log_prob': log_prob,
                           'value': value, 'entropy': entropy})

    def compute_gae(self, rewards, values, dones, next_value):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = (rewards[t] + self.gamma * next_val * next_non_terminal
                     - values[t])
            last_gae = (delta + self.gamma * self.gae_lambda
                        * next_non_terminal * last_gae)
            advantages[t] = last_gae
        returns = advantages + np.array(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self) -> Dict[str, float]:
        if not hasattr(self, 'memory') or len(self.memory) == 0:
            return {'total_loss': 0.0, 'actor_loss': 0.0,
                    'critic_loss': 0.0, 'entropy': 0.0, 'entropy_coeff': 0.0}

        # 汇总 rollout 数据
        states_depth = np.stack([t['state']['depth'] for t in self.memory])
        states_vec = np.stack([t['state']['vec'] for t in self.memory])
        actions = np.array([t['action'] for t in self.memory], dtype=np.float32)
        rewards = np.array([t['reward'] for t in self.memory], dtype=np.float32)
        dones = np.array([t['done'] for t in self.memory], dtype=np.float32)
        old_log_probs = np.array([t['log_prob'] for t in self.memory],
                                  dtype=np.float32)
        values = np.array([t['value'] for t in self.memory], dtype=np.float32)
        next_states_depth = np.stack([t['next_state']['depth'] for t in self.memory])  # P0修复
        next_states_vec = np.stack([t['next_state']['vec'] for t in self.memory])      # P0修复

        total_samples = len(states_depth)
        rollout_steps = total_samples // self.num_envs

        # Per-environment GAE
        rewards_2d = rewards.reshape(rollout_steps, self.num_envs).T
        values_2d = values.reshape(rollout_steps, self.num_envs).T
        dones_2d = dones.reshape(rollout_steps, self.num_envs).T
        all_adv, all_ret = [], []
        for env_idx in range(self.num_envs):
            # P0修复：使用 next_state 而不是 state
            lidx = (rollout_steps - 1) * self.num_envs + env_idx
            d_t = torch.tensor(next_states_depth[lidx], dtype=torch.float32
                              ).unsqueeze(0).to(DEVICE)
            d_t = d_t.permute(0, 3, 1, 2)
            v_t = torch.tensor(next_states_vec[lidx], dtype=torch.float32
                              ).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, _, nv = self.model(d_t, v_t)
            adv, ret = self.compute_gae(
                rewards_2d[env_idx], values_2d[env_idx],
                dones_2d[env_idx], nv.cpu().item())
            all_adv.append(adv); all_ret.append(ret)
        advantages = np.array(all_adv).T.flatten()
        returns = np.array(all_ret).T.flatten()

        # 转张量
        s_d = torch.tensor(states_depth, dtype=torch.float32).to(DEVICE)
        s_d = s_d.permute(0, 3, 1, 2)
        s_v = torch.tensor(states_vec, dtype=torch.float32).to(DEVICE)
        a_t = torch.tensor(actions).to(DEVICE)
        olp = torch.tensor(old_log_probs).to(DEVICE)
        adv = torch.tensor(advantages).to(DEVICE)
        ret = torch.tensor(returns).to(DEVICE)

        total_loss = total_aloss = total_closs = total_ent = 0.0
        n_samples = len(states_depth)
        n_mb = max(1, n_samples // self.minibatch_size)
        if n_mb == 1:
            self.minibatch_size = n_samples

        for _ in range(self.epochs):
            indices = torch.randperm(n_samples)
            for i in range(n_mb):
                start = i * self.minibatch_size
                end = start + self.minibatch_size
                bi = indices[start:end]

                mean, std, value = self.model(s_d[bi], s_v[bi])
                dist = Normal(mean, std)
                
                # P0修复：tanh-squashed action 的 log_prob（加 Jacobian 修正）
                pre_tanh_action = torch.atanh(torch.clamp(a_t[bi] / self.action_max, -0.999, 0.999))
                log_prob_pre_tanh = dist.log_prob(pre_tanh_action).sum(dim=-1)
                tanh_jacobian = torch.log(1 - (a_t[bi] / self.action_max).pow(2) + 1e-6).sum(dim=-1)
                clp = log_prob_pre_tanh - tanh_jacobian
                
                ent = dist.entropy().sum(dim=-1)

                ratio = torch.exp(clp - olp[bi])
                s1 = ratio * adv[bi]
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[bi]
                actor_loss = -torch.min(s1, s2).mean()
                critic_loss = nn.MSELoss()(value, ret[bi])

                ec = (self.entropy_coeff.get_coeff() if self.entropy_coeff
                      else self.fixed_entropy_coeff)
                loss = actor_loss + 0.5 * critic_loss - ec * ent.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                total_aloss += actor_loss.item()
                total_closs += critic_loss.item()
                total_ent += ent.mean().item()

        if self.entropy_coeff:
            self.entropy_coeff.update(
                torch.tensor(total_ent / (self.epochs * n_mb)))

        self.memory = []
        n_upd = self.epochs * n_mb
        return {'total_loss': total_loss / n_upd,
                'actor_loss': total_aloss / n_upd,
                'critic_loss': total_closs / n_upd,
                'entropy': total_ent / n_upd,
                'entropy_coeff': ec}

    def save_model(self, path="visual_ppo_model.pth"):
        torch.save({'model_state_dict': self.model.state_dict(),
                     'action_dim': self.action_dim}, path)
        logger.info(f"Model saved: {path}")

    def load_model(self, path="visual_ppo_model.pth"):
        try:
            ckpt = torch.load(path, map_location=DEVICE)
            self.model.load_state_dict(ckpt['model_state_dict'])
            logger.info(f"Model loaded: {path}")
        except Exception as e:
            logger.warning(f"Load failed: {e}, training from scratch")


# ── 自检 ──
if __name__ == "__main__":
    ppo = VisualPPO(vec_dim=6, action_dim=3, num_envs=1)
    # 模拟观测
    fake_depth = np.random.rand(64, 64, 1).astype(np.float32) * 10
    fake_vec = np.array([1.0, 0.5, -0.3, 5.0, 2.0, 1.0], dtype=np.float32)
    obs = {'depth': fake_depth, 'vec': fake_vec}

    action, lp, val, ent = ppo.get_action(obs)
    print(f"action: {action} | log_prob: {lp:.3f} | value: {val:.3f} | entropy: {ent:.3f}")

    # 模拟一次 update
    for _ in range(8):
        ppo.store_transition(obs, action, 1.0, obs, False, lp, val, ent)
    result = ppo.update()
    print(f"update - total_loss: {result['total_loss']:.4f}, "
          f"actor_loss: {result['actor_loss']:.4f}")
    print("VisualPPO self-check passed")
