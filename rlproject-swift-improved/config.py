import argparse
from datetime import datetime
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Swift 改进版训练参数配置类"""

    def __init__(self):
        self.algorithm = 'ppo'
        self.max_episodes = 3000
        self.rollout_steps = 2048
        self.num_envs = 8
        self.lr = 3e-4
        self.hidden_dim = 128
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.epochs = 10
        self.minibatch_size = 64
        self.use_adaptive_entropy = True
        self.max_grad_norm = 0.5
        self.value_coeff = 0.5

        self.state_dim = 14
        self.action_dim = 3
        self.action_max = 1.0

        self.print_interval = 10
        self.eval_interval = 100
        self.eval_episodes = 50

        self.save_threshold = 100

        # 噪声配置（None 表示无噪声）
        self.noise_pattern: str = None   # 'pos', 'vel', 'target', 'obs', 'full' 或 None
        self.noise_sigma: float = 0.0    # 噪声标准差

    def from_args(self, args) -> 'TrainingConfig':
        """从命令行参数更新配置"""
        if hasattr(args, 'episodes'):
            self.max_episodes = args.episodes
        if hasattr(args, 'lr'):
            self.lr = args.lr
        if hasattr(args, 'num_envs'):
            self.num_envs = args.num_envs
        if hasattr(args, 'rollout_steps'):
            self.rollout_steps = args.rollout_steps
        # 噪声配置
        if hasattr(args, 'noise_pattern'):
            self.noise_pattern = args.noise_pattern
        if hasattr(args, 'noise_sigma'):
            self.noise_sigma = args.noise_sigma
        return self

    def should_save(self) -> bool:
        """判断是否应该保存模型和日志"""
        return self.max_episodes >= self.save_threshold

    def get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_save_paths(self) -> Dict[str, str]:
        """获取保存路径配置"""
        if not self.should_save():
            return {
                'log_dir': None,
                'model_path': None
            }

        timestamp = self.get_timestamp()
        noise_tag = f"_{self.noise_pattern}_s{self.noise_sigma}" if self.noise_pattern else ""
        log_dir = f"logs/ppo_swift_{self.max_episodes}ep{noise_tag}_{timestamp}"
        model_path = f"saved_models/ppo_swift_{self.max_episodes}ep{noise_tag}_{timestamp}"

        return {
            'log_dir': log_dir,
            'model_path': model_path
        }

    def get_environment_config(self) -> Dict[str, Any]:
        """获取环境配置"""
        return {}

    def get_algorithm_config(self) -> Dict[str, Any]:
        """获取算法配置"""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'action_max': self.action_max,
            'lr': self.lr,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_eps': self.clip_eps,
            'epochs': self.epochs,
            'minibatch_size': self.minibatch_size,
            'hidden_dim': self.hidden_dim,
            'use_adaptive_entropy': self.use_adaptive_entropy,
            'max_episodes': self.max_episodes,
            'rollout_steps': self.rollout_steps,
            'num_envs': self.num_envs,
            'print_interval': self.print_interval,
            'eval_interval': self.eval_interval,
            'eval_episodes': self.eval_episodes,
            # 噪声配置
            'noise_pattern': self.noise_pattern,
            'noise_sigma': self.noise_sigma,
        }

    def __str__(self) -> str:
        return f"""
训练配置（Swift 改进版）:
- 算法: {self.algorithm}
- 训练轮数: {self.max_episodes}
- 并行环境数: {self.num_envs}
- 每轮采样步数: {self.rollout_steps}
- 学习率: {self.lr}
- 隐藏层维度: {self.hidden_dim}
- 折扣因子: {self.gamma}
- GAE lambda: {self.gae_lambda}
- PPO裁剪参数: {self.clip_eps}
- 更新轮数: {self.epochs}
- minibatch大小: {self.minibatch_size}
- 自适应熵系数: {self.use_adaptive_entropy}
- 梯度裁剪: {self.max_grad_norm}
- 价值损失系数: {self.value_coeff}
- 状态维度: {self.state_dim}
- 动作维度: {self.action_dim}
"""


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='无人机强化学习训练（Swift改进版）')
    parser.add_argument('--episodes', type=int, default=3000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--num-envs', type=int, default=8, help='并行环境数')
    parser.add_argument('--rollout-steps', type=int, default=2048, help='每轮采样步数')

    # 噪声参数
    parser.add_argument('--noise-pattern', type=str, default=None,
                        choices=['pos', 'vel', 'target', 'obs', 'full'],
                        help='噪声模式 (pos/vel/target/obs/full)，默认无噪声')
    parser.add_argument('--noise-sigma', type=float, default=0.5,
                        help='噪声标准差 (默认 0.5)')

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """从参数创建配置对象"""
    config = TrainingConfig()
    config.from_args(args)
    return config


def create_default_config(algorithm: str = 'ppo', episodes: int = 3000) -> TrainingConfig:
    """创建默认配置"""
    config = TrainingConfig()
    config.algorithm = algorithm
    config.max_episodes = episodes
    return config
