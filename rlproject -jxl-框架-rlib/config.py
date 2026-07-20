import argparse
from datetime import datetime
from typing import Dict, Any


class TrainingConfig:
    """训练参数配置类"""
    
    def __init__(self):
        self.algorithm = 'ppo'
        self.max_episodes = 2000
        self.batch_size = 10
        self.lr = 3e-4
        self.depth_image_size = 16
        self.use_depth_sensor = True
        self.hidden_dim = 256
        self.gamma = 0.99
        self.print_interval = 100
        self.reward_version = 'v1'  # 新增奖励函数版本配置
        
        # PPO specific
        self.clip_eps = 0.2
        self.epochs = 10
        
        # SAC specific
        self.buffer_size = 100000
        self.sac_batch_size = 256
        self.updates_per_step = 1
        self.warmup_steps = 1000
        
        # 保存设置
        self.save_threshold = 500
        
    def from_args(self, args) -> 'TrainingConfig':
        """从命令行参数更新配置"""
        self.algorithm = args.algorithm
        self.max_episodes = args.episodes
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.depth_image_size = args.depth_size
        self.use_depth_sensor = not args.no_sensor
        if hasattr(args, 'reward_version'):
            self.reward_version = args.reward_version
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
        log_dir = f"logs/{self.algorithm}_{self.max_episodes}ep_{timestamp}"
        model_path = f"saved_models/{self.algorithm}_{self.max_episodes}ep_{timestamp}"
        
        return {
            'log_dir': log_dir,
            'model_path': model_path
        }
    
    def get_environment_config(self) -> Dict[str, Any]:
        """获取环境配置"""
        return {
            'use_depth_sensor': self.use_depth_sensor,
            'depth_image_size': self.depth_image_size,
            'reward_version': self.reward_version  # 添加奖励函数版本配置
        }
    
    def get_algorithm_config(self) -> Dict[str, Any]:
        """获取算法配置"""
        base_config = {
            'vec_state_dim': None,  # 将在环境初始化后设置
            'action_dim': 3,
            'action_max': None,  # 将在环境初始化后设置
            'lr': self.lr,
            'gamma': self.gamma,
            'hidden_dim': self.hidden_dim,
            'use_depth_sensor': self.use_depth_sensor,
            'depth_image_size': self.depth_image_size,
            'max_episodes': self.max_episodes,
            'batch_size': self.batch_size,
            'print_interval': self.print_interval
        }
        
        if self.algorithm == 'ppo':
            base_config.update({
                'clip_eps': self.clip_eps,
                'epochs': self.epochs
            })
        elif self.algorithm == 'sac':
            base_config.update({
                'buffer_size': self.buffer_size,
                'sac_batch_size': self.sac_batch_size,
                'updates_per_step': self.updates_per_step,
                'warmup_steps': self.warmup_steps
            })
        
        return base_config
    
    def __str__(self) -> str:
        return f"""
训练配置:
- 算法: {self.algorithm}
- 训练轮数: {self.max_episodes}
- 批次大小: {self.batch_size}
- 学习率: {self.lr}
- 深度图像尺寸: {self.depth_image_size}x{self.depth_image_size}
- 使用深度传感器: {self.use_depth_sensor}
- 隐藏层维度: {self.hidden_dim}
- 折扣因子: {self.gamma}
- 保存阈值: {self.save_threshold}
- 是否保存: {self.should_save()}
"""


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='无人机强化学习训练')
    parser.add_argument('--algorithm', type=str, default='ppo', 
                       choices=['ppo', 'sac', 'compare'],
                       help='选择算法: ppo, sac, 或 compare(对比实验)')
    parser.add_argument('--episodes', type=int, default=2000, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=10, help='每批步数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--depth-size', type=int, default=16, help='深度图像尺寸')
    parser.add_argument('--no-sensor', action='store_true', help='禁用深度传感器')
    parser.add_argument('--reward-version', type=str, default='v1', 
                       choices=['v1', 'v2'], help='奖励函数版本: v1或v2')
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """从参数创建配置对象"""
    config = TrainingConfig()
    config.from_args(args)
    return config


def create_default_config(algorithm: str = 'ppo', episodes: int = 2000) -> TrainingConfig:
    """创建默认配置"""
    config = TrainingConfig()
    config.algorithm = algorithm
    config.max_episodes = episodes
    return config