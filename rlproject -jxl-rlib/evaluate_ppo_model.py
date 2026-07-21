import numpy as np
import torch
import os
import sys
import json
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义环境和PPO智能体
from drone_env import DroneEnv
from ppo_agent import PPO

def evaluate_ppo_model(model_path: str, num_episodes: int = 100, seed: int = None, 
                       use_depth_sensor: bool = True, depth_image_size: int = 16) -> dict:
    """
    批量评估自定义PPO模型的性能（针对训练好的模型）
    :param model_path: 模型文件路径
    :param num_episodes: 评估的episode数量
    :param seed: 随机种子
    :param use_depth_sensor: 是否使用深度传感器
    :param depth_image_size: 深度图像尺寸
    :return: 评估结果字典
    """
    # 初始化环境
    env = DroneEnv(config={
        'use_depth_sensor': use_depth_sensor,
        'depth_image_size': depth_image_size,
        'reward_version': 'v2'
    })
    
    # 获取状态空间维度
    if use_depth_sensor:
        vec_state_dim = env.observation_space['vector'].shape[0]
    else:
        vec_state_dim = env.observation_space.shape[0]
    
    action_dim = 3  # 3维动作（thrust_x, thrust_y, thrust_z）
    action_max = float(env.action_space.high[0])
    
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(env, 'seed'):
            env.seed(seed)
    
    # 初始化PPO智能体
    ppo = PPO(
        vec_state_dim=vec_state_dim,
        action_dim=action_dim,
        action_max=action_max,
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,
        epochs=10,
        hidden_dim=256,
        use_depth_sensor=use_depth_sensor,
        depth_image_size=depth_image_size
    )
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    ppo.load_model(model_path)
    
    # 初始化统计变量
    metrics = {
        "total_episodes": num_episodes,
        "success_count": 0,
        "collision_count": 0,
        "timeout_count": 0,
        "total_rewards": [],
        "episode_lengths": [],
        "success_rewards": [],
        "collision_rewards": [],
        "timeout_rewards": [],
        "final_distances": [],
        "time_to_target": []  # 到达目标的时间（步数）
    }
    
    print(f"\n=== 开始批量评估PPO模型 ({num_episodes} episodes) ===")
    print(f"使用深度传感器: {use_depth_sensor}, 深度图像尺寸: {depth_image_size}")
    print("=" * 60)
    
    # 开始评估循环
    for episode in range(num_episodes):
        # 显示进度
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"进度: {episode + 1}/{num_episodes} episodes")
        
        # 重置环境
        state, info = env.reset()
        target_pos = info["target_pos"]
        
        episode_reward = 0.0
        episode_steps = 0
        done = False
        truncated = False
        
        # 运行episode
        while not done and not truncated:
            # 使用模型预测动作（不探索）
            action, _ = ppo.get_action(state, deterministic=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 累积奖励和步数
            episode_reward += reward
            episode_steps += 1
            
            # 更新状态
            state = next_state
            
            done = terminated or truncated
        
        # 计算最终距离目标的距离
        if use_depth_sensor:
            final_pos = state['vector'][:3]
        else:
            final_pos = state[:3]
        final_distance = np.linalg.norm(final_pos - target_pos)
        
        # 更新统计信息
        metrics["total_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(episode_steps)
        metrics["final_distances"].append(final_distance)
        
        # 判断结果类型
        if info.get("reached_target", False):
            metrics["success_count"] += 1
            metrics["success_rewards"].append(episode_reward)
            metrics["time_to_target"].append(episode_steps)
        elif info.get("collision", False):
            metrics["collision_count"] += 1
            metrics["collision_rewards"].append(episode_reward)
        else:
            metrics["timeout_count"] += 1
            metrics["timeout_rewards"].append(episode_reward)
    
    # 将NumPy类型转换为Python原生类型
    def to_python_type(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        elif isinstance(value, (np.bool_, np.bool_)):
            return bool(value)
        return value
    
    # 计算统计指标
    results = {
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "num_episodes": num_episodes,
        "seed": seed,
        "algorithm": "PPO",
        "framework": "PyTorch",
        "use_depth_sensor": use_depth_sensor,
        "depth_image_size": depth_image_size,
        
        # 核心指标
        "success_rate": to_python_type(metrics["success_count"] / num_episodes),
        "collision_rate": to_python_type(metrics["collision_count"] / num_episodes),
        "timeout_rate": to_python_type(metrics["timeout_count"] / num_episodes),
        
        # 奖励统计
        "average_total_reward": to_python_type(np.mean(metrics["total_rewards"])),
        "std_total_reward": to_python_type(np.std(metrics["total_rewards"])),
        "max_total_reward": to_python_type(np.max(metrics["total_rewards"])),
        "min_total_reward": to_python_type(np.min(metrics["total_rewards"])),
        
        # Episode长度统计
        "average_episode_length": to_python_type(np.mean(metrics["episode_lengths"])),
        "std_episode_length": to_python_type(np.std(metrics["episode_lengths"])),
        "max_episode_length": to_python_type(np.max(metrics["episode_lengths"])),
        "min_episode_length": to_python_type(np.min(metrics["episode_lengths"])),
        
        # 目标相关统计
        "average_final_distance": to_python_type(np.mean(metrics["final_distances"])),
        "average_time_to_target": to_python_type(np.mean(metrics["time_to_target"]) if metrics["time_to_target"] else 0),
        
        # 分类统计
        "success_reward_stats": {
            "mean": to_python_type(np.mean(metrics["success_rewards"]) if metrics["success_rewards"] else 0),
            "count": len(metrics["success_rewards"])
        },
        "collision_reward_stats": {
            "mean": to_python_type(np.mean(metrics["collision_rewards"]) if metrics["collision_rewards"] else 0),
            "count": len(metrics["collision_rewards"])
        },
        "timeout_reward_stats": {
            "mean": to_python_type(np.mean(metrics["timeout_rewards"]) if metrics["timeout_rewards"] else 0),
            "count": len(metrics["timeout_rewards"])
        }
    }
    
    return results

def print_evaluation_results(results: dict):
    """
    打印评估结果
    :param results: 评估结果字典
    """
    print("\n" + "=" * 60)
    print("=== PPO模型批量评估结果 ===")
    print("=" * 60)
    
    print(f"评估时间: {results['evaluation_time']}")
    print(f"模型路径: {results['model_path']}")
    print(f"评估Episode数: {results['num_episodes']}")
    print(f"算法: {results['algorithm']}")
    print(f"框架: {results['framework']}")
    print(f"使用深度传感器: {results['use_depth_sensor']}")
    print(f"深度图像尺寸: {results['depth_image_size']}")
    if results['seed'] is not None:
        print(f"随机种子: {results['seed']}")
    
    print("\n" + "=" * 40)
    print("核心指标:")
    print(f"成功率: {results['success_rate']:.2%} ({results['success_reward_stats']['count']}/{results['num_episodes']})")
    print(f"碰撞率: {results['collision_rate']:.2%} ({results['collision_reward_stats']['count']}/{results['num_episodes']})")
    print(f"超时率: {results['timeout_rate']:.2%} ({results['timeout_reward_stats']['count']}/{results['num_episodes']})")
    
    print("\n" + "=" * 40)
    print("奖励统计:")
    print(f"平均累计奖励: {results['average_total_reward']:.2f}")
    print(f"奖励标准差: {results['std_total_reward']:.2f}")
    print(f"最大累计奖励: {results['max_total_reward']:.2f}")
    print(f"最小累计奖励: {results['min_total_reward']:.2f}")
    
    print("\n" + "=" * 40)
    print("Episode长度统计:")
    print(f"平均Episode长度: {results['average_episode_length']:.2f}")
    print(f"长度标准差: {results['std_episode_length']:.2f}")
    print(f"最大Episode长度: {results['max_episode_length']}")
    print(f"最小Episode长度: {results['min_episode_length']}")
    
    print("\n" + "=" * 40)
    print("目标相关统计:")
    print(f"平均最终距离目标: {results['average_final_distance']:.2f}")
    if results['success_reward_stats']['count'] > 0:
        print(f"平均到达目标时间: {results['average_time_to_target']:.2f} 步")
    
    print("\n" + "=" * 60)

def save_evaluation_results(results: dict, output_path: str):
    """
    保存评估结果到JSON文件
    :param results: 评估结果字典
    :param output_path: 输出文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果已保存到: {output_path}")

def main():
    """
    主函数
    """
    # 配置参数
    model_path = "saved_models/ppo_5000ep_20260127_163348"
    num_episodes = 50
    seed = 42
    use_depth_sensor = True
    depth_image_size = 16
    
    # 生成输出文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./evaluation_results_ppo_{timestamp}.json"
    
    print("=== PPO模型评估脚本 ===")
    print(f"评估模型: {model_path}")
    print(f"评估Episode数: {num_episodes}")
    print(f"使用深度传感器: {use_depth_sensor}")
    print(f"深度图像尺寸: {depth_image_size}")
    
    # 执行评估
    results = evaluate_ppo_model(
        model_path=model_path,
        num_episodes=num_episodes,
        seed=seed,
        use_depth_sensor=use_depth_sensor,
        depth_image_size=depth_image_size
    )
    
    # 打印结果
    print_evaluation_results(results)
    
    # 保存结果
    save_evaluation_results(results, output_path)

if __name__ == "__main__":
    """
    PPO模型评估脚本入口
    
    使用方法:
    python evaluate_ppo_model.py
    """
    main()