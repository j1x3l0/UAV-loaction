"""
基线性能评估脚本
对比PPO和SAC在V1/V2奖励函数下的表现
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone_env import DroneEnv
from ppo_agent import PPO
from sac_agent import SAC


def evaluate_agent(agent, env, num_episodes: int = 10, max_steps: int = 500) -> Dict:
    """评估智能体性能"""
    episode_rewards = []
    episode_steps = []
    success_count = 0
    collision_count = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, deterministic=True)
            else:
                action = agent.act(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        if info.get('reached_target', False):
            success_count += 1
        if info.get('collision', False):
            collision_count += 1
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_steps': np.mean(episode_steps),
        'success_rate': success_count / num_episodes,
        'collision_rate': collision_count / num_episodes,
        'all_rewards': episode_rewards,
        'all_steps': episode_steps
    }


def run_baseline_experiment(
    algorithm: str,
    reward_version: str,
    num_episodes: int = 500,
    eval_episodes: int = 10,
    use_depth_sensor: bool = True,
    depth_image_size: int = 32
) -> Dict:
    """运行基线实验"""
    print(f"\n{'='*80}")
    print(f"🚀 开始实验: {algorithm.upper()} + Reward-{reward_version.upper()}")
    print(f"{'='*80}")
    
    config = {
        'use_depth_sensor': use_depth_sensor,
        'depth_image_size': depth_image_size,
        'reward_version': reward_version
    }
    env = DroneEnv(config=config)
    
    if algorithm == 'ppo':
        agent = PPO(
            vec_state_dim=10,
            action_dim=3,
            action_max=1.0,
            lr=3e-4,
            gamma=0.99,
            clip_eps=0.2,
            epochs=10,
            hidden_dim=256,
            use_depth_sensor=use_depth_sensor,
            depth_image_size=depth_image_size
        )
    else:
        agent = SAC(
            vec_state_dim=10,
            action_dim=3,
            action_max=1.0,
            lr=3e-4,
            gamma=0.99,
            hidden_dim=256,
            buffer_size=100000,
            batch_size=64,
            use_depth_sensor=use_depth_sensor,
            depth_image_size=depth_image_size
        )
    
    episode_rewards = []
    episode_steps = []
    best_avg_reward = float('-inf')
    moving_avg = []
    window_size = 20
    
    print(f"\n📊 训练进度 (共{num_episodes}轮)")
    print("-" * 60)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(1000):
            if algorithm == 'ppo':
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, terminated)
            else:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, terminated)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        if algorithm == 'ppo':
            agent.update()
        else:
            for _ in range(steps // 64 + 1):
                agent.update()
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        if len(episode_rewards) >= window_size:
            avg = np.mean(episode_rewards[-window_size:])
            moving_avg.append(avg)
            if avg > best_avg_reward:
                best_avg_reward = avg
        else:
            moving_avg.append(np.mean(episode_rewards))
        
        if (episode + 1) % 50 == 0 or episode == 0:
            recent_avg = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            print(f"  轮 {episode+1:4d}/{num_episodes} | 奖励: {total_reward:8.2f} | "
                  f"50轮均值: {recent_avg:8.2f} | 步数: {steps:4d}")
    
    print(f"\n🎯 开始评估 (共{eval_episodes}轮)...")
    eval_results = evaluate_agent(agent, env, num_episodes=eval_episodes)
    
    print(f"\n📋 训练结果汇总:")
    print(f"  • 总训练轮数: {num_episodes}")
    print(f"  • 最终平均奖励: {np.mean(episode_rewards[-50:]):.4f}")
    print(f"  • 最佳移动平均奖励: {best_avg_reward:.4f}")
    print(f"\n📋 评估结果 (确定性策略):")
    print(f"  • 平均奖励: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    print(f"  • 平均步数: {eval_results['mean_steps']:.1f}")
    print(f"  • 成功率: {eval_results['success_rate']*100:.1f}%")
    print(f"  • 碰撞率: {eval_results['collision_rate']*100:.1f}%")
    
    env.close()
    
    return {
        'algorithm': algorithm,
        'reward_version': reward_version,
        'training_episodes': num_episodes,
        'episode_rewards': episode_rewards,
        'moving_avg': moving_avg,
        'episode_steps': episode_steps,
        'best_avg_reward': best_avg_reward,
        'eval_results': eval_results,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }


def generate_comparison_report(results: List[Dict]):
    """生成对比报告"""
    print("\n" + "="*80)
    print("📊 基线性能评估报告")
    print("="*80)
    
    df = pd.DataFrame([
        {
            'Algorithm': r['algorithm'].upper(),
            'Reward': r['reward_version'].upper(),
            'Training Episodes': r['training_episodes'],
            'Best Avg Reward': r['best_avg_reward'],
            'Eval Mean Reward': r['eval_results']['mean_reward'],
            'Eval Std': r['eval_results']['std_reward'],
            'Success Rate (%)': r['eval_results']['success_rate'] * 100,
            'Collision Rate (%)': r['eval_results']['collision_rate'] * 100,
            'Mean Steps': r['eval_results']['mean_steps']
        }
        for r in results
    ])
    
    print("\n📈 性能对比表:")
    print("-" * 100)
    print(df.to_string(index=False))
    
    best_row = df.loc[df['Eval Mean Reward'].idxmax()]
    print(f"\n🏆 最佳配置: {best_row['Algorithm']} + {best_row['Reward']}")
    print(f"   评估奖励: {best_row['Eval Mean Reward']:.4f} ± {best_row['Eval Std']:.4f}")
    print(f"   成功率: {best_row['Success Rate (%)']:.1f}%")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    for r in results:
        label = f"{r['algorithm'].upper()}-{r['reward_version'].upper()}"
        ax1.plot(r['moving_avg'], label=label, alpha=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Moving Average Reward')
    ax1.set_title('Training Reward Curves (Window=20)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    algorithms = [f"{r['algorithm'].upper()}\n({r['reward_version'].upper()})" for r in results]
    eval_rewards = [r['eval_results']['mean_reward'] for r in results]
    eval_stds = [r['eval_results']['std_reward'] for r in results]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    bars = ax2.bar(algorithms, eval_rewards, yerr=eval_stds, capsize=5, color=colors, alpha=0.8)
    ax2.set_ylabel('Evaluation Mean Reward')
    ax2.set_title('Evaluation Reward Comparison')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, reward in zip(bars, eval_rewards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{reward:.2f}', ha='center', va='bottom', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = axes[1, 0]
    success_rates = [r['eval_results']['success_rate'] * 100 for r in results]
    ax3.bar(algorithms, success_rates, color=colors, alpha=0.8)
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate Comparison')
    for i, rate in enumerate(success_rates):
        ax3.text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    ax3.set_ylim(0, 110)
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = axes[1, 1]
    collision_rates = [r['eval_results']['collision_rate'] * 100 for r in results]
    ax4.bar(algorithms, collision_rates, color=colors, alpha=0.8)
    ax4.set_ylabel('Collision Rate (%)')
    ax4.set_title('Collision Rate Comparison')
    for i, rate in enumerate(collision_rates):
        ax4.text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    ax4.set_ylim(0, max(collision_rates) * 1.3 + 10 if max(collision_rates) > 0 else 10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'baseline_comparison_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 图表已保存: baseline_comparison_{timestamp}.png")
    
    df.to_csv(f'baseline_results_{timestamp}.csv', index=False)
    print(f"📄 数据已保存: baseline_results_{timestamp}.csv")
    
    return df


def main():
    print("="*80)
    print("🚁 无人机强化学习 - 基线性能评估")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    experiments = [
        ('ppo', 'v1'),
        ('ppo', 'v2'),
        ('sac', 'v1'),
        ('sac', 'v2'),
    ]
    
    num_episodes = 500
    eval_episodes = 10
    
    results = []
    for algo, reward_ver in experiments:
        result = run_baseline_experiment(
            algorithm=algo,
            reward_version=reward_ver,
            num_episodes=num_episodes,
            eval_episodes=eval_episodes,
            use_depth_sensor=True,
            depth_image_size=32
        )
        results.append(result)
    
    generate_comparison_report(results)
    
    print("\n" + "="*80)
    print("✅ 基线评估完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
