"""
步骤5/5: 基线评估结果对比分析
加载所有训练的模型，进行最终的对比分析
"""
import numpy as np
import torch
import os
import json
from datetime import datetime
from drone_env import DroneEnv
from ppo_agent import PPO
from sac_agent import SAC


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model_path: str, model_type: str, reward_version: str, num_episodes: int = 100) -> dict:
    """评估单个模型"""
    set_seed(42)  # 固定种子保证可重复性
    
    env = DroneEnv()
    env.REWARD_VERSION = reward_version
    
    state_dim = env.observation_space.shape[0]
    action_dim = 3
    action_max = float(env.action_space.high[0])
    
    if model_type == "PPO":
        agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            action_max=action_max,
            lr=5e-4,
            gamma=0.99,
            clip_eps=0.2,
            epochs=10,
            hidden_dim=128
        )
        agent.model.load_model(model_path)
        eval_func = lambda: eval_ppo(agent, env, num_episodes)
    elif model_type == "SAC":
        agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            action_max=action_max,
            lr=3e-4,
            gamma=0.99,
            alpha=0.2,
            buffer_size=100000,
            batch_size=256
        )
        agent.load_model(model_path)
        eval_func = lambda: eval_sac(agent, env, num_episodes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return eval_func()


def eval_ppo(agent, env, num_episodes: int) -> dict:
    """评估PPO模型"""
    success_count = 0
    collision_count = 0
    timeout_count = 0
    total_rewards = []
    total_steps = []
    distances = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        steps = 0
        
        while True:
            action, _ = agent.model.get_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            state = next_state
            
            if info.get("reached_target", False):
                success_count += 1
                break
            elif info.get("collision", False):
                collision_count += 1
                break
            elif terminated or truncated:
                timeout_count += 1
                break
        
        total_rewards.append(ep_reward)
        total_steps.append(steps)
        distances.append(info.get("final_distance", -1))
    
    return {
        "success_rate": (success_count / num_episodes) * 100,
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_steps": np.mean(total_steps),
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "avg_distance": np.mean(distances),
        "min_reward": np.min(total_rewards),
        "max_reward": np.max(total_rewards)
    }


def eval_sac(agent, env, num_episodes: int) -> dict:
    """评估SAC模型"""
    success_count = 0
    collision_count = 0
    timeout_count = 0
    total_rewards = []
    total_steps = []
    distances = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        steps = 0
        
        while True:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            state = next_state
            
            if info.get("reached_target", False):
                success_count += 1
                break
            elif info.get("collision", False):
                collision_count += 1
                break
            elif terminated or truncated:
                timeout_count += 1
                break
        
        total_rewards.append(ep_reward)
        total_steps.append(steps)
        distances.append(info.get("final_distance", -1))
    
    return {
        "success_rate": (success_count / num_episodes) * 100,
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_steps": np.mean(total_steps),
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "avg_distance": np.mean(distances),
        "min_reward": np.min(total_rewards),
        "max_reward": np.max(total_rewards)
    }


def main():
    """主函数：对比分析所有训练结果"""
    print(f"\n{'='*80}")
    print(f"步骤5/5: 基线评估结果对比分析")
    print(f"{'='*80}")
    
    # 定义要评估的模型
    models_to_evaluate = [
        {"name": "PPO-V1", "path": "models/baseline/ppo_v1_best.pth", "type": "PPO", "reward": "v1"},
        {"name": "PPO-V2", "path": "models/baseline/ppo_v2_best.pth", "type": "PPO", "reward": "v2"},
        {"name": "SAC-V1", "path": "models/baseline/sac_v1_best.pth", "type": "SAC", "reward": "v1"},
        {"name": "SAC-V2", "path": "models/baseline/sac_v2_best.pth", "type": "SAC", "reward": "v2"},
    ]
    
    results = {}
    
    print("\n开始评估所有模型...")
    
    for model in models_to_evaluate:
        print(f"\n正在评估: {model['name']}")
        
        if not os.path.exists(model['path']):
            print(f"警告: 模型文件不存在 - {model['path']}")
            results[model['name']] = {"error": "Model file not found"}
            continue
        
        try:
            eval_result = evaluate_model(
                model_path=model['path'],
                model_type=model['type'],
                reward_version=model['reward'],
                num_episodes=100
            )
            results[model['name']] = eval_result
            
            print(f"✓ {model['name']} 评估完成")
            print(f"  成功率: {eval_result['success_rate']:.1f}%")
            print(f"  平均奖励: {eval_result['avg_reward']:.2f} ± {eval_result['std_reward']:.2f}")
            print(f"  平均步数: {eval_result['avg_steps']:.1f}")
            
        except Exception as e:
            print(f"✗ {model['name']} 评估失败: {str(e)}")
            results[model['name']] = {"error": str(e)}
    
    # 生成对比分析报告
    print(f"\n{'='*80}")
    print(f"基线评估结果对比报告")
    print(f"{'='*80}")
    
    # 创建对比表格
    print(f"\n{'算法':<8} {'奖励函数':<8} {'成功率':<10} {'平均奖励':<12} {'标准差':<10} {'平均步数':<8}")
    print("-" * 70)
    
    for name, result in results.items():
        if "error" in result:
            print(f"{name:<8} {result['error']}")
            continue
        
        algo, reward = name.split('-')
        print(f"{algo:<8} {reward:<8} {result['success_rate']:>7.1f}% {result['avg_reward']:>9.2f} "
              f"{result['std_reward']:>7.2f} {result['avg_steps']:>6.1f}")
    
    # 分析最佳表现
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if valid_results:
        print(f"\n{'='*60}")
        print(f"性能分析")
        print(f"{'='*60}")
        
        # 最佳成功率
        best_success = max(valid_results.items(), key=lambda x: x[1]['success_rate'])
        print(f"最佳成功率: {best_success[0]} ({best_success[1]['success_rate']:.1f}%)")
        
        # 最佳平均奖励
        best_reward = max(valid_results.items(), key=lambda x: x[1]['avg_reward'])
        print(f"最佳平均奖励: {best_reward[0]} ({best_reward[1]['avg_reward']:.2f})")
        
        # 算法对比
        ppo_results = {k: v for k, v in valid_results.items() if k.startswith('PPO')}
        sac_results = {k: v for k, v in valid_results.items() if k.startswith('SAC')}
        
        if ppo_results and sac_results:
            ppo_avg_success = np.mean([r['success_rate'] for r in ppo_results.values()])
            sac_avg_success = np.mean([r['success_rate'] for r in sac_results.values()])
            ppo_avg_reward = np.mean([r['avg_reward'] for r in ppo_results.values()])
            sac_avg_reward = np.mean([r['avg_reward'] for r in sac_results.values()])
            
            print(f"\n算法对比:")
            print(f"  PPO平均成功率: {ppo_avg_success:.1f}%")
            print(f"  SAC平均成功率: {sac_avg_success:.1f}%")
            print(f"  PPO平均奖励: {ppo_avg_reward:.2f}")
            print(f"  SAC平均奖励: {sac_avg_reward:.2f}")
        
        # 奖励函数对比
        v1_results = {k: v for k, v in valid_results.items() if k.endswith('V1')}
        v2_results = {k: v for k, v in valid_results.items() if k.endswith('V2')}
        
        if v1_results and v2_results:
            v1_avg_success = np.mean([r['success_rate'] for r in v1_results.values()])
            v2_avg_success = np.mean([r['success_rate'] for r in v2_results.values()])
            v1_avg_reward = np.mean([r['avg_reward'] for r in v1_results.values()])
            v2_avg_reward = np.mean([r['avg_reward'] for r in v2_results.values()])
            
            print(f"\n奖励函数对比:")
            print(f"  V1平均成功率: {v1_avg_success:.1f}%")
            print(f"  V2平均成功率: {v2_avg_success:.1f}%")
            print(f"  V1平均奖励: {v1_avg_reward:.2f}")
            print(f"  V2平均奖励: {v2_avg_reward:.2f}")
    
    # 保存结果到文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"results/baseline_evaluation_results_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存至: {results_file}")
    print(f"\n{'='*80}")
    print(f"基线评估完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()