import numpy as np
import torch
import os
import sys
import time
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import SyncVectorEnv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from drone_env import DroneEnv
from drone_env_noisy import NoisyDroneEnv
from ppo_agent import PPO
from config import TrainingConfig, parse_arguments, create_config_from_args


def _make_env(config_dict):
    """
    环境工厂：根据噪声配置创建 DroneEnv 或 NoisyDroneEnv

    噪声为 None 或空时返回纯净 DroneEnv，否则按 noise_pattern 创建带噪声环境。
    """
    noise_pattern = config_dict.get('noise_pattern', None)
    noise_sigma = config_dict.get('noise_sigma', 0.0)

    if noise_pattern and noise_sigma > 0:
        return NoisyDroneEnv.from_pattern(noise_pattern, sigma=noise_sigma)
    else:
        return DroneEnv()


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        mins, secs = divmod(seconds, 60)
        return f"{mins:.0f}分{secs:.0f}秒"
    else:
        hours, mins = divmod(seconds, 3600)
        return f"{hours:.0f}小时{mins:.0f}分"


def evaluate_model(agent, env, eval_episodes=50):
    """
    多维度评估函数（Swift 风格评估体系）
    评估指标：成功率、碰撞率、超时率、平均奖励、路径长度、最小障碍物距离
    """
    results = {
        'success_rate': 0.0,
        'collision_rate': 0.0,
        'timeout_rate': 0.0,
        'avg_reward': 0.0,
        'reward_std': 0.0,
        'avg_steps': 0.0,
        'avg_path_length': 0.0,
        'avg_min_obs_dist': 0.0
    }

    rewards = []
    successes = 0
    collisions = 0
    timeouts = 0
    total_steps = 0
    total_path_length = 0
    min_obs_dists = []

    for _ in range(eval_episodes):
        state, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        path_length = 0.0
        prev_pos = state[:3]
        min_dist = float('inf')

        while True:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            path_length += np.linalg.norm(next_state[:3] - prev_pos)
            prev_pos = next_state[:3]

            if 'current_pos' in info:
                pos = info['current_pos']
                obs_dist = env._get_min_obstacle_distance(pos)
                min_dist = min(min_dist, obs_dist)

            if terminated or truncated:
                if info.get('reached_target', False):
                    successes += 1
                elif info.get('collision', False):
                    collisions += 1
                else:
                    timeouts += 1
                break

            state = next_state

        rewards.append(ep_reward)
        total_steps += ep_steps
        total_path_length += path_length
        min_obs_dists.append(min_dist)

    results['success_rate'] = successes / eval_episodes * 100
    results['collision_rate'] = collisions / eval_episodes * 100
    results['timeout_rate'] = timeouts / eval_episodes * 100
    results['avg_reward'] = np.mean(rewards)
    results['reward_std'] = np.std(rewards)
    results['avg_steps'] = total_steps / eval_episodes if successes > 0 else 0
    results['avg_path_length'] = total_path_length / eval_episodes if successes > 0 else 0
    results['avg_min_obs_dist'] = np.mean(min_obs_dists) if min_obs_dists else 0

    return results


def train_ppo(config_dict, writer=None, model_save_path=None):
    """
    PPO 训练（Swift 改进版）
    特性：8环境并行采样、GAE、自适应熵系数、学习率线性衰减
    """
    num_envs = config_dict['num_envs']
    rollout_steps = config_dict['rollout_steps']
    max_episodes = config_dict['max_episodes']
    minibatch_size = config_dict['minibatch_size']

    # 使用 SyncVectorEnv 实现 8 环境并行采样（支持噪声环境）
    env = SyncVectorEnv([lambda: _make_env(config_dict) for _ in range(num_envs)])

    ppo = PPO(
        state_dim=config_dict['state_dim'],
        action_dim=config_dict['action_dim'],
        action_max=config_dict['action_max'],
        lr=config_dict['lr'],
        gamma=config_dict['gamma'],
        gae_lambda=config_dict['gae_lambda'],
        clip_eps=config_dict['clip_eps'],
        epochs=config_dict['epochs'],
        minibatch_size=minibatch_size,
        hidden_dim=config_dict['hidden_dim'],
        use_adaptive_entropy=config_dict['use_adaptive_entropy'],
        num_envs=num_envs
    )

    if model_save_path and os.path.exists(model_save_path):
        ppo.load_model(model_save_path)

    eval_env = _make_env(config_dict)

    total_steps = max_episodes * rollout_steps
    step_count = 0
    start_time = time.time()
    best_success_rate = 0.0

    noise_pattern = config_dict.get('noise_pattern', None)
    noise_sigma = config_dict.get('noise_sigma', 0.0)
    noise_info = f" | 噪声: {noise_pattern}(σ={noise_sigma})" if noise_pattern else " | 无噪声"

    logger.info("\n" + "=" * 80)
    logger.info(f"PPO训练 (Swift改进版) | 总轮数：{max_episodes} | 并行环境：{num_envs}{noise_info}")
    logger.info(f"每轮采样：{rollout_steps}步 | 总训练步数：{total_steps:,}")
    logger.info("=" * 80)

    try:
        states = env.reset()[0]

        for episode in range(max_episodes):
            ep_rewards = np.zeros(num_envs)
            ep_lengths = np.zeros(num_envs)
            reward_components = {
                'r_dist': [], 'r_heading': [], 'r_obs': [],
                'r_smooth': [], 'r_goal': [], 'r_collision': [], 'r_timeout': []
            }

            for step in range(rollout_steps):
                step_count += num_envs

                actions = []
                log_probs = []
                values = []
                entropies = []

                for i in range(num_envs):
                    action, log_prob, value, entropy = ppo.get_action(states[i], deterministic=False)
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
                    entropies.append(entropy)

                actions_np = np.array(actions)
                next_states, rewards, terminateds, truncateds, infos = env.step(actions_np)

                dones = np.logical_or(terminateds, truncateds)

                for i in range(num_envs):
                    ppo.store_transition(
                        state=states[i],
                        action=actions_np[i],
                        reward=rewards[i],
                        next_state=next_states[i],
                        done=dones[i],
                        log_prob=log_probs[i],
                        value=values[i],
                        entropy=entropies[i]
                    )

                    ep_rewards[i] += rewards[i]
                    ep_lengths[i] += 1

                    if 'reward_components' in infos:
                        comps = infos['reward_components']
                        if isinstance(comps, list) and i < len(comps):
                            for key in reward_components:
                                if key in comps[i]:
                                    reward_components[key].append(comps[i][key])

                states = next_states

            # 学习率线性衰减
            progress = episode / max_episodes
            lr = config_dict['lr'] * (1 - progress)
            ppo.set_lr(lr)

            update_result = ppo.update()

            avg_reward = np.mean(ep_rewards)
            avg_length = np.mean(ep_lengths)

            if episode % config_dict['print_interval'] == 0 or episode == max_episodes - 1:
                elapsed_time = time.time() - start_time
                steps_per_sec = step_count / elapsed_time if elapsed_time > 0 else 0
                remaining_episodes = max_episodes - episode - 1
                eta_seconds = remaining_episodes * elapsed_time / (episode + 1) if elapsed_time > 0 else 0
                eta = format_time(eta_seconds)
                progress_pct = (episode + 1) / max_episodes * 100
                bar_length = 30
                filled_length = int(bar_length * progress_pct / 100)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)

                logger.info(f"\n{bar} {progress_pct:5.1f}%")
                logger.info(f"│ 轮数：{episode+1:5d}/{max_episodes:5d} │ 步数：{step_count:8d}/{total_steps:8d} │ 速度：{steps_per_sec:8.0f} 步/秒")
                logger.info(f"│ 平均奖励：{avg_reward:8.4f} │ 平均长度：{avg_length:6.1f} │ 学习率：{lr:.2e}")
                logger.info(f"│ 总损失：{update_result['total_loss']:8.4f} │ Actor损失：{update_result['actor_loss']:8.4f} │ Critic损失：{update_result['critic_loss']:8.4f}")
                logger.info(f"│ 熵：{update_result['entropy']:8.4f} │ 熵系数：{update_result['entropy_coeff']:.4f}")
                logger.info(f"│ 已用时间：{format_time(elapsed_time)} │ 预计剩余：{eta}")
                logger.info("│" + "-" * 76 + "│")

                sys.stdout.flush()

                if writer:
                    writer.add_scalar("PPO/Average_Reward", avg_reward, episode)
                    writer.add_scalar("PPO/Average_Length", avg_length, episode)
                    writer.add_scalar("PPO/Learning_Rate", lr, episode)
                    writer.add_scalar("PPO/Total_Loss", update_result['total_loss'], episode)
                    writer.add_scalar("PPO/Actor_Loss", update_result['actor_loss'], episode)
                    writer.add_scalar("PPO/Critic_Loss", update_result['critic_loss'], episode)
                    writer.add_scalar("PPO/Entropy", update_result['entropy'], episode)
                    writer.add_scalar("PPO/Entropy_Coeff", update_result['entropy_coeff'], episode)
                    writer.add_scalar("PPO/Steps_Per_Second", steps_per_sec, episode)

                    for key, vals in reward_components.items():
                        if vals:
                            writer.add_scalar(f"Reward/{key}", np.mean(vals), episode)

            if episode % config_dict['eval_interval'] == 0 or episode == max_episodes - 1:
                eval_results = evaluate_model(ppo, eval_env, eval_episodes=config_dict['eval_episodes'])

                logger.info(f"\n📊 评估结果（轮 {episode+1}）:")
                logger.info(f"   成功率: {eval_results['success_rate']:.1f}%")
                logger.info(f"   碰撞率: {eval_results['collision_rate']:.1f}%")
                logger.info(f"   超时率: {eval_results['timeout_rate']:.1f}%")
                logger.info(f"   平均奖励: {eval_results['avg_reward']:.4f} ± {eval_results['reward_std']:.4f}")
                logger.info(f"   成功平均步数: {eval_results['avg_steps']:.1f}")
                logger.info(f"   成功平均路径长度: {eval_results['avg_path_length']:.2f}m")
                logger.info(f"   碰撞前最小障碍物距离: {eval_results['avg_min_obs_dist']:.2f}m")

                if writer:
                    writer.add_scalar("Eval/Success_Rate", eval_results['success_rate'], episode)
                    writer.add_scalar("Eval/Collision_Rate", eval_results['collision_rate'], episode)
                    writer.add_scalar("Eval/Timeout_Rate", eval_results['timeout_rate'], episode)
                    writer.add_scalar("Eval/Average_Reward", eval_results['avg_reward'], episode)
                    writer.add_scalar("Eval/Reward_Std", eval_results['reward_std'], episode)
                    writer.add_scalar("Eval/Average_Steps", eval_results['avg_steps'], episode)
                    writer.add_scalar("Eval/Average_Path_Length", eval_results['avg_path_length'], episode)
                    writer.add_scalar("Eval/Average_Min_Obs_Dist", eval_results['avg_min_obs_dist'], episode)

                if eval_results['success_rate'] > best_success_rate:
                    best_success_rate = eval_results['success_rate']
                    if model_save_path:
                        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                        ppo.save_model(model_save_path)
                        logger.info(f"🏆 新最佳模型已保存：{model_save_path}")

        total_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info(f"🎉 PPO训练完成！")
        logger.info(f"📈 总轮数：{max_episodes} | 总步数：{step_count:,}")
        logger.info(f"⏱️  总耗时：{format_time(total_time)}")
        logger.info(f"📊 平均速度：{step_count/total_time:,.0f} 步/秒")
        logger.info(f"🏆 最佳成功率：{best_success_rate:.1f}%")
        if model_save_path:
            logger.info(f"💾 最佳模型已保存：{model_save_path}")
        logger.info("=" * 80)

        env.close()
        eval_env.close()

        return {
            'algorithm': 'PPO',
            'total_episodes': max_episodes,
            'total_steps': step_count,
            'total_time': total_time,
            'best_success_rate': best_success_rate,
            'avg_steps_per_sec': step_count / total_time,
            'model_saved': model_save_path is not None
        }

    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        logger.info("\n\n⚠️  PPO训练被手动终止")
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            ppo.save_model(model_save_path)
            logger.info(f"💾 模型已保存：{model_save_path}")

        env.close()
        eval_env.close()

        return {
            'algorithm': 'PPO',
            'total_episodes': episode + 1,
            'total_steps': step_count,
            'total_time': elapsed_time,
            'best_success_rate': best_success_rate,
            'avg_steps_per_sec': step_count / elapsed_time if elapsed_time > 0 else 0,
            'model_saved': model_save_path is not None
        }


def main():
    args = parse_arguments()
    config = create_config_from_args(args)

    logger.info(str(config))

    do_save = config.should_save()
    timestamp = config.get_timestamp()

    if do_save:
        paths = config.get_save_paths()
        log_dir = paths['log_dir']
        model_path = paths['model_path']
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"📁 日志将保存至：{log_dir}")
        logger.info(f"💾 模型将保存至：{model_path}")
    else:
        log_dir = None
        model_path = None
        writer = None
        logger.info(f"⚠️  训练轮数较少（{config.max_episodes} < {config.save_threshold}），将不保存模型和日志")

    config_dict = config.get_algorithm_config()

    result = train_ppo(config_dict, writer, model_path)

    if writer:
        writer.close()

    logger.info("\n" + "=" * 80)
    logger.info("🏆 训练完成！")
    logger.info(f"📊 总轮数：{result['total_episodes']} | 总步数：{result['total_steps']:,}")
    logger.info(f"⏱️  总耗时：{format_time(result['total_time'])}")
    logger.info(f"🏆 最佳成功率：{result['best_success_rate']:.1f}%")
    if do_save:
        logger.info(f"📁 日志目录：{log_dir}")
        logger.info(f"💾 模型已保存：{model_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
