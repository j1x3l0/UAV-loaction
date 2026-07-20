import numpy as np
import torch
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
from drone_env import DroneEnv
from ppo_agent import PPO
from sac_agent import SAC


def format_time(seconds: int) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds}秒"
    elif seconds < 3600:
        mins, secs = divmod(seconds, 60)
        return f"{mins}分{secs}秒"
    else:
        hours, mins = divmod(seconds, 3600)
        return f"{hours}小时{mins}分"


SAVE_THRESHOLD = 500


def should_save_models_and_logs(episodes):
    """判断是否保存模型和日志"""
    return episodes >= SAVE_THRESHOLD


def get_save_path(base_name, episodes, timestamp):
    """生成带时间戳和轮数的保存路径"""
    return f"saved_models/{base_name}_{episodes}ep_{timestamp}"


def train_ppo(env, config, writer=None, model_save_path=None):
    """PPO训练"""
    ppo = PPO(
        vec_state_dim=config['vec_state_dim'],
        action_dim=config['action_dim'],
        action_max=config['action_max'],
        lr=config['lr'],
        gamma=config['gamma'],
        clip_eps=config.get('clip_eps', 0.2),
        epochs=config.get('epochs', 10),
        hidden_dim=config['hidden_dim'],
        use_depth_sensor=config['use_depth_sensor'],
        depth_image_size=config['depth_image_size']
    )

    max_episodes = config['max_episodes']
    batch_size = config['batch_size']
    print_interval = config.get('print_interval', 100)
    
    do_save = should_save_models_and_logs(max_episodes)
    save_path = model_save_path if do_save and model_save_path else None
    
    if do_save and model_save_path and os.path.exists(model_save_path):
        ppo.load_model(model_save_path)
    
    total_steps = max_episodes * batch_size
    step_count = 0
    start_time = time.time()
    best_avg_reward = float('-inf')
    
    print("\n" + "=" * 80)
    print(f"🚀 PPO训练 | 总轮数：{max_episodes} | 每批步数：{batch_size}")
    print("=" * 80)
    print(f"📊 总训练步数：{total_steps:,}")
    print("-" * 80)

    try:
        # 初始化损失信息
        loss_info = {'actor_loss': 0, 'critic_loss': 0, 'alpha': 0.2}
        
        for episode in range(max_episodes):
            state, _ = env.reset()
            ep_reward = 0.0

            for step in range(batch_size):
                step_count += 1
                action, log_prob = ppo.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                ppo.store_transition(state, action, reward, next_state, done, log_prob)

                ep_reward += reward
                state = next_state

                if done:
                    state, _ = env.reset()

            loss = ppo.update()

            elapsed_time = time.time() - start_time
            avg_reward = ep_reward / batch_size
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

            if episode % print_interval == 0 or episode == max_episodes - 1:
                steps_per_sec = step_count / elapsed_time if elapsed_time > 0 else 0
                remaining_episodes = max_episodes - episode - 1
                eta_seconds = remaining_episodes / ((episode + 1) / elapsed_time) if elapsed_time > 0 else 0
                eta = format_time(int(eta_seconds))
                progress = (episode + 1) / max_episodes * 100
                bar_length = 30
                filled_length = int(bar_length * progress / 100)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                
                print(f"\n{bar} {progress:5.1f}%")
                print(f"│ 轮数：{episode+1:5d}/{max_episodes:5d} │ 步数：{step_count:8d}/{total_steps:8d} │ 速度：{steps_per_sec:8.0f} 步/秒")
                print(f"│ 平均奖励：{avg_reward:8.4f} │ 最佳奖励：{best_avg_reward:8.4f} │ 损失：{loss:8.4f}")
                print(f"│ 已用时间：{format_time(int(elapsed_time))} │ 预计剩余：{eta}")
                print(f"│ 保存模型：{'是' if do_save else '否':^74}│")
                print("│" + "-" * 76 + "│")
                
                sys.stdout.flush()
                
                if do_save and writer:
                    writer.add_scalar("PPO/Average_Reward", avg_reward, episode)
                    writer.add_scalar("PPO/Total_Loss", loss, episode)
                    writer.add_scalar("PPO/Total_Reward", ep_reward, episode)
                    writer.add_scalar("PPO/Steps_Per_Second", steps_per_sec, episode)

        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"🎉 PPO训练完成！")
        print(f"📈 总轮数：{max_episodes} | 总步数：{step_count:,}")
        print(f"⏱️  总耗时：{format_time(int(total_time))}")
        print(f"📊 平均速度：{step_count/total_time:,.0f} 步/秒")
        print(f"🏆 最佳平均奖励：{best_avg_reward:.4f}")
        if do_save and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ppo.save_model(save_path)
            print(f"💾 模型已保存：{save_path}")
        print("=" * 80)
        
        return {
            'algorithm': 'PPO',
            'total_episodes': max_episodes,
            'total_steps': step_count,
            'total_time': total_time,
            'best_avg_reward': best_avg_reward,
            'avg_steps_per_sec': step_count / total_time,
            'model_saved': do_save and save_path is not None
        }

    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print("\n\n⚠️  PPO训练被手动终止")
        if do_save and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ppo.save_model(save_path)
            print(f"💾 模型已保存：{save_path}")
        return {
            'algorithm': 'PPO',
            'total_episodes': episode + 1,
            'total_steps': step_count,
            'total_time': elapsed_time,
            'best_avg_reward': best_avg_reward,
            'avg_steps_per_sec': step_count / elapsed_time if elapsed_time > 0 else 0,
            'model_saved': do_save and save_path is not None
        }


def train_sac(env, config, writer=None, model_save_path=None):
    """SAC训练"""
    sac = SAC(
        vec_state_dim=config['vec_state_dim'],
        action_dim=config['action_dim'],
        action_max=config['action_max'],
        lr=config['lr'],
        gamma=config['gamma'],
        buffer_size=config.get('buffer_size', 100000),
        batch_size=config.get('sac_batch_size', 256),
        hidden_dim=config['hidden_dim'],
        use_depth_sensor=config['use_depth_sensor'],
        depth_image_size=config['depth_image_size']
    )

    max_episodes = config['max_episodes']
    batch_size = config['batch_size']
    print_interval = config.get('print_interval', 100)
    updates_per_step = config.get('updates_per_step', 1)
    warmup_steps = config.get('warmup_steps', 1000)
    
    do_save = should_save_models_and_logs(max_episodes)
    save_path = model_save_path if do_save and model_save_path else None
    
    if do_save and model_save_path and os.path.exists(model_save_path):
        sac.load_model(model_save_path)
    
    total_steps = max_episodes * batch_size
    step_count = 0
    start_time = time.time()
    best_avg_reward = float('-inf')
    
    print("\n" + "=" * 80)
    print(f"🚀 SAC训练 | 总轮数：{max_episodes} | 每批步数：{batch_size}")
    print("=" * 80)
    print(f"📊 总训练步数：{total_steps:,} | 预热步数：{warmup_steps}")
    print("-" * 80)

    try:
        for episode in range(max_episodes):
            state, _ = env.reset()
            ep_reward = 0.0

            for step in range(batch_size):
                step_count += 1
                
                if step_count < warmup_steps:
                    action = np.random.uniform(-1, 1, 3)
                    log_prob = 0.0
                else:
                    action, log_prob = sac.actor.get_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                sac.store_transition(state, action, reward, next_state, done)

                if step_count >= warmup_steps:
                    update_result = sac.update(updates=updates_per_step)
                    loss_info = update_result
                else:
                    loss_info = {'actor_loss': 0, 'critic_loss': 0}

                ep_reward += reward
                state = next_state

                if done:
                    state, _ = env.reset()

            avg_reward = ep_reward / batch_size
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

            if episode % print_interval == 0 or episode == max_episodes - 1:
                elapsed_time = time.time() - start_time
                steps_per_sec = step_count / elapsed_time if elapsed_time > 0 else 0
                remaining_episodes = max_episodes - episode - 1
                eta_seconds = remaining_episodes / ((episode + 1) / elapsed_time) if elapsed_time > 0 else 0
                eta = format_time(int(eta_seconds))
                progress = (episode + 1) / max_episodes * 100
                bar_length = 30
                filled_length = int(bar_length * progress / 100)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                
                actor_loss = loss_info.get('actor_loss', 0)
                critic_loss = loss_info.get('critic_loss', 0)
                alpha = loss_info.get('alpha', 0.2)
                
                print(f"\n{bar} {progress:5.1f}%")
                print(f"│ 轮数：{episode+1:5d}/{max_episodes:5d} │ 步数：{step_count:8d}/{total_steps:8d} │ 速度：{steps_per_sec:8.0f} 步/秒")
                print(f"│ 平均奖励：{avg_reward:8.4f} │ 最佳奖励：{best_avg_reward:8.4f}")
                print(f"│ Actor损失：{actor_loss:8.4f} │ Critic损失：{critic_loss:8.4f} │ Alpha：{alpha:.4f}")
                print(f"│ 已用时间：{format_time(int(elapsed_time))} │ 预计剩余：{eta}")
                print(f"│ 保存模型：{'是' if do_save else '否':^74}│")
                print("│" + "-" * 76 + "│")
                
                sys.stdout.flush()
                
                if do_save and writer:
                    writer.add_scalar("SAC/Average_Reward", avg_reward, episode)
                    writer.add_scalar("SAC/Actor_Loss", actor_loss, episode)
                    writer.add_scalar("SAC/Critic_Loss", critic_loss, episode)
                    writer.add_scalar("SAC/Alpha", alpha, episode)
                    writer.add_scalar("SAC/Total_Reward", ep_reward, episode)
                    writer.add_scalar("SAC/Steps_Per_Second", steps_per_sec, episode)

        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"🎉 SAC训练完成！")
        print(f"📈 总轮数：{max_episodes} | 总步数：{step_count:,}")
        print(f"⏱️  总耗时：{format_time(int(total_time))}")
        print(f"📊 平均速度：{step_count/total_time:,.0f} 步/秒")
        print(f"🏆 最佳平均奖励：{best_avg_reward:.4f}")
        if do_save and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            sac.save_model(save_path)
            print(f"💾 模型已保存：{save_path}")
        print("=" * 80)
        
        return {
            'algorithm': 'SAC',
            'total_episodes': max_episodes,
            'total_steps': step_count,
            'total_time': total_time,
            'best_avg_reward': best_avg_reward,
            'avg_steps_per_sec': step_count / total_time,
            'model_saved': do_save and save_path is not None
        }

    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print("\n\n⚠️  SAC训练被手动终止")
        if do_save and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            sac.save_model(save_path)
            print(f"💾 模型已保存：{save_path}")
        return {
            'algorithm': 'SAC',
            'total_episodes': episode + 1,
            'total_steps': step_count,
            'total_time': elapsed_time,
            'best_avg_reward': best_avg_reward,
            'avg_steps_per_sec': step_count / elapsed_time if elapsed_time > 0 else 0,
            'model_saved': do_save and save_path is not None
        }


def compare_experiments():
    """对比实验：运行PPO和SAC并比较结果"""
    use_depth_sensor = True
    depth_image_size = 32
    max_episodes = 500
    batch_size = 10
    hidden_dim = 256
    lr = 3e-4
    gamma = 0.99
    print_interval = 50
    
    env = DroneEnv(config={
        'use_depth_sensor': use_depth_sensor,
        'depth_image_size': depth_image_size
    })
    
    vec_state_dim = env.observation_space['vector'].shape[0]
    action_dim = 3
    action_max = float(env.action_space.high[0])
    
    config = {
        'vec_state_dim': vec_state_dim,
        'action_dim': action_dim,
        'action_max': action_max,
        'lr': lr,
        'gamma': gamma,
        'hidden_dim': hidden_dim,
        'use_depth_sensor': use_depth_sensor,
        'depth_image_size': depth_image_size,
        'max_episodes': max_episodes,
        'batch_size': batch_size,
        'print_interval': print_interval,
        'clip_eps': 0.2,
        'epochs': 10,
        'buffer_size': 100000,
        'sac_batch_size': 256,
        'updates_per_step': 1,
        'warmup_steps': 1000
    }
    
    results = []
    
    do_save = should_save_models_and_logs(max_episodes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if do_save:
        compare_log_dir = f"logs/compare_{max_episodes}ep_{timestamp}"
        ppo_model_path = get_save_path("ppo", max_episodes, timestamp)
        sac_model_path = get_save_path("sac", max_episodes, timestamp)
        writer = SummaryWriter(log_dir=compare_log_dir)
        print(f"\n📁 日志将保存至：{compare_log_dir}")
        print(f"💾 模型将保存至：saved_models/")
    else:
        compare_log_dir = None
        ppo_model_path = None
        sac_model_path = None
        writer = None
        print("\n⚠️  训练轮数较少，将不保存模型和日志")
    
    print("\n" + "=" * 80)
    print("🔬 开始对比实验：PPO vs SAC")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("【1/2】训练 PPO...")
    print("-" * 80)
    ppo_config = config.copy()
    ppo_result = train_ppo(env, ppo_config, writer, ppo_model_path)
    results.append(ppo_result)
    
    del env
    env = DroneEnv(config={
        'use_depth_sensor': use_depth_sensor,
        'depth_image_size': depth_image_size
    })
    
    print("\n" + "-" * 80)
    print("【2/2】训练 SAC...")
    print("-" * 80)
    sac_result = train_sac(env, config, writer, sac_model_path)
    results.append(sac_result)
    
    if writer:
        writer.close()
    
    print("\n" + "=" * 80)
    print("📊 对比实验结果汇总")
    print("=" * 80)
    print(f"{'算法':<10} {'总轮数':<10} {'总步数':<12} {'总时间':<15} {'最佳奖励':<12} {'速度(步/秒)':<12} {'已保存':<8}")
    print("-" * 80)
    for r in results:
        time_str = format_time(int(r['total_time']))
        saved = '是' if r.get('model_saved', False) else '否'
        print(f"{r['algorithm']:<10} {r['total_episodes']:<10} {r['total_steps']:<12} {time_str:<15} {r['best_avg_reward']:<12.4f} {r['avg_steps_per_sec']:<12.0f} {saved:<8}")
    
    winner = max(results, key=lambda x: x['best_avg_reward'])
    print("-" * 80)
    print(f"🏆 获胜者：{winner['algorithm']}（最佳平均奖励：{winner['best_avg_reward']:.4f}）")
    if do_save:
        print(f"📁 日志目录：{compare_log_dir}")
    print("=" * 80)
    
def train_both(env, config, writer=None, model_save_path_prefix=None):
    """
    同时训练PPO和SAC算法，共享经验
    """
    # 初始化PPO
    ppo = PPO(
        vec_state_dim=config['vec_state_dim'],
        action_dim=config['action_dim'],
        action_max=config['action_max'],
        lr=config['lr'],
        gamma=config['gamma'],
        clip_eps=config.get('clip_eps', 0.2),
        epochs=config.get('epochs', 10),
        hidden_dim=config['hidden_dim'],
        use_depth_sensor=config['use_depth_sensor'],
        depth_image_size=config['depth_image_size']
    )

    # 初始化SAC
    sac = SAC(
        vec_state_dim=config['vec_state_dim'],
        action_dim=config['action_dim'],
        action_max=config['action_max'],
        lr=config['lr'],
        gamma=config['gamma'],
        buffer_size=config.get('buffer_size', 100000),
        batch_size=config.get('sac_batch_size', 256),
        hidden_dim=config['hidden_dim'],
        use_depth_sensor=config['use_depth_sensor'],
        depth_image_size=config['depth_image_size']
    )

    max_episodes = config['max_episodes']
    batch_size = config['batch_size']
    print_interval = config.get('print_interval', 100)
    updates_per_step = config.get('updates_per_step', 1)
    warmup_steps = config.get('warmup_steps', 1000)
    
    do_save = should_save_models_and_logs(max_episodes)
    ppo_save_path = None
    sac_save_path = None
    if do_save and model_save_path_prefix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ppo_save_path = f"{model_save_path_prefix}_ppo_{max_episodes}ep_{timestamp}"
        sac_save_path = f"{model_save_path_prefix}_sac_{max_episodes}ep_{timestamp}"
    
    total_steps = max_episodes * batch_size
    step_count = 0
    start_time = time.time()
    ppo_best_avg_reward = float('-inf')
    sac_best_avg_reward = float('-inf')
    
    print("\n" + "=" * 80)
    print(f"🚀 同时训练PPO和SAC | 总轮数：{max_episodes} | 每批步数：{batch_size}")
    print("=" * 80)
    print(f"📊 总训练步数：{total_steps:,} | 预热步数：{warmup_steps}")
    print("-" * 80)

    try:
        for episode in range(max_episodes):
            state, _ = env.reset()
            ppo_ep_reward = 0.0
            sac_ep_reward = 0.0

            for step in range(batch_size):
                step_count += 1
                
                # 在预热阶段使用随机动作，之后随机选择PPO或SAC的动作
                if step_count < warmup_steps:
                    action = np.random.uniform(-1, 1, 3)
                    ppo_log_prob = 0.0
                    sac_log_prob = 0.0
                    algorithm_used = "random"
                else:
                    # 随机选择使用PPO或SAC的动作
                    if np.random.random() < 0.5:
                        # 使用PPO动作
                        action, ppo_log_prob = ppo.get_action(state)
                        sac_log_prob = 0.0
                        algorithm_used = "PPO"
                    else:
                        # 使用SAC动作
                        action, sac_log_prob = sac.get_action(state)
                        ppo_log_prob = 0.0
                        algorithm_used = "SAC"

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # 存储经验到PPO（需要log_prob）
                ppo.store_transition(state, action, reward, next_state, done, ppo_log_prob)
                
                # 存储经验到SAC
                sac.store_transition(state, action, reward, next_state, done)

                # 根据使用的算法分配奖励
                if algorithm_used == "random":
                    # 预热阶段，两个算法平分奖励（各得一半）
                    ppo_ep_reward += reward / 2
                    sac_ep_reward += reward / 2
                elif algorithm_used == "PPO":
                    # 只有PPO获得全部奖励
                    ppo_ep_reward += reward
                elif algorithm_used == "SAC":
                    # 只有SAC获得全部奖励
                    sac_ep_reward += reward

                # 更新PPO（每10步更新一次）
                if step_count % 10 == 0 and step_count >= warmup_steps:
                    ppo_loss = ppo.update()
                else:
                    ppo_loss = 0.0

                # 更新SAC（每步更新，但仅在预热后）
                if step_count >= warmup_steps:
                    sac_loss_info = sac.update(updates=updates_per_step)
                else:
                    sac_loss_info = {'actor_loss': 0, 'critic_loss': 0, 'alpha': 0.2}

                state = next_state

                if done:
                    state, _ = env.reset()

            # 计算平均奖励
            ppo_avg_reward = ppo_ep_reward / batch_size
            sac_avg_reward = sac_ep_reward / batch_size
            
            if ppo_avg_reward > ppo_best_avg_reward:
                ppo_best_avg_reward = ppo_avg_reward
            if sac_avg_reward > sac_best_avg_reward:
                sac_best_avg_reward = sac_avg_reward

            if episode % print_interval == 0 or episode == max_episodes - 1:
                elapsed_time = time.time() - start_time
                steps_per_sec = step_count / elapsed_time if elapsed_time > 0 else 0
                remaining_episodes = max_episodes - episode - 1
                eta_seconds = remaining_episodes / ((episode + 1) / elapsed_time) if elapsed_time > 0 else 0
                eta = format_time(int(eta_seconds))
                progress = (episode + 1) / max_episodes * 100
                bar_length = 30
                filled_length = int(bar_length * progress / 100)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                
                ppo_actor_loss = sac_loss_info.get('actor_loss', 0)
                ppo_critic_loss = sac_loss_info.get('critic_loss', 0)
                sac_alpha = sac_loss_info.get('alpha', 0.2)
                
                print(f"\n{bar} {progress:5.1f}%")
                print(f"│ 轮数：{episode+1:5d}/{max_episodes:5d} │ 步数：{step_count:8d}/{total_steps:8d} │ 速度：{steps_per_sec:8.0f} 步/秒")
                print(f"│ PPO平均奖励：{ppo_avg_reward:8.4f} │ PPO最佳：{ppo_best_avg_reward:8.4f} │ PPO损失：{ppo_loss:8.4f}")
                print(f"│ SAC平均奖励：{sac_avg_reward:8.4f} │ SAC最佳：{sac_best_avg_reward:8.4f} │ SAC Alpha：{sac_alpha:.4f}")
                print(f"│ 已用时间：{format_time(int(elapsed_time))} │ 预计剩余：{eta}")
                print(f"│ 保存模型：{'是' if do_save else '否':^74}│")
                print("│" + "-" * 76 + "│")
                
                sys.stdout.flush()
                
                if do_save and writer:
                    writer.add_scalar("Both/PPO_Average_Reward", ppo_avg_reward, episode)
                    writer.add_scalar("Both/SAC_Average_Reward", sac_avg_reward, episode)
                    writer.add_scalar("Both/PPO_Loss", ppo_loss, episode)
                    writer.add_scalar("Both/SAC_Actor_Loss", ppo_actor_loss, episode)
                    writer.add_scalar("Both/SAC_Critic_Loss", ppo_critic_loss, episode)
                    writer.add_scalar("Both/Steps_Per_Second", steps_per_sec, episode)

        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"🎉 PPO和SAC同时训练完成！")
        print(f"📈 总轮数：{max_episodes} | 总步数：{step_count:,}")
        print(f"⏱️  总耗时：{format_time(int(total_time))}")
        print(f"📊 平均速度：{step_count/total_time:,.0f} 步/秒")
        print(f"🏆 PPO最佳平均奖励：{ppo_best_avg_reward:.4f}")
        print(f"🏆 SAC最佳平均奖励：{sac_best_avg_reward:.4f}")
        if do_save and ppo_save_path and sac_save_path:
            os.makedirs(os.path.dirname(ppo_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(sac_save_path), exist_ok=True)
            ppo.save_model(ppo_save_path)
            sac.save_model(sac_save_path)
            print(f"💾 PPO模型已保存：{ppo_save_path}")
            print(f"💾 SAC模型已保存：{sac_save_path}")
        print("=" * 80)
        
        return {
            'algorithm': 'Both',
            'total_episodes': max_episodes,
            'total_steps': step_count,
            'total_time': total_time,
            'ppo_best_avg_reward': ppo_best_avg_reward,
            'sac_best_avg_reward': sac_best_avg_reward,
            'avg_steps_per_sec': step_count / total_time,
            'ppo_model_saved': do_save and ppo_save_path is not None,
            'sac_model_saved': do_save and sac_save_path is not None
        }

    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print("\n\n⚠️  PPO和SAC同时训练被手动终止")
        if do_save and ppo_save_path and sac_save_path:
            os.makedirs(os.path.dirname(ppo_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(sac_save_path), exist_ok=True)
            ppo.save_model(ppo_save_path)
            sac.save_model(sac_save_path)
            print(f"💾 PPO模型已保存：{ppo_save_path}")
            print(f"💾 SAC模型已保存：{sac_save_path}")
        return {
            'algorithm': 'Both',
            'total_episodes': episode + 1,
            'total_steps': step_count,
            'total_time': elapsed_time,
            'ppo_best_avg_reward': ppo_best_avg_reward,
            'sac_best_avg_reward': sac_best_avg_reward,
            'avg_steps_per_sec': step_count / elapsed_time if elapsed_time > 0 else 0,
            'ppo_model_saved': do_save and ppo_save_path is not None,
            'sac_model_saved': do_save and sac_save_path is not None
        }


def main():
    parser = argparse.ArgumentParser(description='无人机强化学习训练')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac', 'both', 'compare'],
                        help='选择算法: ppo, sac, both(同时训练), 或 compare(对比实验)')
    parser.add_argument('--episodes', type=int, default=2000, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=10, help='每批步数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--depth-size', type=int, default=64, help='深度图像尺寸')
    parser.add_argument('--no-sensor', action='store_true', help='禁用深度传感器')
    
    args = parser.parse_args()
    
    use_depth_sensor = not args.no_sensor
    depth_image_size = args.depth_size
    max_episodes = args.episodes
    batch_size = args.batch_size
    lr = args.lr
    hidden_dim = 256
    gamma = 0.99
    print_interval = 50
    
    env = DroneEnv(config={
        'use_depth_sensor': use_depth_sensor,
        'depth_image_size': depth_image_size
    })
    
    vec_state_dim = env.observation_space['vector'].shape[0]
    action_dim = 3
    action_max = float(env.action_space.high[0])
    
    config = {
        'vec_state_dim': vec_state_dim,
        'action_dim': action_dim,
        'action_max': action_max,
        'lr': lr,
        'gamma': gamma,
        'hidden_dim': hidden_dim,
        'use_depth_sensor': use_depth_sensor,
        'depth_image_size': depth_image_size,
        'max_episodes': max_episodes,
        'batch_size': batch_size,
        'print_interval': print_interval
    }
    
    if args.algorithm == 'compare':
        compare_experiments()
        return
    
    do_save = should_save_models_and_logs(max_episodes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if do_save:
        log_dir = f"logs/{args.algorithm}_{max_episodes}ep_{timestamp}"
        writer = SummaryWriter(log_dir=log_dir)
        print(f"\n📁 日志将保存至：{log_dir}")
    else:
        log_dir = None
        writer = None
        print(f"\n⚠️  训练轮数较少（{max_episodes} < {SAVE_THRESHOLD}），将不保存模型和日志")
    
    if args.algorithm == 'ppo':
        model_path = get_save_path(args.algorithm, max_episodes, timestamp) if do_save else None
        result = train_ppo(env, config, writer, model_path)
    elif args.algorithm == 'sac':
        config['buffer_size'] = 100000
        config['sac_batch_size'] = 256
        config['updates_per_step'] = 1
        config['warmup_steps'] = 1000
        model_path = get_save_path(args.algorithm, max_episodes, timestamp) if do_save else None
        result = train_sac(env, config, writer, model_path)
    elif args.algorithm == 'both':
        # 对于同时训练，我们需要一个路径前缀，train_both会为PPO和SAC分别添加后缀
        model_save_path_prefix = None
        if do_save:
            model_save_path_prefix = f"saved_models/both_{max_episodes}ep_{timestamp}"
            print(f"💾 PPO和SAC模型将保存至：{model_save_path_prefix}_ppo 和 {model_save_path_prefix}_sac")
        else:
            model_save_path_prefix = None
        config['buffer_size'] = 100000
        config['sac_batch_size'] = 256
        config['updates_per_step'] = 1
        config['warmup_steps'] = 200
        result = train_both(env, config, writer, model_save_path_prefix)
    
    if writer:
        writer.close()
    
    print("\n" + "=" * 80)
    print("🏆 训练完成！")
    print(f"📊 总轮数：{result['total_episodes']} | 总步数：{result['total_steps']:,}")
    print(f"⏱️  总耗时：{format_time(int(result['total_time']))}")
    if args.algorithm == 'both':
        print(f"🏆 PPO最佳平均奖励：{result['ppo_best_avg_reward']:.4f}")
        print(f"🏆 SAC最佳平均奖励：{result['sac_best_avg_reward']:.4f}")
    else:
        print(f"🏆 最佳平均奖励：{result['best_avg_reward']:.4f}")
    if do_save:
        print(f"📁 日志目录：{log_dir}")
        if args.algorithm == 'both':
            if result.get('ppo_model_saved', False) and result.get('sac_model_saved', False):
                print(f"💾 PPO模型已保存：{model_save_path_prefix}_ppo")
                print(f"💾 SAC模型已保存：{model_save_path_prefix}_sac")
        else:
            print(f"💾 模型已保存：{model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
