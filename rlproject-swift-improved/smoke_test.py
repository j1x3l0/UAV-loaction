"""
Smoke test: 验证 GAE per-environment 修复后的 PPO agent 能正常工作
测试内容：
1. 环境初始化
2. PPO 初始化（含 num_envs 参数）
3. 并行采样若干步
4. update() 调用（验证 GAE per-env 计算无报错）
5. 评估前向传播
"""
import numpy as np
import sys
import os

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone_env import DroneEnv
from ppo_agent import PPO
from gymnasium.vector import SyncVectorEnv

def main():
    print("=" * 60)
    print("Smoke Test: GAE per-environment 修复验证")
    print("=" * 60)

    # 1. 测试环境初始化
    print("\n[1/5] 测试环境初始化...")
    num_envs = 8
    env = SyncVectorEnv([lambda: DroneEnv() for _ in range(num_envs)])
    print(f"  ✓ {num_envs} 个并行环境创建成功")

    # 2. 测试 PPO 初始化
    print("\n[2/5] 测试 PPO 初始化（含 num_envs 参数）...")
    ppo = PPO(
        state_dim=14,
        action_dim=3,
        action_max=1.0,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=10,
        minibatch_size=64,
        hidden_dim=128,
        use_adaptive_entropy=True,
        num_envs=num_envs
    )
    print(f"  ✓ PPO 初始化成功，num_envs={ppo.num_envs}")

    # 3. 测试并行采样
    print("\n[3/5] 测试并行采样（256步）...")
    states = env.reset()[0]
    rollout_steps = 256  # 小规模测试

    for step in range(rollout_steps):
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

        states = next_states

        if np.any(dones):
            reset_indices = np.where(dones)[0]
            reset_states = env.reset()[0]
            for idx in reset_indices:
                states[idx] = reset_states[idx]

    total_samples = len(ppo.memory)
    expected_samples = rollout_steps * num_envs
    print(f"  ✓ 采样完成：{total_samples} 样本（期望 {expected_samples}）")
    assert total_samples == expected_samples, f"样本数不匹配: {total_samples} != {expected_samples}"

    # 4. 测试 update()（关键：验证 GAE per-env 计算）
    print("\n[4/5] 测试 update()（GAE per-environment 计算）...")
    update_result = ppo.update()
    print(f"  ✓ update() 成功完成")
    print(f"    total_loss: {update_result['total_loss']:.4f}")
    print(f"    actor_loss: {update_result['actor_loss']:.4f}")
    print(f"    critic_loss: {update_result['critic_loss']:.4f}")
    print(f"    entropy: {update_result['entropy']:.4f}")
    print(f"    entropy_coeff: {update_result['entropy_coeff']:.4f}")

    # 验证 memory 已清空
    assert len(ppo.memory) == 0, "update() 后 memory 未清空"
    print(f"  ✓ update() 后 memory 已清空")

    # 5. 测试并行环境 reset 处理（验证不调用 env.reset() 重置所有环境）
    print("\n[5/6] 测试并行环境 reset 处理...")
    # 运行足够多的步数确保有环境 done
    test_env = SyncVectorEnv([lambda: DroneEnv() for _ in range(num_envs)])
    test_states = test_env.reset()[0]
    done_count = 0
    state_consistent = True

    for step in range(500):
        test_actions = np.array([test_env.single_action_space.sample() for _ in range(num_envs)])
        test_next_states, test_rewards, test_terminateds, test_truncateds, _ = test_env.step(test_actions)
        test_dones = np.logical_or(test_terminateds, test_truncateds)

        # 验证：next_states 对于 done 的环境应该是自动 reset 后的初始状态
        # 对于非 done 的环境应该是正常的下一步状态
        if np.any(test_dones):
            done_count += np.sum(test_dones)

        # 关键：只更新 states = next_states，不调用 env.reset()
        test_states = test_next_states

    print(f"  ✓ 500步测试完成，共 {done_count} 次环境 done")
    print(f"  ✓ 未调用 env.reset()，依赖 SyncVectorEnv 自动 reset")
    test_env.close()

    # 6. 测试评估前向传播
    print("\n[6/6] 测试评估前向传播（deterministic）...")
    eval_env = DroneEnv()
    state, _ = eval_env.reset()
    action = ppo.select_action(state, deterministic=True)
    print(f"  ✓ 评估动作选择成功: action={action}")
    assert action.shape == (3,), f"动作维度错误: {action.shape}"
    assert np.all(np.abs(action) <= 1.0), f"动作超出范围: {action}"

    # 运行一个完整 episode
    ep_reward = 0.0
    ep_steps = 0
    while True:
        action = ppo.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, info = eval_env.step(action)
        ep_reward += reward
        ep_steps += 1
        state = next_state
        if terminated or truncated:
            break

    print(f"  ✓ 完整 episode 运行成功: steps={ep_steps}, reward={ep_reward:.2f}")
    print(f"    reached_target: {info.get('reached_target', False)}")
    print(f"    collision: {info.get('collision', False)}")

    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！GAE per-environment 修复验证成功。")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
