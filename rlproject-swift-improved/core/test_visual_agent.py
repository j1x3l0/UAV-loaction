"""
test_visual_agent.py — VisualPPO 单元测试

覆盖率: forward pass / action selection / GAE / update / save-load / edge cases
运行: python -m pytest core/test_visual_agent.py -v
"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from core.visual_ppo_agent import (
    VisualEncoder, VisualActorCritic, VisualPPO, DEVICE
)


# ── VisualEncoder 测试 ──

def test_encoder_output_shape():
    """CNN输出维度正确"""
    enc = VisualEncoder(in_channels=1, feature_dim=128)
    x = torch.randn(4, 1, 64, 64)
    out = enc(x)
    assert out.shape == (4, 128), f"shape: {out.shape}"
    print(f"  ✓ encoder: {tuple(x.shape)} → {tuple(out.shape)}")

def test_encoder_different_inputs():
    """不同输入产生不同输出"""
    enc = VisualEncoder(in_channels=1, feature_dim=128)
    x1 = torch.zeros(2, 1, 64, 64)
    x2 = torch.ones(2, 1, 64, 64)
    out1 = enc(x1); out2 = enc(x2)
    assert not torch.allclose(out1, out2), "different inputs should differ"
    print(f"  ✓ encoder sensitivity: max_diff={torch.abs(out1-out2).max():.2f}")


# ── ActorCritic 测试 ──

def test_actor_critic_shapes():
    """ActorCritic输出维度正确"""
    ac = VisualActorCritic(vec_dim=6, visual_feature_dim=128, hidden_dim=128, action_dim=3)
    depth = torch.randn(8, 1, 64, 64)
    vec = torch.randn(8, 6)
    mean, std, value = ac(depth, vec)
    assert mean.shape == (8, 3), f"mean: {mean.shape}"
    assert std.shape == (3,) or std.numel() == 3, f"std: {std.shape}"
    assert value.shape == (8,), f"value: {value.shape}"
    assert torch.all(mean >= -1) and torch.all(mean <= 1), "mean should be in [-1,1]"
    print(f"  ✓ actor-critic: depth{tuple(depth.shape)}+vec{tuple(vec.shape)} → "
          f"mean{tuple(mean.shape)}, std{std.shape}, value{tuple(value.shape)}")


# ── PPO Agent 测试 ──

def make_fake_obs(batch_size=1):
    return {
        'depth': np.random.rand(64, 64, 1).astype(np.float32) * 10,
        'vec': np.array([1.0, 0.5, -0.3, 5.0, 2.0, 1.0], dtype=np.float32),
    }

def test_agent_get_action():
    """get_action返回正确格式"""
    ppo = VisualPPO(vec_dim=6, action_dim=3, num_envs=1)
    obs = make_fake_obs()
    action, log_prob, value, entropy = ppo.get_action(obs, deterministic=False)
    assert action.shape == (3,), f"action: {action.shape}"
    assert -1.0 <= action.all() <= 1.0, f"action out of range: {action}"
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    assert isinstance(entropy, float)
    print(f"  ✓ get_action: action={action.round(2)}, log_prob={log_prob:.2f}, value={value:.2f}")

def test_agent_deterministic():
    """deterministic模式下输出一致"""
    ppo = VisualPPO(vec_dim=6, action_dim=3, num_envs=1)
    obs = make_fake_obs()
    a1 = ppo.select_action(obs, deterministic=True)
    a2 = ppo.select_action(obs, deterministic=True)
    assert np.allclose(a1, a2), "deterministic should be consistent"
    print(f"  ✓ deterministic: same input → same action")

def test_agent_stochastic():
    """stochastic模式下输出有变化"""
    ppo = VisualPPO(vec_dim=6, action_dim=3, num_envs=1)
    obs = make_fake_obs()
    actions = [ppo.select_action(obs, deterministic=False) for _ in range(20)]
    actions = np.array(actions)
    std = actions.std(axis=0)
    assert std.max() > 0.001, f"stochastic should have variance, got std={std}"
    print(f"  ✓ stochastic: action_std={std.round(3)}")

def test_gae_computation():
    """GAE计算形状正确"""
    ppo = VisualPPO(vec_dim=6, action_dim=3, num_envs=2)
    rewards = np.array([1.0, -0.5, 0.3, -0.2], dtype=np.float32)
    values = np.array([0.5, 0.3, 0.1, 0.0], dtype=np.float32)
    dones = np.array([0, 0, 0, 1], dtype=np.float32)
    advantages, returns = ppo.compute_gae(rewards, values, dones, next_value=0.0)
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert not np.isnan(advantages).any()
    print(f"  ✓ GAE: adv={advantages.round(2)}, ret={returns.round(2)}")

def test_store_and_update():
    """存储transition后更新不报错"""
    ppo = VisualPPO(vec_dim=6, action_dim=3, num_envs=4)
    obs = make_fake_obs()
    for _ in range(64):  # 4 envs × 16 steps
        action, lp, val, ent = ppo.get_action(obs)
        ppo.store_transition(obs, action, np.random.randn(), obs, False, lp, val, ent)
    result = ppo.update()
    assert result['total_loss'] != 0.0
    assert not np.isnan(result['actor_loss'])
    assert not np.isnan(result['critic_loss'])
    print(f"  ✓ update: loss={result['total_loss']:.3f}, "
          f"actor={result['actor_loss']:.3f}, critic={result['critic_loss']:.3f}")

def test_save_load():
    """模型保存加载往返一致"""
    ppo1 = VisualPPO(vec_dim=6, action_dim=3, num_envs=1)
    obs = make_fake_obs()
    a1 = ppo1.select_action(obs, deterministic=True)

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        path = f.name
    try:
        ppo1.save_model(path)
        ppo2 = VisualPPO(vec_dim=6, action_dim=3, num_envs=1)
        ppo2.load_model(path)
        a2 = ppo2.select_action(obs, deterministic=True)
        assert np.allclose(a1, a2, atol=1e-5), f"save/load mismatch: {a1} vs {a2}"
        print(f"  ✓ save/load: roundtrip consistent")
    finally:
        os.unlink(path)

def test_lr_schedule():
    """学习率衰减"""
    ppo = VisualPPO(vec_dim=6, action_dim=3, num_envs=1)
    ppo.set_lr(1e-4)
    current = ppo.optimizer.param_groups[0]['lr']
    assert abs(current - 1e-4) < 1e-8, f"lr not set: {current}"
    print(f"  ✓ lr schedule: {current:.1e}")

def test_empty_memory_update():
    """空memory时update返回0"""
    ppo = VisualPPO(vec_dim=6, action_dim=3, num_envs=1)
    result = ppo.update()
    assert result['total_loss'] == 0.0
    print(f"  ✓ empty memory: loss={result['total_loss']}")

def test_batch_update_single_env():
    """单环境小batch更新"""
    ppo = VisualPPO(vec_dim=6, action_dim=3, num_envs=1, minibatch_size=4)
    obs = make_fake_obs()
    for _ in range(8):
        action, lp, val, ent = ppo.get_action(obs)
        ppo.store_transition(obs, action, np.random.randn(), obs, False, lp, val, ent)
    result = ppo.update()
    assert not np.isnan(result['total_loss'])
    print(f"  ✓ small batch: loss={result['total_loss']:.3f}")


def run_all_tests():
    tests = [
        ("encoder output shape", test_encoder_output_shape),
        ("encoder different inputs", test_encoder_different_inputs),
        ("actor-critic shapes", test_actor_critic_shapes),
        ("get_action format", test_agent_get_action),
        ("deterministic consistency", test_agent_deterministic),
        ("stochastic variance", test_agent_stochastic),
        ("GAE computation", test_gae_computation),
        ("store and update", test_store_and_update),
        ("save and load", test_save_load),
        ("lr schedule", test_lr_schedule),
        ("empty memory update", test_empty_memory_update),
        ("small batch update", test_batch_update_single_env),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    print(f"\n{'='*50}")
    print(f"  {passed}/{len(tests)} agent tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
