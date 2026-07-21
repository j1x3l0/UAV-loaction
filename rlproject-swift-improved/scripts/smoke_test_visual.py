"""
smoke_test_visual.py — v2 管线端到端快速验证

验证内容: 环境 | Agent | 训练 | 保存/加载 | 评估 | 退化 | CSV输出
运行: python scripts/smoke_test_visual.py
预期: 所有检查通过 (≈30秒)
"""

import sys, os, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.visual_drone_env import VisualDroneEnv
from core.visual_ppo_agent import VisualPPO


def check(condition, msg):
    if condition:
        print(f"  [PASS] {msg}")
        return True
    else:
        print(f"  [FAIL] {msg}")
        return False


def main():
    print("=" * 60)
    print("v2 Pipeline Smoke Test")
    print("=" * 60)
    passed = 0
    total = 0

    # ── 1. 环境 ──
    print("\n--- 1. Environment ---")
    total += 4
    env = VisualDroneEnv()
    obs, info = env.reset(seed=42)
    passed += check('depth' in obs and 'vec' in obs, "obs dict structure")
    passed += check(obs['depth'].shape == (64, 64, 1), f"depth shape {obs['depth'].shape}")
    passed += check(obs['vec'].shape == (6,), f"vec shape {obs['vec'].shape}")

    for _ in range(10):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    passed += check(True, "10 random steps no crash")

    # ── 2. Agent ──
    print("\n--- 2. Agent ---")
    total += 5
    ppo = VisualPPO(vec_dim=6, action_dim=3, num_envs=2)
    action, lp, val, ent = ppo.get_action(obs)
    passed += check(action.shape == (3,), f"action shape {action.shape}")
    passed += check(isinstance(lp, float), "log_prob is float")
    passed += check(isinstance(val, float), "value is float")

    # Forward pass batch test
    import torch
    d = torch.randn(4, 1, 64, 64)
    v = torch.randn(4, 6)
    mean, std, value = ppo.model(d, v)
    passed += check(mean.shape == (4, 3), f"batch mean {mean.shape}")
    passed += check(value.shape == (4,), f"batch value {value.shape}")

    # ── 3. 训练微型循环 ──
    print("\n--- 3. Training micro-loop ---")
    total += 3
    env = VisualDroneEnv()
    obs, _ = env.reset(seed=42)
    for _ in range(16):  # 2 envs × 8 steps
        action, lp, val, ent = ppo.get_action(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        ppo.store_transition(obs, action, reward, next_obs, term or trunc,
                            lp, val, ent)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()

    result = ppo.update()
    passed += check(not np.isnan(result['total_loss']), "loss not NaN")
    passed += check(result['critic_loss'] > 0, "critic loss positive")
    passed += check(result['entropy'] > 0, "entropy positive")

    # ── 4. 保存/加载 ──
    print("\n--- 4. Save/Load ---")
    total += 2
    path = os.path.join(tempfile.gettempdir(), 'smoke_test_visual.pth')
    ppo.save_model(path)
    passed += check(os.path.exists(path), "model file created")

    ppo2 = VisualPPO(vec_dim=6, action_dim=3, num_envs=1)
    ppo2.load_model(path)
    a1 = ppo.select_action(obs, deterministic=True)
    a2 = ppo2.select_action(obs, deterministic=True)
    passed += check(np.allclose(a1, a2, atol=1e-5), "roundtrip consistent")
    os.unlink(path)

    # ── 5. 退化评估 ──
    print("\n--- 5. Degradation evaluation ---")
    total += 3
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envs.visual_drone_env import apply_resolution_downscale, apply_perlin_depth_noise

    depth = np.random.rand(64, 64, 1).astype(np.float32) * 10
    d1 = apply_resolution_downscale(depth, 16)
    passed += check(d1.shape == (64, 64, 1), "downscale preserves shape")

    d2 = apply_perlin_depth_noise(depth, sigma=0.05)
    passed += check(d2.min() >= 0, "noise non-negative")
    passed += check(d2.std() > 0, "noise adds variance")

    # ── 6. CSV 评估输出 ──
    print("\n--- 6. Eval CSV output ---")
    total += 2
    from scripts.eval_degradation import (
        evaluate_single_config, DEGRADATION_AXES
    )
    env = VisualDroneEnv(config={'degradation': {'resolution': 32}})
    metrics = evaluate_single_config(ppo, env, num_episodes=10)
    passed += check('success_rate' in metrics, "has success_rate")
    passed += check(isinstance(metrics['success_rate'], float), "success_rate is float")
    env.close()

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} checks passed")
    if passed == total:
        print("  ALL CHECKS PASSED - Pipeline is healthy")
    else:
        print(f"  {total - passed} FAILURES - Review above")
    print(f"{'=' * 60}")
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
