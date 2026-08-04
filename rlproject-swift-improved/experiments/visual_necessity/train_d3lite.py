#!/usr/bin/env python3
"""Isolated D3-lite training on the synthetic dense corridor.

Every episode is an obstacle-avoidance episode (avoidance_episode_probability
= 1.0) in the dense corridor: the straight path is always blocked by a wall,
so the policy must read the depth image to find the gaps. Target direction is
kept in the vector (the goal is known); depth is the only avoidance signal.

If the policy learns to thread the corridor with depth (baseline high) but
fails without depth (const_depth collapses), depth necessity is induced.
Does NOT modify envs/, scripts/ or core/. Delete this directory to revert.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from core.visual_ppo_agent import VisualPPO
from scripts.train_visual import make_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--collision-ply", required=True)
    parser.add_argument("--collision-radius", type=float, default=0.5)
    parser.add_argument("--alignment", default=None,
                        help="synthetic alignment config for the corridor camera "
                             "(narrow-FOV fx~97.14 + goal-yaw pose model)")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    scene_config = {
        "collision_ply_path": args.collision_ply,
        "drone_collision_radius": args.collision_radius,
        "auto_scene_bounds": True,
        "avoidance_episode_probability": 1.0,
        # Default (0.5, 0.5, 0.35) shrinks the tight corridor grid to 30
        # nodes, too few for sample_reachable_pair to find >=3m pairs.
        "scene_boundary_margin": (0.2, 0.2, 0.2),
    }
    envs = [
        make_env("clean", "gsplat", args.ply, scene_config=scene_config,
                 alignment_config=args.alignment)
        for _ in range(args.num_envs)
    ]
    ppo = VisualPPO(vec_dim=6, action_dim=3)
    observations = [env.reset()[0] for env in envs]
    step_count = 0

    for episode in range(args.episodes):
        for _ in range(args.rollout_steps):
            step_count += args.num_envs
            actions, log_probs, values, entropies = \
                ppo.get_actions_batch(observations)
            for i in range(args.num_envs):
                next_obs, reward, terminated, truncated, _ = envs[i].step(
                    actions[i])
                done = terminated or truncated
                ppo.store_transition(
                    observations[i], actions[i], reward, next_obs, done,
                    float(log_probs[i]), float(values[i]), float(entropies[i]))
                observations[i] = next_obs
                if done:
                    observations[i], _ = envs[i].reset()
        ppo.set_lr(args.lr * (1.0 - episode / args.episodes))
        result = ppo.update()
        if episode % 50 == 0 or episode == args.episodes - 1:
            print(f"Ep {episode + 1}/{args.episodes} | "
                  f"steps: {step_count} | loss: {result['total_loss']:.3f} | "
                  f"critic: {result['critic_loss']:.3f} | "
                  f"entropy: {result['entropy']:.2f}", flush=True)

    for env in envs:
        env.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    ppo.save_model(args.model_out)
    print(f"saved {args.model_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
