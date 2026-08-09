#!/usr/bin/env python3
"""Train a behavioral-cloning policy from oracle expert data.

Supervised regression: CNN(depth) + vec -> action, minimizing MSE against
the waypoint-oracle action. Architecture mirrors VisualPPO's encoder so the
BC policy is directly comparable (same CNN+MLP, no RL).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.visual_ppo_agent import VisualEncoder


class BCPolicy(nn.Module):
    """CNN(depth) + vec -> action (behavior cloning, no critic)."""

    def __init__(self, vec_dim=6, action_dim=3, hidden_dim=128):
        super().__init__()
        self.visual_encoder = VisualEncoder(in_channels=1,
                                             feature_dim=128)
        self.mlp = nn.Sequential(
            nn.Linear(128 + vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, depth, vec):
        vis = self.visual_encoder(depth)
        combined = torch.cat([vis, vec], dim=-1)
        return torch.tanh(self.mlp(combined))  # match action space (-1,1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="npz with depth/vec/action")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--model-out", required=True)
    args = parser.parse_args()

    data = np.load(args.data)
    depth = torch.tensor(data["depth"], dtype=torch.float32)  # (N,64,64,1)
    depth = depth.permute(0, 3, 1, 2)                         # (N,1,64,64)
    vec = torch.tensor(data["vec"], dtype=torch.float32)
    action = torch.tensor(data["action"], dtype=torch.float32)

    n = len(depth)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_train = int(0.9 * n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    torch.manual_seed(args.seed)
    model = BCPolicy()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, args.batch_size):
            idx = train_idx[start:start + args.batch_size]
            out = model(depth[idx], vec[idx])
            loss = criterion(out, action[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        # validation MSE
        model.eval()
        with torch.no_grad():
            val_out = model(depth[val_idx], vec[val_idx])
            val_mse = criterion(val_out, action[val_idx]).item()
        print(f"epoch {epoch+1}/{args.epochs} | train_mse={total_loss/n_batches:.4f} "
              f"| val_mse={val_mse:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "arch": "bc_cnn_mlp"}, args.model_out)
    print(f"saved {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
