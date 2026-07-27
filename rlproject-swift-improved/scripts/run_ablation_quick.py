#!/usr/bin/env python3
"""Run the real-gsplat ablation protocol with explicit inputs."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    command = [
        sys.executable, "scripts/eval_ablation.py",
        "--model", args.model,
        "--renderer", "gsplat",
        "--ply", args.ply,
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
