#!/usr/bin/env python3
"""Evaluate task solvability with a shortest-path waypoint controller."""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.visual_drone_env import VisualDroneEnv


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z * math.sqrt(
            p * (1.0 - p) / total + z * z / (4.0 * total * total)
        ) / denominator
    )
    return [
        100.0 * (centre - margin),
        100.0 * (centre + margin),
    ]


def shortcut_path(geometry, path, clearance):
    """Greedily remove grid corners while retaining point-cloud clearance."""
    shortened = [path[0]]
    current = 0
    while current < len(path) - 1:
        selected = current + 1
        for candidate in range(len(path) - 1, current, -1):
            if geometry.segment_min_clearance(
                    path[current], path[candidate]) > clearance:
                selected = candidate
                break
        shortened.append(path[selected])
        current = selected
    return np.asarray(shortened, dtype=np.float32)


class WaypointController:
    """Position/velocity feedback controller following a safe waypoint path."""

    def __init__(self, path, waypoint_radius=0.35, kp=3.0, kd=2.5):
        self.path = np.asarray(path, dtype=np.float32)
        self.waypoint_radius = float(waypoint_radius)
        self.kp = float(kp)
        self.kd = float(kd)
        self.index = 1 if len(self.path) > 1 else 0

    def action(self, position, velocity, max_thrust):
        while (
            self.index < len(self.path) - 1
            and np.linalg.norm(
                self.path[self.index] - position) < self.waypoint_radius
        ):
            self.index += 1
        waypoint = self.path[self.index]
        acceleration = (
            self.kp * (waypoint - position)
            - self.kd * velocity
        )
        return np.clip(acceleration / max_thrust, -1.0, 1.0)


def summarize(counts):
    episodes = counts["episodes"]
    return {
        "episodes": episodes,
        "success": counts["success"],
        "collision": counts["collision"],
        "timeout": counts["timeout"],
        "success_rate": 100.0 * counts["success"] / episodes,
        "collision_rate": 100.0 * counts["collision"] / episodes,
        "timeout_rate": 100.0 * counts["timeout"] / episodes,
        "success_wilson_95": wilson_interval(
            counts["success"], episodes),
        "avg_path_length": float(
            np.mean(counts["path_lengths"])),
        "avg_path_efficiency": float(
            np.mean(counts["path_efficiencies"])),
        "avg_min_clearance": float(
            np.mean(counts["min_clearances"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--collision-ply", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shortcut-clearance", type=float, default=0.45)
    parser.add_argument("--waypoint-radius", type=float, default=0.35)
    parser.add_argument("--kp", type=float, default=3.0)
    parser.add_argument("--kd", type=float, default=2.5)
    args = parser.parse_args()

    env = VisualDroneEnv({
        "renderer": "gsplat",
        "ply_path": args.ply,
        "collision_ply_path": args.collision_ply,
        "auto_scene_bounds": True,
        "camera_tracks_motion": True,
        "avoidance_episode_probability": 0.5,
    })
    empty_counts = {
        "episodes": 0,
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "path_lengths": [],
        "path_efficiencies": [],
        "min_clearances": [],
    }
    strata = {
        "all": {key: (
            [] if isinstance(value, list) else value
        ) for key, value in empty_counts.items()},
        "clear": {key: (
            [] if isinstance(value, list) else value
        ) for key, value in empty_counts.items()},
        "avoidance": {key: (
            [] if isinstance(value, list) else value
        ) for key, value in empty_counts.items()},
    }
    episode_records = []

    for episode in range(args.episodes):
        _, reset_info = env.reset(seed=args.seed + episode)
        start = env.state[:3].copy()
        direct_distance = float(np.linalg.norm(env.target_pos - start))
        grid_path = env.scene_geometry.shortest_path(
            start, env.target_pos)
        path = shortcut_path(
            env.scene_geometry, grid_path, args.shortcut_clearance)
        planned_length = float(np.sum(
            np.linalg.norm(np.diff(path, axis=0), axis=1)))
        controller = WaypointController(
            path,
            waypoint_radius=args.waypoint_radius,
            kp=args.kp,
            kd=args.kd,
        )
        trajectory_length = 0.0
        min_clearance = float("inf")
        previous_position = start
        while True:
            action = controller.action(
                env.state[:3], env.state[3:6], env.max_thrust)
            _, _, terminated, truncated, info = env.step(action)
            position = env.state[:3].copy()
            trajectory_length += float(
                np.linalg.norm(position - previous_position))
            previous_position = position
            min_clearance = min(
                min_clearance,
                env._get_min_obstacle_distance(position),
            )
            if terminated or truncated:
                if info.get("reached_target"):
                    result = "success"
                elif info.get("collision"):
                    result = "collision"
                else:
                    result = "timeout"
                break
        stratum = (
            "avoidance"
            if reset_info["requires_avoidance"] else "clear"
        )
        efficiency = (
            direct_distance / trajectory_length
            if trajectory_length > 0 else 0.0
        )
        record = {
            "episode": episode,
            "seed": args.seed + episode,
            "stratum": stratum,
            "result": result,
            "direct_distance": direct_distance,
            "planned_length": planned_length,
            "trajectory_length": trajectory_length,
            "path_efficiency": efficiency,
            "min_clearance": min_clearance,
            "waypoints": len(path),
            "steps": info["step_count"],
        }
        episode_records.append(record)
        for name in ("all", stratum):
            counts = strata[name]
            counts["episodes"] += 1
            counts[result] += 1
            counts["path_lengths"].append(trajectory_length)
            counts["path_efficiencies"].append(efficiency)
            counts["min_clearances"].append(min_clearance)

    env.close()
    output = {
        "summary": {
            name: summarize(counts)
            for name, counts in strata.items()
        },
        "metadata": {
            "renderer": "gsplat",
            "ply": args.ply,
            "collision_ply": args.collision_ply,
            "episodes": args.episodes,
            "seed": args.seed,
            "shortcut_clearance": args.shortcut_clearance,
            "waypoint_radius": args.waypoint_radius,
            "kp": args.kp,
            "kd": args.kd,
        },
        "episodes": episode_records,
    }
    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "waypoint_oracle.json")
    csv_path = os.path.join(args.output, "waypoint_oracle.csv")
    with open(json_path, "w") as handle:
        json.dump(output, handle, indent=2)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(episode_records[0]))
        writer.writeheader()
        writer.writerows(episode_records)
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
