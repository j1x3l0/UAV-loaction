"""Read-only observation bridge: PX4 telemetry -> 3DGS depth + policy vec.

Reads ``LOCAL_POSITION_NED`` + ``ATTITUDE`` from a MAVLink link (or a recorded
telemetry JSON) and produces ``{depth: (64, 64, 1), vec: (6,)}`` observations
that match ``VisualPPO`` training. This module is strictly read-only:

- it never sends MAVLink control (no offboard, setpoint, arm/disarm, mode);
- it loads no policy and performs no inference;
- camera heading comes only from PX4 ``ATTITUDE`` via ``Px4SceneAlignment``.

The renderer uses the alignment config's policy camera intrinsics
(fx/fy/cx/cy = 97.14/97.06/32/32), which is the camera model validated by the
30-pose registration gate. The retrained V3 env must use the same intrinsics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.px4_scene_alignment import Px4SceneAlignment
from envs.gs_renderer import GSplatRenderer

_POLICY_W = 64
_POLICY_H = 64
_MAX_DEPTH = 20.0
_TELEMETRY_SCHEMA = 1


@dataclass(frozen=True)
class ObservationConfig:
    """Explicit camera model for the bridge, from the alignment config."""

    width: int = _POLICY_W
    height: int = _POLICY_H
    fx: float = 97.1433427374
    fy: float = 97.0647991673
    cx: float = 32.0
    cy: float = 32.0
    max_depth: float = _MAX_DEPTH

    def __post_init__(self):
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")


def make_observation(alignment, renderer, position_ned, velocity_ned,
                     roll, pitch, yaw, target_scene,
                     obs_config: ObservationConfig = None) -> Dict[str, np.ndarray]:
    """Build one policy observation from a PX4 pose.

    Args:
        alignment: ``Px4SceneAlignment`` (maps NED -> scene).
        renderer: object exposing ``render(camera_pos, camera_c2w=...)``.
        position_ned / velocity_ned: 3-element NED arrays.
        roll / pitch / yaw: body-FRD Euler angles (radians).
        target_scene: 3-element scene-space target position.
        obs_config: explicit camera model; must match the alignment config.

    Returns:
        ``{"depth": (64, 64, 1) float32, "vec": (6,) float32}`` with
        ``vec = [vel_scene(3), target_scene - pos_scene(3)]``.
    """
    config = obs_config or ObservationConfig()
    position_ned = _finite_vector(position_ned, "position_ned")
    velocity_ned = _finite_vector(velocity_ned, "velocity_ned")
    target_scene = _finite_vector(target_scene, "target_scene")
    for name, value in (("roll", roll), ("pitch", pitch), ("yaw", yaw)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    position_scene = alignment.position_scene_from_ned(position_ned)
    velocity_scene = alignment.vector_scene_from_ned(velocity_ned)
    c2w = alignment.camera_c2w(position_ned, roll, pitch, yaw)
    depth, _rgb = renderer.render(position_scene, camera_c2w=c2w)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (config.height, config.width, 1):
        raise ValueError(
            f"renderer returned depth shape {depth.shape}, expected "
            f"({config.height}, {config.width}, 1)")
    depth = np.clip(depth, 0.1, config.max_depth)

    target_dir = target_scene - position_scene
    vec = np.concatenate([velocity_scene, target_dir]).astype(np.float32)
    return {"depth": depth, "vec": vec}


def _finite_vector(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def load_telemetry(path: str) -> List[dict]:
    """Load and validate a recorded telemetry JSON (schema 1)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("schema") != _TELEMETRY_SCHEMA:
        raise ValueError(
            f"telemetry schema must be {_TELEMETRY_SCHEMA}, got "
            f"{data.get('schema')!r}")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("telemetry has no samples")
    previous_t = -1.0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} is not an object")
        t = sample.get("t_s")
        if not isinstance(t, (int, float)) or not np.isfinite(t) or t < 0:
            raise ValueError(f"sample {index} t_s must be a non-negative finite number")
        if t < previous_t:
            raise ValueError(
                f"sample {index} t_s={t} is not monotonic (previous {previous_t})")
        previous_t = t
        for key in ("position_ned", "velocity_ned", "attitude_rpy_rad"):
            _finite_vector(sample[key], f"sample {index} {key}")
    return samples


class ReadOnlyObservationBridge:
    """Produce policy observations from PX4 telemetry without sending control."""

    def __init__(self, alignment_json: str, ply_path: str,
                 target_scene: Sequence[float], obs_config: ObservationConfig = None):
        self.config = obs_config or ObservationConfig()
        self.alignment = Px4SceneAlignment.from_json(alignment_json)
        self.target_scene = _finite_vector(target_scene, "target_scene")
        with open(alignment_json, "r", encoding="utf-8") as handle:
            self._alignment_raw = json.load(handle)
        self.renderer = GSplatRenderer(
            ply_path,
            width=self.config.width,
            height=self.config.height,
            max_depth=self.config.max_depth,
            fx=self.config.fx,
            fy=self.config.fy,
            cx=self.config.cx,
            cy=self.config.cy,
        )

    @classmethod
    def resolve_config(cls, alignment_json: str, strict_target: bool = False,
                       obs_config: ObservationConfig = None):
        """Resolve the camera config and scene-space target without loading
        the renderer, so the logic is testable without a ply file."""
        with open(alignment_json, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        camera = raw.get("camera", {})
        config = obs_config or ObservationConfig(
            width=int(camera.get("policy_resolution", [_POLICY_W, _POLICY_H])[0]),
            height=int(camera.get("policy_resolution", [_POLICY_W, _POLICY_H])[1]),
            fx=float(camera.get("fx", 97.1433427374)),
            fy=float(camera.get("fy", 97.0647991673)),
            cx=float(camera.get("cx", 32.0)),
            cy=float(camera.get("cy", 32.0)),
        )
        alignment = Px4SceneAlignment.from_json(alignment_json)
        target = raw.get("target")
        if target is not None:
            frame = target.get("frame")
            position = target.get("position_m")
            if frame not in ("NED", "scene"):
                raise ValueError(f"target.frame must be NED or scene, got {frame!r}")
            _finite_vector(position, "target.position_m")
            if frame == "NED":
                target_scene = alignment.position_scene_from_ned(position)
            else:
                target_scene = np.asarray(position, dtype=np.float64)
        else:
            if strict_target:
                raise ValueError("target not configured")
            anchor_ned = raw["anchor"]["px4_position_ned_m"]
            target_scene = alignment.position_scene_from_ned(anchor_ned)
            print("WARNING: target not configured; "
                  "defaulting to the alignment anchor")
        return config, np.asarray(target_scene, dtype=np.float64)

    @classmethod
    def from_config_defaults(cls, alignment_json: str, ply_path: str,
                             strict_target: bool = False,
                             obs_config: ObservationConfig = None):
        """Build the bridge using the alignment config's camera and target."""
        config, target_scene = cls.resolve_config(
            alignment_json, strict_target=strict_target, obs_config=obs_config)
        return cls(alignment_json, ply_path, target_scene, config)

    def observe_from_telemetry_file(self, telemetry_path: str) -> List[dict]:
        """Replay recorded telemetry into observations (no PX4, no control)."""
        samples = load_telemetry(telemetry_path)
        results = []
        for index, sample in enumerate(samples):
            observation = make_observation(
                self.alignment, self.renderer,
                sample["position_ned"], sample["velocity_ned"],
                *sample["attitude_rpy_rad"],
                self.target_scene, self.config)
            results.append({
                "index": index,
                "t_s": sample["t_s"],
                "position_ned": sample["position_ned"],
                "velocity_ned": sample["velocity_ned"],
                "attitude_rpy_rad": sample["attitude_rpy_rad"],
                "position_scene": self.alignment.position_scene_from_ned(
                    sample["position_ned"]).tolist(),
                "velocity_scene": self.alignment.vector_scene_from_ned(
                    sample["velocity_ned"]).tolist(),
                "target_scene": self.target_scene.tolist(),
                "vec": observation["vec"].tolist(),
                "observation": observation,
            })
        return results

    def observe_from_mavlink(self, client, duration_s: float, rate_hz: float,
                             telemetry_timeout_s: float = 1.0) -> List[dict]:
        """Stream observations from a live MAVLink link.

        Read-only: this method only calls ``receive_available``; it never
        sends control. ``client`` must already be ``connect()``-ed.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        period = 1.0 / rate_hz
        deadline = time.monotonic() + duration_s
        last_position = None
        last_attitude = None
        last_position_time = None
        results = []
        index = 0
        while time.monotonic() < deadline:
            started = time.monotonic()
            for message in client.receive_available():
                message_type = message.get_type()
                if message_type == "LOCAL_POSITION_NED":
                    last_position = message
                    last_position_time = started
                elif message_type == "ATTITUDE":
                    last_attitude = message
            if last_position is None:
                raise RuntimeError("no LOCAL_POSITION_NED telemetry")
            if last_attitude is None:
                raise RuntimeError("no ATTITUDE telemetry")
            if last_position_time is None \
                    or started - last_position_time > telemetry_timeout_s:
                raise TimeoutError(
                    "LOCAL_POSITION_NED telemetry became stale "
                    f"({started - (last_position_time or started):.2f}s)")

            observation = make_observation(
                self.alignment, self.renderer,
                [last_position.x, last_position.y, last_position.z],
                [last_position.vx, last_position.vy, last_position.vz],
                last_attitude.roll, last_attitude.pitch, last_attitude.yaw,
                self.target_scene, self.config)
            results.append({
                "index": index,
                "t_s": started,
                "position_ned": [last_position.x, last_position.y, last_position.z],
                "velocity_ned": [last_position.vx, last_position.vy, last_position.vz],
                "attitude_rpy_rad": [
                    last_attitude.roll, last_attitude.pitch, last_attitude.yaw],
                "position_scene": self.alignment.position_scene_from_ned(
                    [last_position.x, last_position.y, last_position.z]).tolist(),
                "velocity_scene": self.alignment.vector_scene_from_ned(
                    [last_position.vx, last_position.vy, last_position.vz]).tolist(),
                "target_scene": self.target_scene.tolist(),
                "vec": observation["vec"].tolist(),
                "observation": observation,
            })
            index += 1
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        return results


def save_observations(out_dir: str, results: List[dict]) -> str:
    """Persist observations: samples.json summary + per-sample depth .npy."""
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for row in results:
        depth = row.pop("observation")["depth"]
        depth_path = f"depth_{row['index']:05d}.npy"
        np.save(os.path.join(out_dir, depth_path), depth)
        valid = depth < _MAX_DEPTH - 1e-4
        row["depth_path"] = depth_path
        row["depth_stats"] = {
            "min": float(depth.min()),
            "max": float(depth.max()),
            "mean": float(depth.mean()),
            "invalid_ratio": float(1.0 - valid.mean()),
        }
        summary.append(row)
    with open(os.path.join(out_dir, "samples.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return os.path.abspath(out_dir)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--ply", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--telemetry", help="recorded telemetry JSON (replay)")
    group.add_argument("--connection", help="MAVLink connection string (live)")
    parser.add_argument("--strict-target", action="store_true",
                        help="fail instead of defaulting target to the anchor")
    parser.add_argument("--out-dir", default="obs_bridge_out")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    args = parser.parse_args(argv)

    bridge = ReadOnlyObservationBridge.from_config_defaults(
        args.alignment, args.ply, strict_target=args.strict_target)

    if args.telemetry:
        results = bridge.observe_from_telemetry_file(args.telemetry)
        print(f"replayed {len(results)} samples (no MAVLink, no control sent)")
    else:
        from integrations.mavlink_offboard import MavlinkOffboardClient
        client = MavlinkOffboardClient(args.connection)
        try:
            client.connect()
            print(f"connected read-only; streaming {args.rate_hz:g} Hz for "
                  f"{args.duration:g} s")
            results = bridge.observe_from_mavlink(
                client, args.duration, args.rate_hz)
        finally:
            client.close()

    saved = save_observations(args.out_dir, results)
    print(f"saved {len(results)} observations to {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
