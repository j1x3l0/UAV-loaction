"""Bounded RL-policy-to-PX4 interface smoke test.

This gate intentionally uses a synthetic constant-depth observation.  It proves
that a real VisualPPO checkpoint can drive the verified MAVLink Offboard path,
but it is not an aligned visual-navigation evaluation and must not be reported
as a V3 scientific result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from integrations.mavlink_offboard import MavlinkOffboardClient
from integrations.px4_offboard import Px4OffboardConfig, policy_action_to_setpoint


@dataclass(frozen=True)
class PolicySmokeSafety:
    """Conservative limits for an interface-only SIH flight."""

    duration_s: float = 3.0
    horizontal_acceleration_limit: float = 0.5
    vertical_acceleration_limit: float = 0.3
    horizontal_radius_limit_m: float = 1.0
    minimum_altitude_m: float = 0.35
    maximum_altitude_m: float = 1.65
    telemetry_timeout_s: float = 1.0

    def __post_init__(self):
        if not 0.1 <= self.duration_s <= 10.0:
            raise ValueError("smoke duration must be between 0.1 and 10 seconds")
        if self.horizontal_acceleration_limit <= 0:
            raise ValueError("horizontal acceleration limit must be positive")
        if self.vertical_acceleration_limit <= 0:
            raise ValueError("vertical acceleration limit must be positive")
        if self.horizontal_radius_limit_m <= 0:
            raise ValueError("horizontal radius limit must be positive")
        if not 0 < self.minimum_altitude_m < self.maximum_altitude_m:
            raise ValueError("altitude envelope is invalid")
        if self.telemetry_timeout_s <= 0:
            raise ValueError("telemetry timeout must be positive")


def ned_to_enu(vector: Sequence[float]) -> np.ndarray:
    """Convert PX4 NED ``[north, east, down]`` to ENU."""
    value = np.asarray(vector, dtype=np.float32)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("NED vector must contain three finite values")
    return np.array([value[1], value[0], -value[2]], dtype=np.float32)


def make_smoke_observation(message, depth_m: float = 5.0) -> dict:
    """Build the legacy model input using live velocity and synthetic depth."""
    if not math.isfinite(depth_m) or depth_m < 0:
        raise ValueError("depth must be a finite non-negative value")
    velocity_enu = ned_to_enu((message.vx, message.vy, message.vz))
    # A fixed unit target vector exercises all input plumbing without claiming
    # that the 3DGS scene and PX4 local origin have already been calibrated.
    vector = np.concatenate((velocity_enu, np.array([1.0, 0.0, 0.0], np.float32)))
    return {
        "depth": np.full((64, 64, 1), depth_m, dtype=np.float32),
        "vec": vector,
    }


def assert_within_envelope(message, safety: PolicySmokeSafety) -> None:
    """Abort before another policy setpoint if PX4 leaves the smoke envelope."""
    horizontal_radius = math.hypot(float(message.x), float(message.y))
    altitude = -float(message.z)
    if horizontal_radius > safety.horizontal_radius_limit_m:
        raise RuntimeError(
            f"horizontal safety envelope exceeded: {horizontal_radius:.3f} m"
        )
    if not safety.minimum_altitude_m <= altitude <= safety.maximum_altitude_m:
        raise RuntimeError(f"altitude safety envelope exceeded: {altitude:.3f} m")


def load_policy(checkpoint: str):
    """Strictly load the repository's 3-action VisualPPO checkpoint."""
    import torch
    from core.visual_ppo_agent import VisualPPO

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_data.get("action_dim") != 3:
        raise ValueError("checkpoint action_dim must be 3")
    policy = VisualPPO(vec_dim=6, action_dim=3)
    policy.model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    policy.model.eval()
    return policy


def latest_local_position(client, previous=None):
    """Drain telemetry and return the newest LOCAL_POSITION_NED message."""
    latest = previous
    for message in client.receive_available():
        if message.get_type() == "LOCAL_POSITION_NED":
            latest = message
        elif message.get_type() == "HEARTBEAT":
            if not client._heartbeat_is_offboard(message):
                raise RuntimeError("PX4 left OFFBOARD mode")
            if not client._heartbeat_is_armed(message):
                raise RuntimeError("PX4 became disarmed during policy stream")
    return latest


def run_policy_stream(
    client,
    policy,
    hold_position_ned,
    safety: PolicySmokeSafety,
    depth_m: float,
):
    """Run deterministic inference while PX4 position control remains active."""
    rate_hz = client._config.setpoint_rate_hz
    acceleration_config = Px4OffboardConfig(
        horizontal_acceleration_limit=safety.horizontal_acceleration_limit,
        vertical_acceleration_limit=safety.vertical_acceleration_limit,
        publish_rate_hz=rate_hz,
    )
    deadline = time.monotonic() + safety.duration_s
    last_position = None
    last_position_time = 0.0
    samples = []

    while time.monotonic() < deadline:
        started = time.monotonic()
        newest = latest_local_position(client, last_position)
        if newest is not last_position:
            last_position = newest
            last_position_time = started
        if last_position is None:
            raise RuntimeError("no LOCAL_POSITION_NED telemetry")
        if started - last_position_time > safety.telemetry_timeout_s:
            raise RuntimeError("LOCAL_POSITION_NED telemetry became stale")
        assert_within_envelope(last_position, safety)

        observation = make_smoke_observation(last_position, depth_m)
        action = np.asarray(
            policy.select_action(observation, deterministic=True), dtype=np.float32
        )
        setpoint = policy_action_to_setpoint(action, acceleration_config)
        client.send_heartbeat()
        client.send_setpoint(
            position_ned=hold_position_ned,
            acceleration_ned=setpoint.acceleration_ned,
        )
        samples.append({
            "t_s": started,
            "position_ned": [
                float(last_position.x),
                float(last_position.y),
                float(last_position.z),
            ],
            "velocity_ned": [
                float(last_position.vx),
                float(last_position.vy),
                float(last_position.vz),
            ],
            "action": action.tolist(),
            "acceleration_ned": list(setpoint.acceleration_ned),
        })
        remaining = 1.0 / rate_hz - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)

    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--connection", default="udpin:127.0.0.1:14540")
    parser.add_argument("--altitude", type=float, default=1.0)
    parser.add_argument("--takeoff-seconds", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--depth", type=float, default=5.0)
    parser.add_argument("--result", default="/tmp/rl_policy_offboard_smoke.json")
    args = parser.parse_args()

    safety = PolicySmokeSafety(duration_s=args.duration)
    policy = load_policy(args.checkpoint)
    client = MavlinkOffboardClient(args.connection)
    target = (0.0, 0.0, -args.altitude)
    armed = False
    samples = []
    error = None
    try:
        client.connect()
        print("connected; checkpoint loaded strictly", flush=True)
        client.prime_offboard(target, 2.0)
        client.set_offboard_and_wait(target, 10.0)
        client.arm_and_wait(target, 10.0)
        armed = True
        client.takeoff_and_hover(args.altitude, args.takeoff_seconds)
        print("position hold established; starting bounded policy stream", flush=True)
        samples = run_policy_stream(client, policy, target, safety, args.depth)
        print(f"policy stream completed: {len(samples)} setpoints", flush=True)
        client.land_and_wait(15.0)
        client.disarm_and_wait((0.0, 0.0, 0.0), 10.0)
        armed = False
        print("native landing and disarm confirmed", flush=True)
        return 0
    except Exception as exc:
        error = str(exc)
        print(f"policy smoke failed: {exc}", file=sys.stderr, flush=True)
        if armed:
            try:
                client.land_and_wait(15.0)
                client.disarm_and_wait((0.0, 0.0, 0.0), 10.0)
                armed = False
            except Exception as landing_exc:
                print(f"recovery landing failed: {landing_exc}", file=sys.stderr)
        return 1
    finally:
        if armed:
            try:
                client.disarm(force=True)
            except Exception:
                pass
        client.close()
        report = {
            "gate": "legacy_checkpoint_interface_smoke_only",
            "aligned_v3_result": False,
            "checkpoint": os.path.abspath(args.checkpoint),
            "safety": asdict(safety),
            "sample_count": len(samples),
            "error": error,
            "samples": samples,
        }
        with open(args.result, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
        print(f"result written: {args.result}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
