"""Record a PX4 SIH hover into telemetry JSON for read-only replay.

Runs the verified offboard takeoff/hover/land gate and, during the position
hold, records ``LOCAL_POSITION_NED`` + ``ATTITUDE`` at a fixed rate. The
recorded file (schema 1) is consumed by the read-only observation bridge
replay; this script itself is a control-side recording tool.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Sequence

from integrations.mavlink_offboard import MavlinkOffboardClient


def _round_vector(values: Sequence[float]) -> List[float]:
    return [round(float(value), 6) for value in values]


def record_hover(client, target, hover_seconds: float,
                 rate_hz: float) -> List[dict]:
    """Hold position and record telemetry for ``hover_seconds``."""
    if hover_seconds <= 0:
        raise ValueError("hover_seconds must be positive")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    period = 1.0 / rate_hz
    deadline = time.monotonic() + hover_seconds
    samples = []
    while time.monotonic() < deadline:
        started = time.monotonic()
        client.send_heartbeat()
        client.send_setpoint(
            position_ned=target, acceleration_ned=(0.0, 0.0, 0.0))
        position = None
        attitude = None
        for message in client.receive_available():
            message_type = message.get_type()
            if message_type == "LOCAL_POSITION_NED":
                position = message
            elif message_type == "ATTITUDE":
                attitude = message
        if position is not None and attitude is not None:
            samples.append({
                "t_s": round(started, 4),
                "position_ned": _round_vector(
                    (position.x, position.y, position.z)),
                "velocity_ned": _round_vector(
                    (position.vx, position.vy, position.vz)),
                "attitude_rpy_rad": _round_vector(
                    (attitude.roll, attitude.pitch, attitude.yaw)),
            })
        remaining = period - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", default="udpin:127.0.0.1:14540")
    parser.add_argument("--altitude", type=float, default=1.0)
    parser.add_argument("--takeoff-seconds", type=float, default=5.0)
    parser.add_argument("--hover-seconds", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not 0.3 <= args.altitude <= 3.0:
        raise ValueError("altitude must be between 0.3 and 3.0 m")
    target = (0.0, 0.0, -args.altitude)
    client = MavlinkOffboardClient(args.connection)
    samples = []
    try:
        client.connect()
        print("connected; priming offboard", flush=True)
        client.prime_offboard(target, 2.0)
        client.set_offboard_and_wait(target, 10.0)
        client.arm_and_wait(target, 10.0)
        client.takeoff_and_hover(args.altitude, args.takeoff_seconds)
        print("position hold established; recording "
              f"{args.hover_seconds}s at {args.rate_hz} Hz", flush=True)
        samples = record_hover(client, target, args.hover_seconds, args.rate_hz)
        print(f"recorded {len(samples)} samples; landing", flush=True)
        client.land_and_wait(15.0)
        client.disarm_and_wait((0.0, 0.0, 0.0), 10.0)
        print("landed and disarmed", flush=True)
    except Exception as exc:
        print(f"recording failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        client.close()

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"schema": 1, "samples": samples}, handle, indent=1)
    print(f"saved {len(samples)} samples to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
