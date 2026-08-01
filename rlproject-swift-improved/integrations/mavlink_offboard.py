"""MAVLink offboard client for PX4 SIH/SITL.

Transports the pure-conversion layer in ``integrations.px4_offboard`` over a
real MAVLink link so the SIH gate (connect -> prestream -> OFFBOARD -> arm ->
takeoff/hover -> native land -> disarm) can run without Gazebo or ROS 2. The only external dependency is
``pymavlink`` (transfer the wheel to the offline server with the SIH bundle).

The connection string is deliberately configurable. ``udpin:127.0.0.1:14540``
is PX4 SITL's default "onboard" link; it must be confirmed on the deployed
server before the gate counts as passed.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from pymavlink import mavutil

from integrations.px4_offboard import (
    OffboardHeartbeatGuard,
    Px4OffboardConfig,
    TrajectorySetpoint,
)

# SET_POSITION_TARGET_LOCAL_NED field-ignore bits (MAV_POSITION_TARGET_TYPEMASK).
_TM_X, _TM_Y, _TM_Z = 1 << 0, 1 << 1, 1 << 2
_TM_VX, _TM_VY, _TM_VZ = 1 << 3, 1 << 4, 1 << 5
_TM_AX, _TM_AY, _TM_AZ = 1 << 6, 1 << 7, 1 << 8
_TM_YAW, _TM_YAW_RATE = 1 << 10, 1 << 11

# PX4 custom_mode layout is: reserved:16, main_mode:8, sub_mode:8.
_PX4_OFFBOARD_CUSTOM_MODE = 6 << 16

_MAV_TYPE_GCS = 6
_MAV_AUTOPILOT_INVALID = 8
_MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
_MAV_MODE_FLAG_SAFETY_ARMED = 128
_MAV_SYS_STATUS_STANDBY = 4
_MAV_FRAME_LOCAL_NED = 1
_MAV_CMD_NAV_LAND = 21
_MAV_CMD_COMPONENT_ARM_DISARM = 400
_MAV_ARM_FORCE_PARAM = 21196
_MAV_LANDED_STATE_ON_GROUND = 1


@dataclass
class MavlinkOffboardConfig:
    """Wire-level defaults for the MAVLink offboard client."""

    target_system: int = 1
    target_component: int = 1
    setpoint_rate_hz: float = 20.0
    heartbeat_rate_hz: float = 2.0
    connect_timeout_s: float = 10.0


class MavlinkOffboardClient:
    """Sends PX4 offboard setpoints over a MAVLink UDP link.

    Gate sequence::

        client = MavlinkOffboardClient("udpin:127.0.0.1:14540")
        client.connect()
        target = (0.0, 0.0, -2.0)
        client.prime_offboard(target)
        client.set_offboard_and_wait(target)
        client.arm_and_wait(target)
        client.takeoff_and_hover(altitude_m=2.0, duration_s=10.0)
        client.land_and_wait()
        client.disarm_and_wait((0.0, 0.0, 0.0))
    """

    def __init__(
        self,
        connection_string: str = "udpin:127.0.0.1:14540",
        config: Optional[MavlinkOffboardConfig] = None,
    ) -> None:
        self._connection_string = connection_string
        self._config = config or MavlinkOffboardConfig()
        self._master: Optional[mavutil.mavfile] = None

    # -- connection -----------------------------------------------------

    def connect(self) -> None:
        """Open the link and wait for the target's heartbeat."""
        self._master = mavutil.mavlink_connection(self._connection_string)
        heartbeat = self._master.wait_heartbeat(
            timeout=self._config.connect_timeout_s
        )
        if heartbeat is None:
            self.close()
            raise TimeoutError(
                f"no MAVLink heartbeat within {self._config.connect_timeout_s}s"
            )
        if self._master.target_system == 0:
            self._master.target_system = self._config.target_system
        if self._master.target_component == 0:
            self._master.target_component = self._config.target_component

    def close(self) -> None:
        if self._master is not None:
            self._master.close()
            self._master = None

    # -- primitives -----------------------------------------------------

    def send_heartbeat(self) -> None:
        self._master.mav.heartbeat_send(
            _MAV_TYPE_GCS,
            _MAV_AUTOPILOT_INVALID,
            _MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            0,
            _MAV_SYS_STATUS_STANDBY,
        )

    def _command(self, command: int, param1: float, param2: float = 0.0) -> None:
        self._master.mav.command_long_send(
            self._config.target_system,
            self._config.target_component,
            command,
            0,  # confirmation
            param1, param2, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

    def arm(self, force: bool = False) -> None:
        self._command(
            _MAV_CMD_COMPONENT_ARM_DISARM,
            1.0,
            _MAV_ARM_FORCE_PARAM if force else 0.0,
        )

    def disarm(self, force: bool = False) -> None:
        self._command(
            _MAV_CMD_COMPONENT_ARM_DISARM,
            0.0,
            _MAV_ARM_FORCE_PARAM if force else 0.0,
        )

    def set_offboard_mode(self) -> None:
        self._master.mav.set_mode_send(
            self._config.target_system,
            _MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            _PX4_OFFBOARD_CUSTOM_MODE,
        )

    def land(self) -> None:
        """Request PX4's native land mode."""
        self._command(_MAV_CMD_NAV_LAND, 0.0)

    @staticmethod
    def _heartbeat_is_offboard(message) -> bool:
        return ((message.custom_mode >> 16) & 0xFF) == 6

    @staticmethod
    def _heartbeat_is_armed(message) -> bool:
        return bool(message.base_mode & _MAV_MODE_FLAG_SAFETY_ARMED)

    @staticmethod
    def _local_position_is_landed(
        message,
        altitude_tolerance_m: float = 0.15,
        vertical_speed_tolerance_m_s: float = 0.3,
    ) -> bool:
        return (
            message.get_type() == "LOCAL_POSITION_NED"
            and abs(float(message.z)) <= altitude_tolerance_m
            and abs(float(message.vz)) <= vertical_speed_tolerance_m_s
        )

    def _stream_position_until(
        self,
        position_ned: Sequence[float],
        timeout_s: float,
        predicate,
        description: str,
        retry_command=None,
    ):
        """Keep the Offboard proof-of-life alive while waiting for PX4 state."""
        deadline = time.monotonic() + timeout_s
        period = 1.0 / self._config.setpoint_rate_hz
        next_command = 0.0
        status_text = []

        while time.monotonic() < deadline:
            now = time.monotonic()
            if retry_command is not None and now >= next_command:
                retry_command()
                next_command = now + 1.0

            self.send_heartbeat()
            self.send_setpoint(position_ned=position_ned)

            while True:
                message = self._master.recv_match(blocking=False)
                if message is None:
                    break
                if message.get_type() == "STATUSTEXT":
                    status_text.append(str(message.text).strip())
                    status_text = status_text[-5:]
                if predicate(message):
                    return message

            time.sleep(period)

        detail = f"; PX4 status: {' | '.join(status_text)}" if status_text else ""
        raise TimeoutError(f"timed out waiting for {description}{detail}")

    def _discard_pending_messages(self) -> None:
        """Discard queued telemetry so a state gate only sees fresh messages."""
        while self._master.recv_match(blocking=False) is not None:
            pass

    def receive_available(self):
        """Return all telemetry currently queued on the MAVLink connection."""
        messages = []
        while True:
            message = self._master.recv_match(blocking=False)
            if message is None:
                return messages
            messages.append(message)

    def prime_offboard(
        self,
        position_ned: Sequence[float],
        duration_s: float = 2.0,
    ) -> None:
        """Publish setpoints before requesting Offboard, as required by PX4."""
        deadline = time.monotonic() + duration_s
        period = 1.0 / self._config.setpoint_rate_hz
        while time.monotonic() < deadline:
            self.send_heartbeat()
            self.send_setpoint(position_ned=position_ned)
            time.sleep(period)

    def set_offboard_and_wait(
        self,
        position_ned: Sequence[float],
        timeout_s: float = 5.0,
    ) -> None:
        """Request Offboard repeatedly and verify it from PX4 heartbeat state."""
        self._stream_position_until(
            position_ned,
            timeout_s,
            lambda msg: (
                msg.get_type() == "HEARTBEAT"
                and self._heartbeat_is_offboard(msg)
            ),
            "PX4 OFFBOARD mode",
            retry_command=self.set_offboard_mode,
        )

    def arm_and_wait(
        self,
        position_ned: Sequence[float],
        timeout_s: float = 10.0,
    ) -> None:
        """Request arming while streaming setpoints and verify the armed flag."""
        self._stream_position_until(
            position_ned,
            timeout_s,
            lambda msg: (
                msg.get_type() == "HEARTBEAT"
                and self._heartbeat_is_armed(msg)
            ),
            "PX4 armed state",
            retry_command=self.arm,
        )

    def disarm_and_wait(
        self,
        position_ned: Sequence[float],
        timeout_s: float = 5.0,
        force: bool = False,
    ) -> None:
        """Request disarming and verify that PX4 clears its armed flag."""
        self._stream_position_until(
            position_ned,
            timeout_s,
            lambda msg: (
                msg.get_type() == "HEARTBEAT"
                and not self._heartbeat_is_armed(msg)
            ),
            "PX4 disarmed state",
            retry_command=lambda: self.disarm(force=force),
        )

    def land_and_wait(
        self,
        timeout_s: float = 15.0,
    ) -> None:
        """Request native landing and wait for PX4's own land detector."""
        self._discard_pending_messages()
        deadline = time.monotonic() + timeout_s
        next_command = 0.0
        status_text = []

        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_command:
                self.land()
                next_command = now + 1.0

            self.send_heartbeat()
            while True:
                message = self._master.recv_match(blocking=False)
                if message is None:
                    break
                message_type = message.get_type()
                if message_type == "STATUSTEXT":
                    status_text.append(str(message.text).strip())
                    status_text = status_text[-5:]
                elif (
                    message_type == "EXTENDED_SYS_STATE"
                    and message.landed_state == _MAV_LANDED_STATE_ON_GROUND
                ):
                    return
                elif (
                    message_type == "HEARTBEAT"
                    and not self._heartbeat_is_armed(message)
                ):
                    return
            time.sleep(1.0 / self._config.setpoint_rate_hz)

        detail = f"; PX4 status: {' | '.join(status_text)}" if status_text else ""
        raise TimeoutError(f"timed out waiting for PX4 landed state{detail}")

    def send_setpoint(
        self,
        position_ned: Optional[Sequence[float]] = None,
        velocity_ned: Optional[Sequence[float]] = None,
        acceleration_ned: Optional[Sequence[float]] = None,
        yaw: float = float("nan"),
        yaw_rate: float = float("nan"),
    ) -> None:
        """Send SET_POSITION_TARGET_LOCAL_NED, ignoring unset fields."""
        type_mask = _TM_YAW | _TM_YAW_RATE
        if position_ned is None:
            type_mask |= _TM_X | _TM_Y | _TM_Z
        if velocity_ned is None:
            type_mask |= _TM_VX | _TM_VY | _TM_VZ
        if acceleration_ned is None:
            type_mask |= _TM_AX | _TM_AY | _TM_AZ
        pos = position_ned or (0.0, 0.0, 0.0)
        vel = velocity_ned or (0.0, 0.0, 0.0)
        acc = acceleration_ned or (0.0, 0.0, 0.0)
        self._master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (PX4 ignores for offboard setpoints)
            self._config.target_system,
            self._config.target_component,
            _MAV_FRAME_LOCAL_NED,
            type_mask,
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            acc[0], acc[1], acc[2],
            yaw, yaw_rate,
        )

    def send_trajectory_setpoint(self, setpoint: TrajectorySetpoint) -> None:
        """Stream a policy TrajectorySetpoint (acceleration-only)."""
        self.send_setpoint(acceleration_ned=setpoint.acceleration_ned)

    # -- flight helpers -------------------------------------------------

    def takeoff_and_hover(self, altitude_m: float, duration_s: float = 10.0) -> None:
        """Climb to a fixed NED altitude via position setpoint, then hold."""
        rate = self._config.setpoint_rate_hz
        steps = max(1, int(duration_s * rate))
        for _ in range(steps):
            self.send_heartbeat()
            # NED z is down; a negative z is up.
            self.send_setpoint(position_ned=(0.0, 0.0, -altitude_m))
            time.sleep(1.0 / rate)

    def stream_trajectory_setpoints(
        self, setpoints: Sequence[TrajectorySetpoint]
    ) -> None:
        """Stream heartbeat + acceleration setpoints at the configured rate."""
        guard = OffboardHeartbeatGuard(Px4OffboardConfig())
        for setpoint in setpoints:
            self.send_heartbeat()
            self.send_trajectory_setpoint(setpoint)
            guard.mark_published()
            time.sleep(guard.publish_period)
            if guard.is_stale():
                raise TimeoutError("offboard setpoint stream stalled")


def main() -> int:
    """Run the verified Offboard takeoff, hover, native-land and disarm gate."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", default="udpin:127.0.0.1:14540")
    parser.add_argument("--altitude", type=float, default=2.0)
    parser.add_argument("--hover-seconds", type=float, default=10.0)
    parser.add_argument("--prestream-seconds", type=float, default=2.0)
    parser.add_argument("--state-timeout", type=float, default=10.0)
    parser.add_argument("--land-timeout", type=float, default=15.0)
    args = parser.parse_args()

    client = MavlinkOffboardClient(args.connection)
    target = (0.0, 0.0, -args.altitude)
    armed = False

    try:
        client.connect()
        print("connected: heartbeat confirmed", flush=True)
        client.prime_offboard(target, args.prestream_seconds)
        print(
            f"offboard setpoints primed for {args.prestream_seconds}s",
            flush=True,
        )
        client.set_offboard_and_wait(target, args.state_timeout)
        print("offboard mode confirmed", flush=True)
        client.arm_and_wait(target, args.state_timeout)
        armed = True
        print("armed state confirmed", flush=True)
        client.takeoff_and_hover(args.altitude, args.hover_seconds)
        print(f"hovered {args.hover_seconds}s at {args.altitude}m", flush=True)
        client.land_and_wait(args.land_timeout)
        print("landed position confirmed", flush=True)
        client.disarm_and_wait((0.0, 0.0, 0.0), args.state_timeout)
        armed = False
        print("disarmed state confirmed", flush=True)
        return 0
    except Exception as exc:
        print(f"offboard gate failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if armed:
            try:
                client.disarm(force=True)
            except Exception:
                pass
        client.close()


if __name__ == "__main__":
    sys.exit(main())
