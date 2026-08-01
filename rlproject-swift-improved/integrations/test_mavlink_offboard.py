"""Unit tests for the MAVLink offboard client (no live link needed)."""

from integrations.mavlink_offboard import MavlinkOffboardClient
from integrations.px4_offboard import TrajectorySetpoint


class _MockMaster:
    """Records MAVLink send calls; enough to exercise message construction."""

    target_system = 0
    target_component = 0

    def __init__(self):
        self.sent = []
        self.mav = self

    def set_position_target_local_ned_send(self, *args):
        self.sent.append(("sptn", args))

    def heartbeat_send(self, *args):
        self.sent.append(("heartbeat", args))

    def set_mode_send(self, *args):
        self.sent.append(("mode", args))

    def command_long_send(self, *args):
        self.sent.append(("cmd", args))

    def close(self):
        pass


def _client() -> MavlinkOffboardClient:
    client = MavlinkOffboardClient()
    client._master = _MockMaster()
    return client


def test_acceleration_only_type_mask():
    client = _client()
    client.send_trajectory_setpoint(TrajectorySetpoint(acceleration_ned=(0.0, 0.0, 1.0)))
    type_mask = client._master.sent[-1][1][4]
    # Ignore position (1|2|4), velocity (8|16|32), yaw (1024), yaw_rate (2048).
    assert type_mask == 3135


def test_velocity_hover_type_mask():
    client = _client()
    client.send_setpoint(velocity_ned=(0.0, 0.0, 0.0))
    type_mask = client._master.sent[-1][1][4]
    # Ignore position (1|2|4), acceleration (64|128|256), yaw, yaw_rate.
    assert type_mask == 3527


def test_position_takeoff_type_mask():
    client = _client()
    client.send_setpoint(position_ned=(0.0, 0.0, -2.0))
    type_mask = client._master.sent[-1][1][4]
    # Ignore velocity (8|16|32), acceleration (64|128|256), yaw, yaw_rate.
    assert type_mask == 3576


def test_position_with_acceleration_feedforward_type_mask():
    client = _client()
    client.send_setpoint(
        position_ned=(0.0, 0.0, -1.0),
        acceleration_ned=(0.1, -0.2, 0.0),
    )
    type_mask = client._master.sent[-1][1][4]
    # Ignore velocity (8|16|32), yaw (1024), yaw_rate (2048) only.
    assert type_mask == 3128


def test_offboard_custom_mode():
    client = _client()
    client.set_offboard_mode()
    custom_mode = client._master.sent[-1][1][2]
    assert custom_mode == 6 << 16


def test_heartbeat_state_decoding():
    class _Heartbeat:
        base_mode = 128
        custom_mode = 6 << 16

    heartbeat = _Heartbeat()
    assert MavlinkOffboardClient._heartbeat_is_armed(heartbeat)
    assert MavlinkOffboardClient._heartbeat_is_offboard(heartbeat)


def test_landed_local_position_decoding():
    class _LocalPosition:
        z = 0.08
        vz = -0.1

        @staticmethod
        def get_type():
            return "LOCAL_POSITION_NED"

    position = _LocalPosition()
    assert MavlinkOffboardClient._local_position_is_landed(position)
    position.z = -0.5
    assert not MavlinkOffboardClient._local_position_is_landed(position)


def test_arm_disarm_command():
    client = _client()
    client.arm()
    client.disarm()
    arm = client._master.sent[-2]
    disarm = client._master.sent[-1]
    assert arm[0] == "cmd" and arm[1][2] == 400  # MAV_CMD_COMPONENT_ARM_DISARM
    assert arm[1][4] == 1.0                      # param1 = 1 -> arm
    assert disarm[1][4] == 0.0                   # param1 = 0 -> disarm


def test_land_command():
    client = _client()
    client.land()
    command = client._master.sent[-1]
    assert command[0] == "cmd"
    assert command[1][2] == 21  # MAV_CMD_NAV_LAND
