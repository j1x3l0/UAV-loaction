"""Unit tests for the transport-neutral PX4 adapter."""

import math

import numpy as np

from integrations.px4_offboard import (
    OffboardHeartbeatGuard,
    Px4OffboardConfig,
    enu_to_ned,
    policy_action_to_setpoint,
)


def test_enu_to_ned():
    converted = enu_to_ned([1.0, 2.0, 3.0])
    assert np.allclose(converted, [2.0, 1.0, -3.0])


def test_policy_action_limits_and_nan_fields():
    config = Px4OffboardConfig(
        horizontal_acceleration_limit=3.0,
        vertical_acceleration_limit=2.0,
    )
    setpoint = policy_action_to_setpoint([2.0, -0.5, 1.5], config)
    assert np.allclose(setpoint.acceleration_ned, [-1.5, 3.0, -2.0])
    assert all(math.isnan(value) for value in setpoint.position_ned)
    assert all(math.isnan(value) for value in setpoint.velocity_ned)
    assert math.isnan(setpoint.yaw)


def test_reject_invalid_action():
    try:
        policy_action_to_setpoint([0.0, math.nan, 0.0])
    except ValueError:
        return
    raise AssertionError("non-finite action was accepted")


def test_heartbeat_guard():
    guard = OffboardHeartbeatGuard(
        Px4OffboardConfig(
            publish_rate_hz=20.0,
            stale_after_seconds=0.25,
        ))
    assert guard.should_publish(timestamp=1.0)
    assert guard.is_stale(timestamp=1.0)
    guard.mark_published(timestamp=1.0)
    assert not guard.should_publish(timestamp=1.01)
    assert guard.should_publish(timestamp=1.05)
    assert not guard.is_stale(timestamp=1.25)
    assert guard.is_stale(timestamp=1.251)


if __name__ == "__main__":
    test_enu_to_ned()
    test_policy_action_limits_and_nan_fields()
    test_reject_invalid_action()
    test_heartbeat_guard()
    print("PX4 offboard adapter tests passed")
