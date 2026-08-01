"""Dependency-free policy-to-PX4 offboard setpoint conversion.

The current policy acts in an ENU-like world frame and outputs normalized
accelerations. PX4 ``TrajectorySetpoint`` uses NED. This module keeps that
conversion testable without requiring ROS 2 or PX4 Python packages.
"""

from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class Px4OffboardConfig:
    """Safety limits for the first PX4 SITL interface gate."""

    horizontal_acceleration_limit: float = 3.0
    vertical_acceleration_limit: float = 2.0
    publish_rate_hz: float = 20.0
    stale_after_seconds: float = 0.25

    def __post_init__(self):
        if self.horizontal_acceleration_limit <= 0:
            raise ValueError("horizontal acceleration limit must be positive")
        if self.vertical_acceleration_limit <= 0:
            raise ValueError("vertical acceleration limit must be positive")
        if self.publish_rate_hz <= 2.0:
            raise ValueError(
                "PX4 offboard proof-of-life must be published above 2 Hz")
        if self.stale_after_seconds <= 0:
            raise ValueError("stale timeout must be positive")


@dataclass(frozen=True)
class TrajectorySetpoint:
    """Transport-neutral subset of PX4 TrajectorySetpoint."""

    acceleration_ned: tuple
    position_ned: tuple = (math.nan, math.nan, math.nan)
    velocity_ned: tuple = (math.nan, math.nan, math.nan)
    yaw: float = math.nan
    yaw_speed: float = math.nan


def enu_to_ned(vector):
    """Convert ``[east, north, up]`` to ``[north, east, down]``."""
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("vector must contain three finite values")
    return np.array([vector[1], vector[0], -vector[2]], dtype=np.float32)


def policy_action_to_setpoint(action, config=None):
    """Map normalized policy acceleration to a bounded PX4 NED setpoint."""
    config = config or Px4OffboardConfig()
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (3,) or not np.all(np.isfinite(action)):
        raise ValueError("policy action must contain three finite values")
    action = np.clip(action, -1.0, 1.0)
    acceleration_enu = np.array([
        action[0] * config.horizontal_acceleration_limit,
        action[1] * config.horizontal_acceleration_limit,
        action[2] * config.vertical_acceleration_limit,
    ], dtype=np.float32)
    acceleration_ned = enu_to_ned(acceleration_enu)
    return TrajectorySetpoint(
        acceleration_ned=tuple(float(value) for value in acceleration_ned))


class OffboardHeartbeatGuard:
    """Track setpoint publication freshness for fail-safe tests."""

    def __init__(self, config=None, clock=None):
        self.config = config or Px4OffboardConfig()
        self.clock = clock or time.monotonic
        self.last_publish_time = None

    @property
    def publish_period(self):
        return 1.0 / self.config.publish_rate_hz

    def mark_published(self, timestamp=None):
        self.last_publish_time = (
            self.clock() if timestamp is None else float(timestamp))

    def should_publish(self, timestamp=None):
        now = self.clock() if timestamp is None else float(timestamp)
        return (
            self.last_publish_time is None
            or now - self.last_publish_time >= self.publish_period
        )

    def is_stale(self, timestamp=None):
        if self.last_publish_time is None:
            return True
        now = self.clock() if timestamp is None else float(timestamp)
        return (
            now - self.last_publish_time
            > self.config.stale_after_seconds
        )
