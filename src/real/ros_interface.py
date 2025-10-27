"""ROS interface skeleton for safe deployment on the real Panda arm."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class SafetyLimits:
    joint_velocity_max: np.ndarray
    joint_position_min: np.ndarray
    joint_position_max: np.ndarray
    command_rate_hz: float = 50.0


class PandaRosInterface:
    """Placeholder interface for integrating the policy with ROS controllers."""

    def __init__(self, safety_limits: SafetyLimits, rate_limiter: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        self.safety = safety_limits
        self.rate_limiter = rate_limiter
        self.lock = threading.Lock()
        self.last_command = np.zeros_like(self.safety.joint_velocity_max)
        self.enabled = False

    def start(self) -> None:
        # TODO: Initialize ROS publishers/subscribers and safety watchdogs.
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False
        self.zero_velocity()

    def zero_velocity(self) -> None:
        self.send_command(np.zeros_like(self.last_command))

    def send_command(self, velocity_cmd: np.ndarray) -> None:
        with self.lock:
            if not self.enabled:
                return
            cmd = np.asarray(velocity_cmd, dtype=np.float32)
            cmd = np.clip(cmd, -self.safety.joint_velocity_max, self.safety.joint_velocity_max)
            if self.rate_limiter is not None:
                cmd = self.rate_limiter(cmd)
            self.last_command = cmd
            # TODO: Publish cmd to ROS topic with safety checks.

    def validate_positions(self, joint_positions: np.ndarray) -> bool:
        within_limits = np.all(joint_positions >= self.safety.joint_position_min) and np.all(joint_positions <= self.safety.joint_position_max)
        if not within_limits:
            self.stop()
        return within_limits
