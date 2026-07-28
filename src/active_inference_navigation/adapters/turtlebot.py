"""Closed-loop TurtleBot grid action execution using odometry and Twist."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import atan2, hypot, pi
from time import monotonic, sleep
from typing import Any, Protocol

from ..frame import ArenaFrameTransform
from ..geometry import GridGeometry
from ..interfaces import ActionExecutionError
from ..models import NavigationAction


@dataclass(frozen=True)
class RobotPose:
    """Planar robot pose obtained from odometry."""

    x: float
    y: float
    yaw: float


class VelocityPublisher(Protocol):
    """Publish planar velocity commands."""

    def publish_velocity(self, linear: float, angular: float) -> None:
        """Publish forward and angular velocity in SI units."""


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the interval ``[-pi, pi)``."""

    return (angle + pi) % (2.0 * pi) - pi


def proportional_angular_velocity(
    yaw_error: float,
    maximum_speed: float,
    *,
    gain: float = 2.0,
) -> float:
    """Return a bounded command that slows as heading error approaches zero."""

    return max(-maximum_speed, min(maximum_speed, gain * yaw_error))


@dataclass
class TurtleBotActionExecutor:
    """Rotate then translate one grid cell using closed-loop odometry."""

    geometry: GridGeometry
    pose_provider: Callable[[], RobotPose]
    velocity_publisher: VelocityPublisher
    frame_transform: ArenaFrameTransform = field(default_factory=ArenaFrameTransform)
    linear_speed: float = 0.15
    angular_speed: float = 0.5
    position_tolerance: float = 0.02
    yaw_tolerance: float = 0.03
    control_period: float = 0.05
    action_timeout: float = 30.0
    clock: Callable[[], float] = monotonic
    sleeper: Callable[[float], None] = sleep
    shutdown_requested: Callable[[], bool] = lambda: False
    target_position: tuple[float, float] | None = field(default=None, init=False)
    _target_yaw: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if min(
            self.linear_speed,
            self.angular_speed,
            self.position_tolerance,
            self.yaw_tolerance,
            self.control_period,
            self.action_timeout,
        ) <= 0.0:
            raise ValueError("Motion settings must all be positive.")

    def execute(self, action: NavigationAction) -> None:
        """Validate an action and calculate its metric displacement target."""

        if not isinstance(action, NavigationAction):
            raise TypeError("TurtleBotActionExecutor requires a NavigationAction.")
        current_pose = self.pose_provider()
        current_arena = self.frame_transform.position_to_arena(
            current_pose.x,
            current_pose.y,
        )
        current_cell = self.geometry.metric_to_grid(*current_arena)
        target_cell = self.geometry.target_cell(current_cell, action)
        delta_x, delta_y = action.cell_delta
        if delta_x == 0 and delta_y == 0:
            self.target_position = (current_pose.x, current_pose.y)
            self._target_yaw = current_pose.yaw
        else:
            target_arena = self.geometry.grid_to_metric(target_cell)
            self.target_position = self.frame_transform.position_to_odom(*target_arena)
            self._target_yaw = atan2(
                self.target_position[1] - current_pose.y,
                self.target_position[0] - current_pose.x,
            )

    def wait_for_completion(self) -> None:
        """Rotate, translate, and stop; stop immediately on every failure."""

        if self.target_position is None or self._target_yaw is None:
            raise ActionExecutionError("No navigation action is pending.")
        deadline = self.clock() + self.action_timeout
        try:
            if self._distance_to_target(self.pose_provider()) <= self.position_tolerance:
                self.stop()
                return
            self._rotate_to_target(deadline)
            self.stop()
            self._move_to_target(deadline)
            self.stop()
            self._rotate_to_target(deadline)
            self.stop()
        except Exception as error:
            self.stop()
            if isinstance(error, ActionExecutionError):
                raise
            raise ActionExecutionError("TurtleBot action execution failed.") from error
        finally:
            self.target_position = None
            self._target_yaw = None

    def stop(self) -> None:
        """Command zero velocity."""

        self.velocity_publisher.publish_velocity(0.0, 0.0)

    def _check_running(self, deadline: float) -> None:
        if self.shutdown_requested():
            raise ActionExecutionError("Robot shutdown requested during action.")
        if self.clock() > deadline:
            raise ActionExecutionError("Robot action timed out.")

    def _rotate_to_target(self, deadline: float) -> None:
        assert self._target_yaw is not None
        while True:
            self._check_running(deadline)
            pose = self.pose_provider()
            error = normalize_angle(self._target_yaw - pose.yaw)
            if abs(error) <= self.yaw_tolerance:
                return
            angular = proportional_angular_velocity(error, self.angular_speed)
            self.velocity_publisher.publish_velocity(0.0, angular)
            self.sleeper(self.control_period)

    def _move_to_target(self, deadline: float) -> None:
        assert self.target_position is not None
        while True:
            self._check_running(deadline)
            pose = self.pose_provider()
            if self._distance_to_target(pose) <= self.position_tolerance:
                return
            desired_yaw = atan2(
                self.target_position[1] - pose.y,
                self.target_position[0] - pose.x,
            )
            yaw_error = normalize_angle(desired_yaw - pose.yaw)
            angular = proportional_angular_velocity(yaw_error, self.angular_speed)
            self.velocity_publisher.publish_velocity(self.linear_speed, angular)
            self.sleeper(self.control_period)

    def _distance_to_target(self, pose: RobotPose) -> float:
        assert self.target_position is not None
        return hypot(self.target_position[0] - pose.x, self.target_position[1] - pose.y)


class RosTwistPublisher:
    """Adapt an rclpy publisher to the technology-neutral velocity interface."""

    def __init__(self, publisher: Any) -> None:
        try:
            from geometry_msgs.msg import Twist
        except ImportError as error:
            raise RuntimeError("ROS 2 geometry_msgs is required for Twist commands.") from error
        self._publisher = publisher
        self._twist_type = Twist

    def publish_velocity(self, linear: float, angular: float) -> None:
        """Publish a ``geometry_msgs/msg/Twist`` command."""

        message = self._twist_type()
        message.linear.x = float(linear)
        message.angular.z = float(angular)
        self._publisher.publish(message)


@dataclass
class OdometryPoseStore:
    """Thread-safe-enough latest planar pose store for rclpy callbacks."""

    pose: RobotPose | None = None

    def callback(self, message: Any) -> None:
        """Update pose from a ``nav_msgs/msg/Odometry``-compatible message."""

        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        sin_yaw = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cos_yaw = 1.0 - 2.0 * (orientation.y**2 + orientation.z**2)
        self.pose = RobotPose(float(position.x), float(position.y), atan2(sin_yaw, cos_yaw))

    def read(self) -> RobotPose:
        """Return the latest pose or fail clearly before odometry arrives."""

        if self.pose is None:
            raise ActionExecutionError("No odometry pose is available.")
        return self.pose
