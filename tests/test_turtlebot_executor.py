from dataclasses import dataclass
from math import cos, sin

import pytest

from active_inference_navigation.adapters.turtlebot import (
    RobotPose,
    TurtleBotActionExecutor,
    normalize_angle,
    proportional_angular_velocity,
)
from active_inference_navigation.frame import ArenaFrameTransform
from active_inference_navigation.geometry import GridGeometry
from active_inference_navigation.interfaces import ActionExecutionError
from active_inference_navigation.models import NavigationAction


@dataclass
class SimulatedRobot:
    pose: RobotPose
    linear: float = 0.0
    angular: float = 0.0
    elapsed: float = 0.0

    def publish_velocity(self, linear, angular):
        self.linear = linear
        self.angular = angular

    def step(self, duration):
        yaw = self.pose.yaw + self.angular * duration
        x = self.pose.x + self.linear * cos(yaw) * duration
        y = self.pose.y + self.linear * sin(yaw) * duration
        self.pose = RobotPose(x, y, normalize_angle(yaw))
        self.elapsed += duration


def build_executor(robot, **overrides):
    return TurtleBotActionExecutor(
        geometry=GridGeometry(),
        pose_provider=lambda: robot.pose,
        velocity_publisher=robot,
        linear_speed=0.3,
        angular_speed=1.0,
        position_tolerance=0.01,
        yaw_tolerance=0.01,
        control_period=0.01,
        action_timeout=10.0,
        clock=lambda: robot.elapsed,
        sleeper=robot.step,
        **overrides,
    )


def test_executor_converts_action_to_exact_cell_displacement():
    robot = SimulatedRobot(RobotPose(0.175, 0.175, 0.0))
    executor = build_executor(robot)

    executor.execute(NavigationAction.from_sequence((2, 0)))

    assert executor.target_position == pytest.approx((0.525, 0.175))


def test_executor_corrects_residual_error_to_exact_target_cell_centre():
    geometry = GridGeometry(origin_x=-0.175, origin_y=-0.175)
    robot = SimulatedRobot(RobotPose(0.3356, -0.0008, 0.0181))
    executor = TurtleBotActionExecutor(
        geometry=geometry,
        pose_provider=lambda: robot.pose,
        velocity_publisher=robot,
    )

    executor.execute(NavigationAction.from_sequence((0, 2)))

    assert executor.target_position == pytest.approx((0.35, 0.35))


def test_executor_applies_rotated_arena_frame_to_action_target():
    robot = SimulatedRobot(RobotPose(0.0, 0.0, 0.0))
    executor = TurtleBotActionExecutor(
        geometry=GridGeometry(),
        frame_transform=ArenaFrameTransform(
            arena_x_from_odom="-y",
            arena_y_from_odom="+x",
            odom_zero_arena_x=0.175,
            odom_zero_arena_y=0.175,
        ),
        pose_provider=lambda: robot.pose,
        velocity_publisher=robot,
    )

    executor.execute(NavigationAction.from_sequence((2, 0)))
    assert executor.target_position == pytest.approx((0.0, -0.35))

    executor.execute(NavigationAction.from_sequence((0, 2)))
    assert executor.target_position == pytest.approx((0.35, 0.0))


def test_executor_rejects_grid_boundary_crossing():
    robot = SimulatedRobot(RobotPose(0.175, 0.175, 0.0))
    executor = build_executor(robot)

    with pytest.raises(ValueError, match="outside"):
        executor.execute(NavigationAction.from_sequence((1, 0)))


def test_executor_rotates_moves_and_stops_at_target():
    robot = SimulatedRobot(RobotPose(0.175, 0.175, 0.0))
    executor = build_executor(robot)
    executor.execute(NavigationAction.from_sequence((0, 2)))

    executor.wait_for_completion()

    assert (robot.pose.x, robot.pose.y) == pytest.approx((0.175, 0.525), abs=0.015)
    assert robot.linear == 0.0
    assert robot.angular == 0.0


def test_executor_restores_positive_arena_y_heading_after_x_motion():
    robot = SimulatedRobot(RobotPose(0.175, 0.175, 0.0))
    executor = build_executor(robot, final_heading="positive_y", settling_time=0.2)
    executor.execute(NavigationAction.from_sequence((2, 0)))

    executor.wait_for_completion()

    assert robot.pose.yaw == pytest.approx(3.141592653589793 / 2.0, abs=0.015)
    assert robot.elapsed >= 0.2


def test_executor_restores_positive_arena_x_heading_by_default():
    robot = SimulatedRobot(RobotPose(0.175, 0.175, 1.0))
    executor = build_executor(robot, settling_time=0.0)
    executor.execute(NavigationAction.from_sequence((0, 0)))

    executor.wait_for_completion()

    assert robot.pose.yaw == pytest.approx(0.0, abs=0.015)


def test_executor_transforms_final_arena_heading_to_odometry_frame():
    robot = SimulatedRobot(RobotPose(0.0, 0.0, 1.0))
    executor = build_executor(
        robot,
        frame_transform=ArenaFrameTransform(
            arena_x_from_odom="-y",
            arena_y_from_odom="+x",
            odom_zero_arena_x=0.175,
            odom_zero_arena_y=0.175,
        ),
        final_heading="positive_y",
        settling_time=0.0,
    )
    executor.execute(NavigationAction.from_sequence((0, 0)))

    executor.wait_for_completion()

    assert robot.pose.yaw == pytest.approx(0.0, abs=0.015)


def test_executor_stops_on_shutdown_error():
    robot = SimulatedRobot(RobotPose(0.175, 0.175, 0.0))
    executor = build_executor(robot, shutdown_requested=lambda: True)
    executor.execute(NavigationAction.from_sequence((2, 0)))

    with pytest.raises(ActionExecutionError, match="shutdown"):
        executor.wait_for_completion()

    assert robot.linear == 0.0
    assert robot.angular == 0.0


def test_normalize_angle_handles_wraparound():
    assert normalize_angle(3.0 * 3.141592653589793) == pytest.approx(-3.141592653589793)


def test_proportional_rotation_slows_near_target_and_respects_limit():
    assert proportional_angular_velocity(1.0, 0.3) == pytest.approx(0.3)
    assert proportional_angular_velocity(-1.0, 0.3) == pytest.approx(-0.3)
    assert proportional_angular_velocity(0.04, 0.3) == pytest.approx(0.08)
