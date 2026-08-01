"""Command-line test for one explicit closed-loop TurtleBot grid action."""

from __future__ import annotations

import argparse
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import Any

from .adapters.turtlebot import (
    OdometryPoseStore,
    RosTwistPublisher,
    TurtleBotActionExecutor,
)
from .config import NavigationConfig, load_cli_navigation_config
from .interfaces import ActionExecutionError
from .models import NavigationAction
from .ros_runtime import build_termination_condition
from .termination import SourceFootprintTermination

ACTION_VALUES = {
    "stay": (0, 0),
    "negative_x": (1, 0),
    "positive_x": (2, 0),
    "negative_y": (0, 1),
    "positive_y": (0, 2),
}


def parse_action(name: str) -> NavigationAction:
    """Convert a command-line action name to a typed navigation action."""

    try:
        values = ACTION_VALUES[name]
    except KeyError as error:
        raise ValueError(f"Unknown actuator test action: {name}") from error
    return NavigationAction.from_sequence(values)


def build_actuator(node: Any, config: NavigationConfig) -> tuple[TurtleBotActionExecutor, Any]:
    """Create the production actuator and its odometry subscription."""

    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
    except ImportError as error:
        raise RuntimeError("A sourced ROS 2 Python environment is required.") from error

    pose_store = OdometryPoseStore()
    subscription = node.create_subscription(
        Odometry,
        config.topics.odom,
        pose_store.callback,
        10,
    )
    publisher = node.create_publisher(Twist, config.topics.cmd_vel, 10)
    motion = config.motion
    termination = build_termination_condition(config)
    actuator = TurtleBotActionExecutor(
        geometry=config.grid.geometry(),
        pose_provider=pose_store.read,
        velocity_publisher=RosTwistPublisher(publisher),
        frame_transform=config.frame.transform(),
        linear_speed=motion.linear_speed,
        angular_speed=motion.angular_speed,
        position_tolerance=motion.position_tolerance,
        yaw_tolerance=motion.yaw_tolerance,
        control_period=motion.control_period,
        action_timeout=motion.action_timeout,
        final_heading=motion.final_heading,
        settling_time=motion.settling_time,
        shutdown_requested=lambda: not rclpy.ok(),
        movement_stop_condition=(
            termination.is_position_met
            if isinstance(termination, SourceFootprintTermination)
            else None
        ),
    )
    return actuator, subscription


def wait_for_odometry(actuator: TurtleBotActionExecutor, timeout: float = 5.0) -> None:
    """Wait until the executor's pose provider has received odometry."""

    deadline = monotonic() + timeout
    while True:
        try:
            actuator.pose_provider()
            return
        except ActionExecutionError:
            if monotonic() >= deadline:
                raise ActionExecutionError("Timed out waiting for odometry.")
            sleep(0.05)


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit actuator-test command parser."""

    parser = argparse.ArgumentParser(
        description="Execute one explicit closed-loop TurtleBot grid action."
    )
    parser.add_argument(
        "--config",
        type=Path,
    )
    parser.add_argument("--action", choices=tuple(ACTION_VALUES), required=True)
    return parser


def main() -> None:
    """Execute one requested action and always stop the robot afterward."""

    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
    except ImportError as error:
        raise SystemExit("A sourced ROS 2 Python environment is required.") from error

    args = build_parser().parse_args()
    config = load_cli_navigation_config(args.config)
    action = parse_action(args.action)
    rclpy.init()
    node = rclpy.create_node("active_inference_actuator_test")
    ros_executor = MultiThreadedExecutor()
    ros_executor.add_node(node)
    spin_thread = Thread(target=ros_executor.spin, daemon=True)
    spin_thread.start()
    actuator = None
    try:
        actuator, subscription = build_actuator(node, config)
        wait_for_odometry(actuator)
        start = actuator.pose_provider()
        print(f"start odometry: x={start.x:.4f}, y={start.y:.4f}, yaw={start.yaw:.4f}")
        actuator.execute(action)
        actuator.wait_for_completion()
        finish = actuator.pose_provider()
        print(f"finish odometry: x={finish.x:.4f}, y={finish.y:.4f}, yaw={finish.yaw:.4f}")
        _ = subscription
    finally:
        if actuator is not None:
            actuator.stop()
        ros_executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
