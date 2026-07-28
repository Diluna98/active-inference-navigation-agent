"""Return a robot to its configured trial start using odometry only."""

from __future__ import annotations

import argparse
from pathlib import Path
from threading import Thread

from .config import load_cli_navigation_config
from .models import NavigationAction
from .ros_actuator_test import build_actuator, wait_for_odometry

_NEGATIVE_X = NavigationAction.from_sequence((1, 0))
_POSITIVE_X = NavigationAction.from_sequence((2, 0))
_NEGATIVE_Y = NavigationAction.from_sequence((0, 1))
_POSITIVE_Y = NavigationAction.from_sequence((0, 2))


def plan_return_actions(
    current: tuple[int, int],
    target: tuple[int, int],
) -> tuple[NavigationAction, ...]:
    """Build a deterministic x-then-y Manhattan path between grid cells."""

    delta_x = target[0] - current[0]
    delta_y = target[1] - current[1]
    x_action = _POSITIVE_X if delta_x > 0 else _NEGATIVE_X
    y_action = _POSITIVE_Y if delta_y > 0 else _NEGATIVE_Y
    return (
        *((x_action,) * abs(delta_x)),
        *((y_action,) * abs(delta_y)),
    )


def action_name(action: NavigationAction) -> str:
    """Return a readable cardinal action name."""

    names = {
        (1, 0): "negative_x",
        (2, 0): "positive_x",
        (0, 1): "negative_y",
        (0, 2): "positive_y",
    }
    return names[tuple(action.as_array())]


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic return-home command parser."""

    parser = argparse.ArgumentParser(
        description="Return to the configured start cell using closed-loop odometry only."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the return path without moving the robot.",
    )
    return parser


def main() -> None:
    """Plan and optionally execute the odometry-only return path."""

    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
    except ImportError as error:
        raise SystemExit("A sourced ROS 2 Python environment is required.") from error

    args = build_parser().parse_args()
    config = load_cli_navigation_config(args.config)
    geometry = config.grid.geometry()
    frame = config.frame.transform()
    target_cell = (
        config.experiment.start_column,
        config.experiment.start_row,
    )

    rclpy.init()
    node = rclpy.create_node("active_inference_return_home")
    ros_executor = MultiThreadedExecutor()
    ros_executor.add_node(node)
    spin_thread = Thread(target=ros_executor.spin, daemon=True)
    spin_thread.start()
    actuator = None
    try:
        actuator, subscription = build_actuator(node, config)
        wait_for_odometry(actuator)
        raw_pose = actuator.pose_provider()
        arena_position = frame.position_to_arena(raw_pose.x, raw_pose.y)
        current_cell = geometry.metric_to_grid(*arena_position)
        actions = plan_return_actions(current_cell, target_cell)
        path = ", ".join(action_name(action) for action in actions) or "already home"
        print(f"current cell: {current_cell}")
        print(f"target cell: {target_cell}")
        print(f"return actions: {path}")

        if not args.dry_run:
            for step, action in enumerate(actions, start=1):
                print(f"step {step}/{len(actions)}: {action_name(action)}")
                actuator.execute(action)
                actuator.wait_for_completion()
            finish = actuator.pose_provider()
            finish_arena = frame.position_to_arena(finish.x, finish.y)
            finish_cell = geometry.metric_to_grid(*finish_arena)
            print(
                "finish odometry: "
                f"x={finish.x:.4f}, y={finish.y:.4f}, yaw={finish.yaw:.4f}"
            )
            print(
                "finish arena: "
                f"x={finish_arena[0]:.4f}, y={finish_arena[1]:.4f}, "
                f"cell={finish_cell}"
            )
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
