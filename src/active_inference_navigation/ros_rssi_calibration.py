"""Collect raw RSSI calibration samples from ROS 2 odometry and RSSI topics.

The collection logic is ROS-independent and therefore unit-testable. ROS
message imports and subscriptions are confined to :func:`main`.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
from pathlib import Path
from threading import Lock, Thread
from time import monotonic, sleep, time
from typing import Any, TextIO

from .config import load_cli_navigation_config
from .frame import ArenaFrameTransform
from .geometry import GridGeometry
from .interfaces import ActionExecutionError
from .models import NavigationAction
from .ros_actuator_test import build_actuator, parse_action, wait_for_odometry
from .ros_return_home import action_name

CSV_FIELDNAMES = (
    "sample_index",
    "timestamp_unix_s",
    "rssi_dbm",
    "odom_x_m",
    "odom_y_m",
    "arena_x_m",
    "arena_y_m",
    "arena_yaw_rad",
    "source_x_m",
    "source_y_m",
    "horizontal_distance_m",
    "distance_bin_m",
    "linear_speed_mps",
    "angular_speed_radps",
)

_HEADING_YAWS = {
    "positive_x": 0.0,
    "positive_y": pi / 2.0,
    "negative_x": pi,
    "negative_y": -pi / 2.0,
}

_MOVEMENT_ACTIONS = {
    "up": "positive_y",
    "u": "positive_y",
    "down": "negative_y",
    "d": "negative_y",
    "left": "negative_x",
    "l": "negative_x",
    "right": "positive_x",
    "r": "positive_x",
}


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the half-open interval ``[-pi, pi)``."""

    return (angle + pi) % (2.0 * pi) - pi


@dataclass(frozen=True)
class CalibrationPose:
    """Latest transformed robot pose and velocity measurement."""

    odom_x: float
    odom_y: float
    arena_x: float
    arena_y: float
    arena_yaw: float
    linear_speed: float
    angular_speed: float
    received_at: float


class CsvRowSink:
    """Write calibration rows immediately so interruption does not lose data."""

    def __init__(self, stream: TextIO, *, write_header: bool = True) -> None:
        self._stream = stream
        self._writer = csv.DictWriter(stream, fieldnames=CSV_FIELDNAMES)
        if write_header:
            self._writer.writeheader()
            self._stream.flush()

    def __call__(self, row: dict[str, float | int]) -> None:
        """Write and flush one accepted RSSI packet."""

        self._writer.writerow(row)
        self._stream.flush()


class RssiCalibrationCollector:
    """Pair RSSI packets with stationary, fresh, transformed odometry."""

    def __init__(
        self,
        *,
        source_x: float,
        source_y: float,
        frame_transform: ArenaFrameTransform,
        row_sink: Callable[[dict[str, float | int]], None],
        settling_time: float = 2.5,
        max_linear_speed: float = 0.01,
        max_angular_speed: float = 0.02,
        odom_timeout: float = 1.0,
        required_heading: str | None = "positive_x",
        heading_tolerance: float = 0.05,
        distance_bin_width: float = 0.05,
        clock: Callable[[], float] = monotonic,
        wall_clock: Callable[[], float] = time,
    ) -> None:
        """Configure sample acceptance and the known transmitter position."""

        if settling_time < 0.0:
            raise ValueError("settling_time must not be negative.")
        if min(max_linear_speed, max_angular_speed, odom_timeout) <= 0.0:
            raise ValueError("Speed thresholds and odom_timeout must be positive.")
        if heading_tolerance <= 0.0:
            raise ValueError("heading_tolerance must be positive.")
        if distance_bin_width <= 0.0:
            raise ValueError("distance_bin_width must be positive.")
        if required_heading is not None and required_heading not in _HEADING_YAWS:
            raise ValueError(f"Unknown required heading: {required_heading}")

        self.source_x = float(source_x)
        self.source_y = float(source_y)
        self.frame_transform = frame_transform
        self.row_sink = row_sink
        self.settling_time = float(settling_time)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.odom_timeout = float(odom_timeout)
        self.required_heading = required_heading
        self.heading_tolerance = float(heading_tolerance)
        self.distance_bin_width = float(distance_bin_width)
        self.clock = clock
        self.wall_clock = wall_clock
        self._pose: CalibrationPose | None = None
        self._stationary_since: float | None = None
        self._sample_count = 0
        self._batch_count = 0
        self._batch_remaining: int | None = None
        self._lock = Lock()

    @property
    def sample_count(self) -> int:
        """Return the number of accepted RSSI packets."""

        with self._lock:
            return self._sample_count

    @property
    def batch_count(self) -> int:
        """Return the number of samples accepted in the current location batch."""

        with self._lock:
            return self._batch_count

    @property
    def batch_complete(self) -> bool:
        """Return whether the requested location batch has been collected."""

        with self._lock:
            return self._batch_remaining == 0

    def begin_batch(self, sample_count: int) -> None:
        """Start accepting exactly ``sample_count`` packets at one location."""

        if sample_count < 1:
            raise ValueError("Batch sample_count must be positive.")
        with self._lock:
            self._batch_count = 0
            self._batch_remaining = sample_count

    def odometry_callback(self, message: Any, *, received_at: float | None = None) -> None:
        """Update pose, heading, velocities, and stationary settling state."""

        timestamp = self.clock() if received_at is None else float(received_at)
        try:
            pose = message.pose.pose
            twist = message.twist.twist
            odom_x = float(pose.position.x)
            odom_y = float(pose.position.y)
            quaternion = pose.orientation
            qx = float(quaternion.x)
            qy = float(quaternion.y)
            qz = float(quaternion.z)
            qw = float(quaternion.w)
            linear_speed = hypot(float(twist.linear.x), float(twist.linear.y))
            angular_speed = abs(float(twist.angular.z))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Odometry message is missing a valid pose or twist.") from error

        odom_yaw = atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        heading_x, heading_y = self.frame_transform.vector_to_arena(
            cos(odom_yaw),
            sin(odom_yaw),
        )
        arena_yaw = atan2(heading_y, heading_x)
        arena_x, arena_y = self.frame_transform.position_to_arena(odom_x, odom_y)
        stationary = (
            linear_speed <= self.max_linear_speed
            and angular_speed <= self.max_angular_speed
        )

        with self._lock:
            if stationary:
                if self._stationary_since is None:
                    self._stationary_since = timestamp
            else:
                self._stationary_since = None
            self._pose = CalibrationPose(
                odom_x=odom_x,
                odom_y=odom_y,
                arena_x=arena_x,
                arena_y=arena_y,
                arena_yaw=arena_yaw,
                linear_speed=linear_speed,
                angular_speed=angular_speed,
                received_at=timestamp,
            )

    def rssi_callback(self, message: Any, *, received_at: float | None = None) -> bool:
        """Record an RSSI packet if the latest pose satisfies all safeguards."""

        timestamp = self.clock() if received_at is None else float(received_at)
        try:
            rssi = float(message.data)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("RSSI message does not contain a valid Float32 value.") from error

        with self._lock:
            pose = self._pose
            stationary_since = self._stationary_since
            if self._batch_remaining == 0:
                return False
            if pose is None or stationary_since is None:
                return False
            if timestamp - pose.received_at > self.odom_timeout:
                return False
            if timestamp - stationary_since < self.settling_time:
                return False
            if self.required_heading is not None:
                desired_yaw = _HEADING_YAWS[self.required_heading]
                if abs(normalize_angle(pose.arena_yaw - desired_yaw)) > self.heading_tolerance:
                    return False

            distance = hypot(pose.arena_x - self.source_x, pose.arena_y - self.source_y)
            distance_bin = (
                round(distance / self.distance_bin_width) * self.distance_bin_width
            )
            self._sample_count += 1
            self._batch_count += 1
            if self._batch_remaining is not None:
                self._batch_remaining -= 1
            row: dict[str, float | int] = {
                "sample_index": self._sample_count,
                "timestamp_unix_s": self.wall_clock(),
                "rssi_dbm": rssi,
                "odom_x_m": pose.odom_x,
                "odom_y_m": pose.odom_y,
                "arena_x_m": pose.arena_x,
                "arena_y_m": pose.arena_y,
                "arena_yaw_rad": pose.arena_yaw,
                "source_x_m": self.source_x,
                "source_y_m": self.source_y,
                "horizontal_distance_m": distance,
                "distance_bin_m": distance_bin,
                "linear_speed_mps": pose.linear_speed,
                "angular_speed_radps": pose.angular_speed,
            }
            self.row_sink(row)
            return True


def parse_movement_command(command: str) -> NavigationAction:
    """Convert an interactive cardinal direction into a navigation action."""

    normalized = command.strip().lower()
    try:
        action_name = _MOVEMENT_ACTIONS[normalized]
    except KeyError as error:
        raise ValueError("Enter up, down, left, right, or quit.") from error
    return parse_action(action_name)


def parse_target_cell(command: str) -> tuple[int, int] | None:
    """Parse ``(column,row)`` or ``column,row`` and return ``None`` otherwise."""

    match = re.fullmatch(
        r"\s*(?:\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)|(-?\d+)\s*,\s*(-?\d+))\s*",
        command,
    )
    if match is None:
        return None
    values = match.group(1, 2) if match.group(1) is not None else match.group(3, 4)
    return int(values[0]), int(values[1])


def plan_calibration_movement(
    command: str,
    current_cell: tuple[int, int],
    geometry: GridGeometry,
    *,
    blocked_cells: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[NavigationAction, ...]:
    """Plan one cardinal command or a shortest safe path to an absolute cell."""

    target_cell = parse_target_cell(command)
    if target_cell is not None:
        return plan_grid_path(
            current_cell,
            target_cell,
            geometry,
            blocked_cells=blocked_cells,
        )

    action = parse_movement_command(command)
    target_cell = geometry.target_cell(current_cell, action)
    if target_cell in blocked_cells:
        raise ValueError(f"Movement would enter blocked grid cell {target_cell}.")
    return (action,)


def plan_grid_path(
    current_cell: tuple[int, int],
    target_cell: tuple[int, int],
    geometry: GridGeometry,
    *,
    blocked_cells: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[NavigationAction, ...]:
    """Find a shortest cardinal grid path without entering blocked cells."""

    if not geometry.contains_cell(current_cell):
        raise ValueError(f"Current grid cell {current_cell} is outside the arena.")
    if not geometry.contains_cell(target_cell):
        raise ValueError(f"Target grid cell {target_cell} is outside the arena.")
    if current_cell in blocked_cells:
        raise ValueError(f"Current grid cell {current_cell} is blocked.")
    if target_cell in blocked_cells:
        raise ValueError(f"Target grid cell {target_cell} is blocked.")
    if current_cell == target_cell:
        return ()

    actions = (
        parse_action("positive_x"),
        parse_action("negative_x"),
        parse_action("positive_y"),
        parse_action("negative_y"),
    )
    frontier = deque([current_cell])
    parents: dict[
        tuple[int, int],
        tuple[tuple[int, int], NavigationAction] | None,
    ] = {current_cell: None}

    while frontier:
        cell = frontier.popleft()
        candidates: list[tuple[int, tuple[int, int], NavigationAction]] = []
        for action in actions:
            delta_x, delta_y = action.cell_delta
            neighbor = cell[0] + delta_x, cell[1] + delta_y
            if (
                not geometry.contains_cell(neighbor)
                or neighbor in blocked_cells
                or neighbor in parents
            ):
                continue
            remaining_distance = abs(target_cell[0] - neighbor[0]) + abs(
                target_cell[1] - neighbor[1]
            )
            candidates.append((remaining_distance, neighbor, action))

        for _, neighbor, action in sorted(candidates, key=lambda item: item[0]):
            parents[neighbor] = cell, action
            if neighbor == target_cell:
                path: list[NavigationAction] = []
                cursor = target_cell
                while cursor != current_cell:
                    parent = parents[cursor]
                    assert parent is not None
                    cursor, previous_action = parent
                    path.append(previous_action)
                return tuple(reversed(path))
            frontier.append(neighbor)

    raise ValueError(f"No safe grid path exists from {current_cell} to {target_cell}.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect stationary RSSI/odometry pairs while driving with teleop."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-x", type=float, required=True, help="BLE source arena x (m).")
    parser.add_argument("--source-y", type=float, required=True, help="BLE source arena y (m).")
    parser.add_argument("--output", type=Path, default=Path("rssi_calibration.csv"))
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--settling-time", type=float, default=2.5)
    parser.add_argument("--max-linear-speed", type=float, default=0.01)
    parser.add_argument("--max-angular-speed", type=float, default=0.02)
    parser.add_argument("--heading-tolerance", type=float, default=0.05)
    parser.add_argument("--distance-bin-width", type=float, default=0.05)
    parser.add_argument(
        "--samples-per-location",
        type=int,
        default=100,
        help="Accepted raw RSSI packets before asking for the next movement.",
    )
    parser.add_argument(
        "--required-heading",
        choices=("any", *tuple(_HEADING_YAWS)),
        default=None,
        help="Required arena heading; defaults to motion.final_heading from YAML.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main() -> None:
    """Run the ROS 2 calibration recorder until interrupted."""

    args = _build_parser().parse_args()
    config = load_cli_navigation_config(args.config)
    geometry = config.grid.geometry()
    frame_transform = config.frame.transform()
    source_cell = geometry.metric_to_grid(args.source_x, args.source_y)
    blocked_cells = frozenset({source_cell})
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive.")
    if args.samples_per_location < 1:
        raise ValueError("--samples-per-location must be positive.")
    if args.output.exists() and not args.append:
        raise FileExistsError(
            f"{args.output} already exists; choose another path or pass --append."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    mode = "a" if args.append else "x"

    try:
        import rclpy
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Float32
    except ImportError as error:
        raise RuntimeError("ROS 2 rclpy, nav_msgs, and std_msgs are required.") from error

    from rclpy.executors import MultiThreadedExecutor

    rclpy.init()
    node = rclpy.create_node("active_inference_rssi_calibration")
    ros_executor = MultiThreadedExecutor()
    ros_executor.add_node(node)
    spin_thread = Thread(target=ros_executor.spin, daemon=True)
    spin_thread.start()
    actuator = None
    try:
        with args.output.open(mode, newline="", encoding="utf-8") as stream:
            sink = CsvRowSink(stream, write_header=write_header)
            required_heading = (
                config.motion.final_heading
                if args.required_heading is None
                else None
                if args.required_heading == "any"
                else args.required_heading
            )
            collector = RssiCalibrationCollector(
                source_x=args.source_x,
                source_y=args.source_y,
                frame_transform=frame_transform,
                row_sink=sink,
                settling_time=args.settling_time,
                max_linear_speed=args.max_linear_speed,
                max_angular_speed=args.max_angular_speed,
                odom_timeout=config.sensors.odom_timeout,
                required_heading=required_heading,
                heading_tolerance=args.heading_tolerance,
                distance_bin_width=args.distance_bin_width,
            )

            def rssi_callback(message: Any) -> None:
                if collector.rssi_callback(message):
                    batch_count = collector.batch_count
                    if batch_count % args.progress_every == 0:
                        node.get_logger().info(
                            f"location progress: {batch_count}/"
                            f"{args.samples_per_location} samples"
                        )

            odom_subscription = node.create_subscription(
                Odometry,
                config.topics.odom,
                collector.odometry_callback,
                10,
            )
            rssi_subscription = node.create_subscription(
                Float32,
                config.topics.rssi,
                rssi_callback,
                50,
            )
            actuator, actuator_subscription = build_actuator(node, config)
            subscriptions = (
                odom_subscription,
                rssi_subscription,
                actuator_subscription,
            )
            _ = subscriptions
            wait_for_odometry(actuator)
            actuator.execute(parse_action("stay"))
            actuator.wait_for_completion()
            node.get_logger().info(
                "Interactive RSSI collection is ready. "
                f"Source arena position=({args.source_x:.3f}, {args.source_y:.3f}) m; "
                f"blocked source cell={source_cell}."
            )
            try:
                quit_requested = False
                while rclpy.ok():
                    collector.begin_batch(args.samples_per_location)
                    node.get_logger().info(
                        f"Collecting {args.samples_per_location} samples at this location. "
                        "Keep the robot still."
                    )
                    while rclpy.ok() and not collector.batch_complete:
                        sleep(0.05)
                    if not rclpy.ok():
                        break

                    node.get_logger().info(
                        f"Location complete; total saved samples: {collector.sample_count}."
                    )
                    while rclpy.ok():
                        command = input(
                            "Move [up/down/left/right], target cell [(x,y)], "
                            "or quit [q]: "
                        ).strip()
                        if command.lower() in {"q", "quit", "exit"}:
                            quit_requested = True
                            break
                        try:
                            pose = actuator.pose_provider()
                            arena_position = frame_transform.position_to_arena(
                                pose.x,
                                pose.y,
                            )
                            current_cell = geometry.metric_to_grid(*arena_position)
                            actions = plan_calibration_movement(
                                command,
                                current_cell,
                                geometry,
                                blocked_cells=blocked_cells,
                            )
                            target_cell = parse_target_cell(command)
                            if target_cell is not None:
                                print(
                                    f"Moving from cell {current_cell} to "
                                    f"{target_cell} in {len(actions)} step(s).",
                                    flush=True,
                                )
                            for step, action in enumerate(actions, start=1):
                                final_step = step == len(actions)
                                if len(actions) > 1:
                                    print(
                                        f"step {step}/{len(actions)}: "
                                        f"{action_name(action)}",
                                        flush=True,
                                    )
                                actuator.execute(
                                    action,
                                    restore_final_heading=final_step,
                                    settle_after_completion=final_step,
                                )
                                actuator.wait_for_completion()
                        except (ValueError, ActionExecutionError) as error:
                            actuator.stop()
                            print(f"Movement rejected: {error}", flush=True)
                            continue
                        break
                    if quit_requested:
                        break
            except KeyboardInterrupt:
                pass
            node.get_logger().info(
                f"Finished with {collector.sample_count} samples in {args.output}."
            )
    finally:
        if actuator is not None:
            actuator.stop()
        ros_executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
