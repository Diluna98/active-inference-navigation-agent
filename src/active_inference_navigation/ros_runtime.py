"""ROS 2 composition root for real-world navigation."""

from __future__ import annotations

import argparse
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import Any

from .adapters.ros_observation import (
    RosObservationSource,
    attach_ros_observation_subscriptions,
)
from .adapters.turtlebot import (
    OdometryPoseStore,
    RosTwistPublisher,
    TurtleBotActionExecutor,
)
from .agent import NavigationAgentConfig, build_navigation_agent
from .config import NavigationConfig, load_cli_navigation_config
from .constraints import GridBoundaryConstraint
from .interfaces import ObservationUnavailableError
from .runtime import NavigationRuntime, NavigationRuntimeResult
from .termination import (
    NeverTermination,
    PersistentRssiTermination,
    SourceDistanceTermination,
)


def build_termination_condition(config: NavigationConfig) -> Any:
    """Build the configured hardware-independent stopping rule."""

    termination = config.termination
    if termination.provider == "persistent_rssi":
        return PersistentRssiTermination(
            minimum_rssi=termination.rssi_threshold,
            consecutive_observations=termination.consecutive_observations,
        )
    if termination.provider == "never":
        return NeverTermination()
    if termination.provider == "source_distance":
        assert termination.source_x is not None
        assert termination.source_y is not None
        return SourceDistanceTermination(
            source_x=termination.source_x,
            source_y=termination.source_y,
            maximum_distance=termination.distance_threshold,
        )
    raise ValueError(f"Unknown termination provider: {termination.provider}")


def build_action_constraint(config: NavigationConfig) -> GridBoundaryConstraint:
    """Build arena boundaries with an optional known-source blocked cell."""

    geometry = config.grid.geometry()
    blocked_cells: frozenset[tuple[int, int]] = frozenset()
    if config.termination.provider == "source_distance":
        assert config.termination.source_x is not None
        assert config.termination.source_y is not None
        blocked_cells = frozenset(
            {
                geometry.metric_to_grid(
                    config.termination.source_x,
                    config.termination.source_y,
                )
            }
        )
    return GridBoundaryConstraint(geometry, blocked_cells=blocked_cells)


def run_ros_navigation(
    node: Any,
    config: NavigationConfig,
    *,
    planning_windows: int,
) -> NavigationRuntimeResult:
    """Compose ROS adapters and run the technology-neutral navigation runtime."""

    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
    except ImportError as error:
        raise RuntimeError("A sourced ROS 2 Python environment is required.") from error

    geometry = config.grid.geometry()
    frame_transform = config.frame.transform()
    source = RosObservationSource(
        rssi_median_window=config.sensors.rssi_median_window,
        odom_timeout=config.sensors.odom_timeout,
        rssi_timeout=config.sensors.rssi_timeout,
        position_transform=frame_transform.position_to_arena,
    )
    subscriptions = list(
        attach_ros_observation_subscriptions(
            node,
            source,
            odom_topic=config.topics.odom,
            rssi_topic=config.topics.rssi,
        )
    )
    pose_store = OdometryPoseStore()
    subscriptions.append(
        node.create_subscription(Odometry, config.topics.odom, pose_store.callback, 10)
    )
    publisher = node.create_publisher(Twist, config.topics.cmd_vel, 10)
    actuator = TurtleBotActionExecutor(
        geometry=geometry,
        frame_transform=frame_transform,
        pose_provider=pose_store.read,
        velocity_publisher=RosTwistPublisher(publisher),
        linear_speed=config.motion.linear_speed,
        angular_speed=config.motion.angular_speed,
        position_tolerance=config.motion.position_tolerance,
        yaw_tolerance=config.motion.yaw_tolerance,
        control_period=config.motion.control_period,
        action_timeout=config.motion.action_timeout,
        final_heading=config.motion.final_heading,
        settling_time=config.motion.settling_time,
        shutdown_requested=lambda: not rclpy.ok(),
    )
    inference = config.active_inference
    agent = build_navigation_agent(
        NavigationAgentConfig(
            model_size=config.grid.columns,
            model_rows=config.grid.rows,
            workspace_size=config.grid.width,
            workspace_height=config.grid.height,
            goal_resolution=inference.goal_resolution,
            temporal_horizon=inference.temporal_horizon,
            message_passing_iterations=inference.message_passing_iterations,
            policy_samples=inference.policy_samples,
            exact_state_limit=inference.exact_state_limit,
            random_seed=inference.random_seed,
            policy_workers=inference.policy_workers,
            normalized_signal_preference=inference.normalized_signal_preference,
            likelihood_provider=config.likelihood_provider,
            reference_rssi=config.rssi_likelihood.reference_rssi,
            path_loss_exponent=config.rssi_likelihood.path_loss_exponent,
            signal_sigma=config.rssi_likelihood.signal_sigma,
            minimum_calibrated_distance=(
                config.rssi_likelihood.minimum_calibrated_distance
            ),
            minimum_rssi=config.rssi_likelihood.minimum_rssi,
            maximum_rssi=config.rssi_likelihood.maximum_rssi,
            bearing_cosine_coefficient=(
                config.rssi_likelihood.bearing_cosine_coefficient
            ),
            bearing_sine_coefficient=(
                config.rssi_likelihood.bearing_sine_coefficient
            ),
        )
    )
    runtime = NavigationRuntime(
        agent=agent,
        observation_source=source,
        action_executor=actuator,
        termination_condition=build_termination_condition(config),
        action_constraint=build_action_constraint(config),
        temporal_horizon=inference.temporal_horizon,
    )
    try:
        deadline = monotonic() + max(
            config.sensors.odom_timeout,
            config.sensors.rssi_timeout,
        )
        while True:
            try:
                source.read_observation()
                break
            except ObservationUnavailableError:
                if monotonic() >= deadline:
                    raise
                sleep(0.05)
        return runtime.run(planning_windows=planning_windows)
    finally:
        actuator.stop()
        _ = subscriptions


def build_parser() -> argparse.ArgumentParser:
    """Build the ROS runtime command-line parser."""

    parser = argparse.ArgumentParser(description="Run navigation using ROS 2 sensors and Twist.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--planning-windows", type=int, default=20)
    return parser


def main() -> None:
    """Run the ROS node and spin subscriptions in a background executor."""

    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
    except ImportError as error:
        raise SystemExit("A sourced ROS 2 Python environment is required.") from error

    args = build_parser().parse_args()
    config = load_cli_navigation_config(args.config)
    rclpy.init()
    node = rclpy.create_node("active_inference_navigation")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        result = run_ros_navigation(node, config, planning_windows=args.planning_windows)
        print(f"actions completed: {len(result.actions)}")
        print(f"goal condition reached: {result.terminated}")
        if result.observations:
            final = result.observations[-1]
            print(
                "final observation: "
                f"x={final.x:.4f}, y={final.y:.4f}, rssi={final.rssi:.2f}"
            )
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
