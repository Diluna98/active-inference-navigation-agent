"""Typed configuration loaded from YAML without ROS dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from .frame import ArenaFrameTransform
from .geometry import GridGeometry


@dataclass(frozen=True)
class GridConfig:
    """Grid and metric arena dimensions."""

    columns: int = 20
    rows: int = 20
    width: float = 7.0
    height: float = 7.0
    origin_x: float = 0.0
    origin_y: float = 0.0

    def geometry(self) -> GridGeometry:
        """Build the shared coordinate converter."""

        return GridGeometry(**self.__dict__)


@dataclass(frozen=True)
class TopicConfig:
    """ROS topic names kept at the composition boundary."""

    odom: str = "/tb4_08/odom"
    rssi: str = "/tb4_08/rssi"
    cmd_vel: str = "/tb4_08/cmd_vel"

    def __post_init__(self) -> None:
        if not all(topic.startswith("/") for topic in (self.odom, self.rssi, self.cmd_vel)):
            raise ValueError("Topic names must be absolute and start with '/'.")


@dataclass(frozen=True)
class FrameConfig:
    """Mapping from raw odometry axes into continuous arena coordinates."""

    arena_x_from_odom: str = "-y"
    arena_y_from_odom: str = "+x"
    odom_zero_arena_x: float = 0.175
    odom_zero_arena_y: float = 0.175

    def transform(self) -> ArenaFrameTransform:
        """Build the shared bidirectional coordinate transform."""

        return ArenaFrameTransform(**self.__dict__)


@dataclass(frozen=True)
class ActiveInferenceConfig:
    """Inference and source-state settings for the navigation agent."""

    goal_resolution: int = 10
    temporal_horizon: int = 1
    message_passing_iterations: int = 5
    policy_samples: int = 200
    exact_state_limit: int = 100
    random_seed: int = 0
    policy_workers: int = 1
    normalized_signal_preference: bool = False

    def __post_init__(self) -> None:
        if min(
            self.goal_resolution,
            self.temporal_horizon,
            self.message_passing_iterations,
            self.policy_samples,
            self.exact_state_limit,
            self.policy_workers,
        ) < 1:
            raise ValueError("Active Inference dimensions and iteration counts must be positive.")


@dataclass(frozen=True)
class ExperimentConfig:
    """Repeatable trial start location in movement-grid coordinates."""

    start_column: int = 0
    start_row: int = 0

    def validate_for(self, grid: GridConfig) -> None:
        """Validate the configured start cell against the movement grid."""

        if not (0 <= self.start_column < grid.columns):
            raise ValueError("Experiment start_column is outside the movement grid.")
        if not (0 <= self.start_row < grid.rows):
            raise ValueError("Experiment start_row is outside the movement grid.")


@dataclass(frozen=True)
class TerminationConfig:
    """Real-trial stopping rule applied to aggregated observations."""

    provider: str = "persistent_rssi"
    rssi_threshold: float = -62.0
    consecutive_observations: int = 3

    def __post_init__(self) -> None:
        if self.provider not in {"persistent_rssi", "never"}:
            raise ValueError(f"Unknown termination provider: {self.provider}")
        if self.consecutive_observations < 1:
            raise ValueError("consecutive_observations must be positive.")


@dataclass(frozen=True)
class SensorConfig:
    """RSSI aggregation and sensor freshness settings."""

    rssi_median_window: int = 5
    odom_timeout: float = 1.0
    rssi_timeout: float = 1.0

    def __post_init__(self) -> None:
        if self.rssi_median_window < 1:
            raise ValueError("RSSI median window must be positive.")
        if self.odom_timeout <= 0.0 or self.rssi_timeout <= 0.0:
            raise ValueError("Sensor timeouts must be positive.")


@dataclass(frozen=True)
class MotionConfig:
    """Closed-loop actuator settings."""

    linear_speed: float = 0.15
    angular_speed: float = 0.3
    position_tolerance: float = 0.02
    yaw_tolerance: float = 0.03
    control_period: float = 0.05
    action_timeout: float = 30.0
    final_heading: str | None = "positive_x"
    settling_time: float = 6.0

    def __post_init__(self) -> None:
        if min(
            self.linear_speed,
            self.angular_speed,
            self.position_tolerance,
            self.yaw_tolerance,
            self.control_period,
            self.action_timeout,
        ) <= 0.0:
            raise ValueError("Motion speeds, tolerances, periods, and timeout must be positive.")
        if self.settling_time < 0.0:
            raise ValueError("Motion settling_time must not be negative.")
        valid_headings = {"positive_x", "negative_x", "positive_y", "negative_y"}
        if self.final_heading is not None and self.final_heading not in valid_headings:
            raise ValueError(
                "final_heading must be positive_x, negative_x, positive_y, "
                "negative_y, or null."
            )


@dataclass(frozen=True)
class RssiLikelihoodConfig:
    """Parameters for the calibrated median-aggregated dBm likelihood."""

    reference_rssi: float = -63.109
    path_loss_exponent: float = 3.104
    signal_sigma: float = 7.0
    minimum_calibrated_distance: float = 0.35
    minimum_rssi: float = -95.0
    maximum_rssi: float = -25.0
    bearing_cosine_coefficient: float = 4.761
    bearing_sine_coefficient: float = -9.065

    def __post_init__(self) -> None:
        if min(
            self.path_loss_exponent,
            self.signal_sigma,
            self.minimum_calibrated_distance,
        ) <= 0.0:
            raise ValueError("Calibrated RSSI model scale parameters must be positive.")
        if self.minimum_rssi >= self.maximum_rssi:
            raise ValueError("minimum_rssi must be lower than maximum_rssi.")


@dataclass(frozen=True)
class NavigationConfig:
    """Complete technology-neutral navigation configuration."""

    grid: GridConfig = field(default_factory=GridConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    active_inference: ActiveInferenceConfig = field(default_factory=ActiveInferenceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    topics: TopicConfig = field(default_factory=TopicConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    rssi_likelihood: RssiLikelihoodConfig = field(default_factory=RssiLikelihoodConfig)
    likelihood_provider: str = "bearing_calibrated_dbm"

    def __post_init__(self) -> None:
        self.grid.geometry()
        self.frame.transform()
        self.experiment.validate_for(self.grid)
        if not self.likelihood_provider.strip():
            raise ValueError("likelihood_provider must not be empty.")


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{key}' must be a mapping.")
    return value


def _parse_navigation_config(data: Any) -> NavigationConfig:
    """Build typed settings from decoded YAML data."""

    if not isinstance(data, dict):
        raise TypeError("Navigation configuration must be a YAML mapping.")
    return NavigationConfig(
        grid=GridConfig(**_section(data, "grid")),
        frame=FrameConfig(**_section(data, "frame")),
        active_inference=ActiveInferenceConfig(**_section(data, "active_inference")),
        experiment=ExperimentConfig(**_section(data, "experiment")),
        termination=TerminationConfig(**_section(data, "termination")),
        topics=TopicConfig(**_section(data, "topics")),
        sensors=SensorConfig(**_section(data, "sensors")),
        motion=MotionConfig(**_section(data, "motion")),
        rssi_likelihood=RssiLikelihoodConfig(**_section(data, "rssi_likelihood")),
        likelihood_provider=str(data.get("likelihood_provider", "bearing_calibrated_dbm")),
    )


def load_navigation_config(path: str | Path) -> NavigationConfig:
    """Load and validate navigation settings from a YAML file."""

    with Path(path).open(encoding="utf-8") as stream:
        return _parse_navigation_config(yaml.safe_load(stream) or {})


def load_default_navigation_config() -> NavigationConfig:
    """Load the default YAML bundled in the installed Python package."""

    resource = files("active_inference_navigation.resources").joinpath("navigation.yaml")
    return _parse_navigation_config(yaml.safe_load(resource.read_text(encoding="utf-8")) or {})


def load_cli_navigation_config(path: Path | None) -> NavigationConfig:
    """Load an explicit YAML path or fall back to the packaged default."""

    return load_default_navigation_config() if path is None else load_navigation_config(path)
