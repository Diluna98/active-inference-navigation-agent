"""Reusable hardware-independent termination conditions."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .models import Observation

_BODY_DIRECTIONS = {
    "positive_x": (1.0, 0.0),
    "negative_x": (-1.0, 0.0),
    "positive_y": (0.0, 1.0),
    "negative_y": (0.0, -1.0),
}


class NeverTermination:
    """Continue until the runtime's configured planning limit."""

    def is_met(self, observation: Observation) -> bool:
        """Always return false."""

        return False


@dataclass(frozen=True)
class RssiThresholdTermination:
    """Stop when the aggregated RSSI reaches a configured threshold."""

    minimum_rssi: float

    def is_met(self, observation: Observation) -> bool:
        """Return whether RSSI is at or above the threshold."""

        return observation.rssi >= self.minimum_rssi


@dataclass
class PersistentRssiTermination:
    """Require several consecutive high-RSSI observations before stopping."""

    minimum_rssi: float
    consecutive_observations: int = 3
    _consecutive_hits: int = 0

    def __post_init__(self) -> None:
        if self.consecutive_observations < 1:
            raise ValueError("consecutive_observations must be positive.")

    def is_met(self, observation: Observation) -> bool:
        """Update the hit count and return whether persistence is satisfied."""

        if observation.rssi >= self.minimum_rssi:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0
        return self._consecutive_hits >= self.consecutive_observations


@dataclass(frozen=True)
class SourceDistanceTermination:
    """Stop within a configured metric distance of a known evaluation source."""

    source_x: float
    source_y: float
    maximum_distance: float

    def __post_init__(self) -> None:
        if self.maximum_distance <= 0.0:
            raise ValueError("maximum_distance must be positive.")

    def is_met(self, observation: Observation) -> bool:
        """Return whether odometry places the robot close enough to the source."""

        return (
            hypot(observation.x - self.source_x, observation.y - self.source_y)
            <= self.maximum_distance
        )


@dataclass(frozen=True)
class SourceFootprintTermination:
    """Stop at a safe standoff boundary around an obstacle-mounted source."""

    source_x: float
    source_y: float
    body_direction: str
    transmitter_radius: float
    navigation_robot_radius: float
    safety_clearance: float

    def __post_init__(self) -> None:
        if self.body_direction not in _BODY_DIRECTIONS:
            raise ValueError(
                "body_direction must be positive_x, negative_x, "
                "positive_y, or negative_y."
            )
        if self.transmitter_radius <= 0.0 or self.navigation_robot_radius <= 0.0:
            raise ValueError("Robot footprint radii must be positive.")
        if self.safety_clearance < 0.0:
            raise ValueError("Footprint safety_clearance must not be negative.")

    @property
    def transmitter_center(self) -> tuple[float, float]:
        """Return the body center when the source is at the midpoint of its edge."""

        direction_x, direction_y = _BODY_DIRECTIONS[self.body_direction]
        return (
            self.source_x + direction_x * self.transmitter_radius,
            self.source_y + direction_y * self.transmitter_radius,
        )

    @property
    def safe_center_distance(self) -> float:
        """Return the minimum safe separation between the two robot centers."""

        return (
            self.transmitter_radius
            + self.navigation_robot_radius
            + self.safety_clearance
        )

    def is_position_met(self, x: float, y: float) -> bool:
        """Return whether a robot center has reached the safe goal boundary."""

        center_x, center_y = self.transmitter_center
        return hypot(x - center_x, y - center_y) <= self.safe_center_distance

    def is_met(self, observation: Observation) -> bool:
        """Return whether odometry places the robot at the safe goal boundary."""

        return self.is_position_met(observation.x, observation.y)
