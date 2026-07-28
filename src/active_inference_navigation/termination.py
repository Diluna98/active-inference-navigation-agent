"""Reusable hardware-independent termination conditions."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Observation


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
