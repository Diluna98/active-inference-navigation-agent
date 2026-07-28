"""Adapters that expose the legacy grid environment to the generic runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..environment import GridNavigationEnvironment
from ..models import NavigationAction, Observation


@dataclass
class SimulationObservationSource:
    """Read hardware-independent observations from a simulation environment."""

    environment: GridNavigationEnvironment
    _latest: np.ndarray | None = field(default=None, init=False, repr=False)

    def reset(self) -> Observation:
        """Reset the environment and return its initial observation."""

        self._latest = self.environment.reset()
        return self.read_observation()

    def update(self, observation: np.ndarray) -> None:
        """Store the observation produced by the most recent simulation step."""

        self._latest = np.asarray(observation, dtype=float)

    def read_observation(self) -> Observation:
        """Return the latest simulator observation."""

        if self._latest is None:
            self._latest = self.environment.observe()
        return Observation.from_sequence(self._latest)


@dataclass
class SimulationActionExecutor:
    """Execute navigation actions through ``GridNavigationEnvironment``."""

    environment: GridNavigationEnvironment
    observation_source: SimulationObservationSource

    def execute(self, action: NavigationAction) -> None:
        """Apply one typed grid action to the simulator."""

        observation, _ = self.environment.step(action.as_array())
        self.observation_source.update(observation)

    def wait_for_completion(self) -> None:
        """Return immediately because simulation steps are synchronous."""


@dataclass(frozen=True)
class SimulationGoalTermination:
    """Terminate when the simulated position is sufficiently near its goal."""

    goal: tuple[float, float]
    threshold: float

    def is_met(self, observation: Observation) -> bool:
        """Return whether the observed position is within the goal threshold."""

        distance = np.linalg.norm(
            np.asarray((observation.x, observation.y), dtype=float)
            - np.asarray(self.goal, dtype=float)
        )
        return bool(distance <= self.threshold)
