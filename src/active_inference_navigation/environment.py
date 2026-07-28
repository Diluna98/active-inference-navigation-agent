"""Continuous RSSI grid-navigation environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import NavigationAction


@dataclass
class GridNavigationEnvironment:
    """A bounded grid world with a continuous RSSI observation."""

    model_size: int = 20
    workspace_size: float = 500.0
    start: tuple[float, float] = (487.5, 487.5)
    goal: tuple[float, float] = (212.5, 312.5)
    maximum_rssi: float = 30.0
    signal_decay: float = 0.01
    signal_noise: float = 0.0
    goal_threshold: float = 18.0
    random_seed: int = 0
    position: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.model_size < 2:
            raise ValueError("model_size must be at least two.")
        self.step_size = self.workspace_size / self.model_size
        self._rng = np.random.default_rng(self.random_seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.position = np.asarray(self.start, dtype=float).copy()
        return self.observe()

    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.position - np.asarray(self.goal, dtype=float)))

    def observe(self) -> np.ndarray:
        signal = self.maximum_rssi * np.exp(-self.signal_decay * self.distance_to_goal())
        if self.signal_noise:
            signal += self._rng.normal(0.0, self.signal_noise)
        return np.array(
            [self.position[0], self.position[1], max(0.0, signal)],
            dtype=float,
        )

    def step(self, action) -> tuple[np.ndarray, bool]:
        navigation_action = NavigationAction.from_sequence(action)
        x_action, y_action = (int(navigation_action.x), int(navigation_action.y))
        action_delta = {0: 0.0, 1: -self.step_size, 2: self.step_size}
        self.position += np.array([action_delta[x_action], action_delta[y_action]])
        self.position = np.clip(self.position, 0.0, self.workspace_size)
        done = self.distance_to_goal() <= self.goal_threshold
        return self.observe(), done

