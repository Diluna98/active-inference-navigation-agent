"""Hardware-independent navigation observations and actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from math import isfinite

import numpy as np


class AxisAction(IntEnum):
    """Movement command for one grid axis."""

    NONE = 0
    NEGATIVE = 1
    POSITIVE = 2

    @property
    def direction(self) -> int:
        """Return the signed cell displacement represented by this action."""

        return {self.NONE: 0, self.NEGATIVE: -1, self.POSITIVE: 1}[self]


@dataclass(frozen=True)
class NavigationAction:
    """A cardinal movement command in grid coordinates."""

    x: AxisAction
    y: AxisAction

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "x", AxisAction(self.x))
            object.__setattr__(self, "y", AxisAction(self.y))
        except ValueError as error:
            raise ValueError("Action components must be 0, 1, or 2.") from error
        if self.x is not AxisAction.NONE and self.y is not AxisAction.NONE:
            raise ValueError("Diagonal navigation actions are not supported.")

    @classmethod
    def from_sequence(cls, action: Sequence[int] | np.ndarray) -> NavigationAction:
        """Create an action from the first two values returned by an agent."""

        if len(action) < 2:
            raise ValueError("A navigation action must contain x and y components.")
        try:
            return cls(AxisAction(int(action[0])), AxisAction(int(action[1])))
        except ValueError as error:
            if "Diagonal" in str(error):
                raise
            raise ValueError("Action components must be 0, 1, or 2.") from error

    @property
    def cell_delta(self) -> tuple[int, int]:
        """Return the signed displacement in grid cells."""

        return self.x.direction, self.y.direction

    def as_array(self) -> np.ndarray:
        """Return the representation expected by legacy simulation callers."""

        return np.asarray((int(self.x), int(self.y)), dtype=int)


@dataclass(frozen=True)
class Observation:
    """A position and RSSI measurement independent of its sensor technology."""

    x: float
    y: float
    rssi: float
    timestamp: float | None = None

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.rssi)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("Observation values must be finite.")
        if self.timestamp is not None and not isfinite(float(self.timestamp)):
            raise ValueError("Observation timestamp must be finite when provided.")

    @classmethod
    def from_sequence(
        cls,
        observation: Sequence[float] | np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> Observation:
        """Create an observation from an ``[x, y, rssi]`` sequence."""

        if len(observation) < 3:
            raise ValueError("An observation must contain x, y, and RSSI values.")
        return cls(
            x=float(observation[0]),
            y=float(observation[1]),
            rssi=float(observation[2]),
            timestamp=timestamp,
        )

    def as_array(self) -> np.ndarray:
        """Return the numeric representation consumed by PyAIF."""

        return np.asarray((self.x, self.y, self.rssi), dtype=float)
