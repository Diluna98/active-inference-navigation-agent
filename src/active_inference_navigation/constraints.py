"""Hardware-independent constraints on executable navigation actions."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import GridGeometry
from .models import NavigationAction, Observation

CARDINAL_ACTIONS = tuple(
    NavigationAction.from_sequence(values)
    for values in ((0, 0), (1, 0), (2, 0), (0, 1), (0, 2))
)


@dataclass(frozen=True)
class GridBoundaryConstraint:
    """Expose only actions whose target cells remain inside the arena."""

    geometry: GridGeometry

    def allowed_actions(self, observation: Observation) -> tuple[NavigationAction, ...]:
        """Return cardinal actions valid at the observed arena position."""

        current_cell = self.geometry.metric_to_grid(observation.x, observation.y)
        allowed = []
        for action in CARDINAL_ACTIONS:
            try:
                self.geometry.target_cell(current_cell, action)
            except ValueError:
                continue
            allowed.append(action)
        return tuple(allowed)
