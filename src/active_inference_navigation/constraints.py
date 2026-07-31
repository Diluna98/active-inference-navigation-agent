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
    """Expose only actions inside the arena and outside configured blocked cells."""

    geometry: GridGeometry
    blocked_cells: frozenset[tuple[int, int]] = frozenset()

    def __post_init__(self) -> None:
        blocked = frozenset(
            (int(column), int(row)) for column, row in self.blocked_cells
        )
        for column, row in blocked:
            if not (0 <= column < self.geometry.columns and 0 <= row < self.geometry.rows):
                raise ValueError("Blocked cells must lie inside the configured grid.")
        object.__setattr__(self, "blocked_cells", blocked)

    def allowed_actions(self, observation: Observation) -> tuple[NavigationAction, ...]:
        """Return cardinal actions valid at the observed arena position."""

        current_cell = self.geometry.metric_to_grid(observation.x, observation.y)
        allowed = []
        for action in CARDINAL_ACTIONS:
            try:
                target_cell = self.geometry.target_cell(current_cell, action)
            except ValueError:
                continue
            if target_cell in self.blocked_cells:
                continue
            allowed.append(action)
        return tuple(allowed)
