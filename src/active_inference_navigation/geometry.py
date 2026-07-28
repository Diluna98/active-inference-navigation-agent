"""Grid and metric coordinate conversions for navigation."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

from .models import NavigationAction


@dataclass(frozen=True)
class GridGeometry:
    """Describe a rectangular metric arena discretised into grid cells.

    Metric coordinates use the arena's lower-left corner as ``(origin_x,
    origin_y)``. Grid indices are zero based. A grid position maps to the
    centre of its cell.
    """

    columns: int = 20
    rows: int = 20
    width: float = 7.0
    height: float = 7.0
    origin_x: float = 0.0
    origin_y: float = 0.0

    def __post_init__(self) -> None:
        if self.columns < 1 or self.rows < 1:
            raise ValueError("Grid dimensions must be positive.")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Arena dimensions must be positive.")

    @property
    def cell_width(self) -> float:
        """Width of one cell in metres."""

        return self.width / self.columns

    @property
    def cell_height(self) -> float:
        """Height of one cell in metres."""

        return self.height / self.rows

    def contains_cell(self, cell: tuple[int, int]) -> bool:
        """Return whether a grid cell lies inside the arena."""

        column, row = cell
        return 0 <= column < self.columns and 0 <= row < self.rows

    def grid_to_metric(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Convert a grid cell to its metric centre coordinate."""

        if not self.contains_cell(cell):
            raise ValueError(f"Grid cell {cell} is outside the arena.")
        column, row = cell
        return (
            self.origin_x + (column + 0.5) * self.cell_width,
            self.origin_y + (row + 0.5) * self.cell_height,
        )

    def metric_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert an in-arena metric coordinate to its containing cell."""

        relative_x = x - self.origin_x
        relative_y = y - self.origin_y
        if not (0.0 <= relative_x < self.width and 0.0 <= relative_y < self.height):
            raise ValueError(f"Metric position ({x}, {y}) is outside the arena.")
        return floor(relative_x / self.cell_width), floor(relative_y / self.cell_height)

    def target_cell(
        self,
        current: tuple[int, int],
        action: NavigationAction,
    ) -> tuple[int, int]:
        """Return the neighboring target cell, rejecting boundary crossings."""

        if not self.contains_cell(current):
            raise ValueError(f"Current grid cell {current} is outside the arena.")
        delta_x, delta_y = action.cell_delta
        target = current[0] + delta_x, current[1] + delta_y
        if not self.contains_cell(target):
            raise ValueError(f"Action would move outside the arena to cell {target}.")
        return target
