"""Coordinate transforms between raw odometry and arena axes."""

from __future__ import annotations

from dataclasses import dataclass

_AXIS_VECTORS = {
    "+x": (1.0, 0.0),
    "-x": (-1.0, 0.0),
    "+y": (0.0, 1.0),
    "-y": (0.0, -1.0),
}


@dataclass(frozen=True)
class ArenaFrameTransform:
    """A signed-axis transform plus the arena position of odometry zero."""

    arena_x_from_odom: str = "+x"
    arena_y_from_odom: str = "+y"
    odom_zero_arena_x: float = 0.0
    odom_zero_arena_y: float = 0.0

    def __post_init__(self) -> None:
        try:
            x_axis = _AXIS_VECTORS[self.arena_x_from_odom]
            y_axis = _AXIS_VECTORS[self.arena_y_from_odom]
        except KeyError as error:
            raise ValueError("Frame axes must be one of +x, -x, +y, or -y.") from error
        dot_product = x_axis[0] * y_axis[0] + x_axis[1] * y_axis[1]
        if dot_product != 0.0:
            raise ValueError("Arena x and y must map to different odometry axes.")

    def vector_to_arena(self, odom_x: float, odom_y: float) -> tuple[float, float]:
        """Rotate/reflect an odometry vector into arena axes."""

        x_axis = _AXIS_VECTORS[self.arena_x_from_odom]
        y_axis = _AXIS_VECTORS[self.arena_y_from_odom]
        return (
            x_axis[0] * odom_x + x_axis[1] * odom_y,
            y_axis[0] * odom_x + y_axis[1] * odom_y,
        )

    def vector_to_odom(self, arena_x: float, arena_y: float) -> tuple[float, float]:
        """Apply the inverse signed-axis transform to an arena vector."""

        x_axis = _AXIS_VECTORS[self.arena_x_from_odom]
        y_axis = _AXIS_VECTORS[self.arena_y_from_odom]
        return (
            x_axis[0] * arena_x + y_axis[0] * arena_y,
            x_axis[1] * arena_x + y_axis[1] * arena_y,
        )

    def position_to_arena(self, odom_x: float, odom_y: float) -> tuple[float, float]:
        """Convert a raw odometry position to continuous arena coordinates."""

        arena_x, arena_y = self.vector_to_arena(odom_x, odom_y)
        return (
            arena_x + self.odom_zero_arena_x,
            arena_y + self.odom_zero_arena_y,
        )

    def position_to_odom(self, arena_x: float, arena_y: float) -> tuple[float, float]:
        """Convert an arena coordinate to the raw odometry frame."""

        return self.vector_to_odom(
            arena_x - self.odom_zero_arena_x,
            arena_y - self.odom_zero_arena_y,
        )
