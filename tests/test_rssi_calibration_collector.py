from math import cos, pi, sin
from types import SimpleNamespace

import pytest

from active_inference_navigation.frame import ArenaFrameTransform
from active_inference_navigation.geometry import GridGeometry
from active_inference_navigation.ros_rssi_calibration import (
    RssiCalibrationCollector,
    parse_movement_command,
    parse_target_cell,
    plan_calibration_movement,
    plan_grid_path,
)


def _odometry(
    *,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    linear_x: float = 0.0,
    linear_y: float = 0.0,
    angular_z: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y),
                orientation=SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=sin(yaw / 2.0),
                    w=cos(yaw / 2.0),
                ),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=linear_x, y=linear_y),
                angular=SimpleNamespace(z=angular_z),
            )
        ),
    )


def _rssi(value: float) -> SimpleNamespace:
    return SimpleNamespace(data=value)


def _rotated_frame() -> ArenaFrameTransform:
    return ArenaFrameTransform(
        arena_x_from_odom="-y",
        arena_y_from_odom="+x",
        odom_zero_arena_x=0.175,
        odom_zero_arena_y=0.175,
    )


def test_collector_transforms_pose_heading_and_distance() -> None:
    rows: list[dict[str, float | int]] = []
    collector = RssiCalibrationCollector(
        source_x=2.175,
        source_y=2.175,
        frame_transform=_rotated_frame(),
        row_sink=rows.append,
        settling_time=2.0,
        odom_timeout=1.0,
        required_heading="positive_y",
        wall_clock=lambda: 1234.5,
    )

    collector.odometry_callback(_odometry(x=1.0, y=-2.0), received_at=10.0)
    collector.odometry_callback(_odometry(x=1.0, y=-2.0), received_at=12.0)

    assert collector.rssi_callback(_rssi(-70.0), received_at=12.0)
    assert collector.sample_count == 1
    assert rows[0]["timestamp_unix_s"] == 1234.5
    assert rows[0]["arena_x_m"] == pytest.approx(2.175)
    assert rows[0]["arena_y_m"] == pytest.approx(1.175)
    assert rows[0]["arena_yaw_rad"] == pytest.approx(pi / 2.0)
    assert rows[0]["horizontal_distance_m"] == pytest.approx(1.0)
    assert rows[0]["distance_bin_m"] == pytest.approx(1.0)


def test_collector_rejects_motion_unsettled_stale_and_wrong_heading() -> None:
    rows: list[dict[str, float | int]] = []
    collector = RssiCalibrationCollector(
        source_x=1.0,
        source_y=1.0,
        frame_transform=_rotated_frame(),
        row_sink=rows.append,
        settling_time=1.0,
        odom_timeout=1.0,
        required_heading="positive_y",
    )

    collector.odometry_callback(_odometry(linear_x=0.1), received_at=0.0)
    assert not collector.rssi_callback(_rssi(-75.0), received_at=0.1)

    collector.odometry_callback(_odometry(), received_at=1.0)
    assert not collector.rssi_callback(_rssi(-75.0), received_at=1.5)

    collector.odometry_callback(_odometry(yaw=-pi / 2.0), received_at=3.0)
    assert not collector.rssi_callback(_rssi(-75.0), received_at=3.0)

    collector.odometry_callback(_odometry(), received_at=3.1)
    assert not collector.rssi_callback(_rssi(-75.0), received_at=4.2)
    assert rows == []


def test_collector_can_accept_any_heading() -> None:
    rows: list[dict[str, float | int]] = []
    collector = RssiCalibrationCollector(
        source_x=0.0,
        source_y=0.0,
        frame_transform=_rotated_frame(),
        row_sink=rows.append,
        settling_time=0.0,
        required_heading=None,
    )

    collector.odometry_callback(_odometry(yaw=pi), received_at=5.0)

    assert collector.rssi_callback(_rssi(-60.0), received_at=5.0)
    assert len(rows) == 1


def test_collector_stops_exactly_at_requested_batch_size() -> None:
    rows: list[dict[str, float | int]] = []
    collector = RssiCalibrationCollector(
        source_x=0.0,
        source_y=0.0,
        frame_transform=_rotated_frame(),
        row_sink=rows.append,
        settling_time=0.0,
    )
    collector.odometry_callback(_odometry(yaw=-pi / 2.0), received_at=1.0)
    collector.begin_batch(2)

    assert collector.rssi_callback(_rssi(-70.0), received_at=1.0)
    assert collector.rssi_callback(_rssi(-71.0), received_at=1.0)
    assert not collector.rssi_callback(_rssi(-72.0), received_at=1.0)
    assert collector.batch_complete
    assert collector.batch_count == 2
    assert len(rows) == 2


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("up", (0, 2)),
        ("down", (0, 1)),
        ("left", (1, 0)),
        ("right", (2, 0)),
    ],
)
def test_parse_movement_command_uses_arena_directions(command, expected) -> None:
    assert tuple(parse_movement_command(command).as_array()) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("(8,12)", (8, 12)),
        (" ( 8, 12 ) ", (8, 12)),
        ("8,12", (8, 12)),
        ("up", None),
    ],
)
def test_parse_target_cell(command, expected) -> None:
    assert parse_target_cell(command) == expected


def test_plan_calibration_movement_builds_path_to_absolute_cell() -> None:
    actions = plan_calibration_movement(
        "(3,2)",
        current_cell=(1, 1),
        geometry=GridGeometry(columns=5, rows=5, width=5.0, height=5.0),
    )

    assert [tuple(action.as_array()) for action in actions] == [
        (2, 0),
        (2, 0),
        (0, 2),
    ]


def test_grid_path_avoids_blocked_source_cell() -> None:
    geometry = GridGeometry()
    blocked_source = (8, 12)
    actions = plan_grid_path(
        current_cell=(7, 12),
        target_cell=(9, 12),
        geometry=geometry,
        blocked_cells=frozenset({blocked_source}),
    )
    visited = []
    cell = (7, 12)
    for action in actions:
        delta_x, delta_y = action.cell_delta
        cell = cell[0] + delta_x, cell[1] + delta_y
        visited.append(cell)

    assert cell == (9, 12)
    assert blocked_source not in visited
    assert len(actions) == 4


def test_grid_path_rejects_blocked_target_cell() -> None:
    with pytest.raises(ValueError, match="blocked"):
        plan_grid_path(
            current_cell=(7, 12),
            target_cell=(8, 12),
            geometry=GridGeometry(),
            blocked_cells=frozenset({(8, 12)}),
        )


def test_cardinal_movement_rejects_blocked_source_cell() -> None:
    with pytest.raises(ValueError, match="blocked"):
        plan_calibration_movement(
            "right",
            current_cell=(7, 12),
            geometry=GridGeometry(),
            blocked_cells=frozenset({(8, 12)}),
        )


def test_plan_calibration_movement_accepts_same_cell() -> None:
    geometry = GridGeometry(columns=5, rows=5, width=5.0, height=5.0)

    assert plan_calibration_movement("2,3", (2, 3), geometry) == ()


@pytest.mark.parametrize("command", ["(20,0)", "left"])
def test_plan_calibration_movement_rejects_boundary_crossing(command) -> None:
    geometry = GridGeometry()

    with pytest.raises(ValueError, match="outside"):
        plan_calibration_movement(command, (0, 0), geometry)
