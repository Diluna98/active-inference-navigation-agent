import pytest

from active_inference_navigation.geometry import GridGeometry
from active_inference_navigation.models import NavigationAction


def test_real_arena_cell_dimensions():
    geometry = GridGeometry()

    assert geometry.cell_width == pytest.approx(0.35)
    assert geometry.cell_height == pytest.approx(0.35)


def test_grid_to_metric_uses_cell_centres():
    geometry = GridGeometry()

    assert geometry.grid_to_metric((0, 0)) == pytest.approx((0.175, 0.175))
    assert geometry.grid_to_metric((19, 19)) == pytest.approx((6.825, 6.825))


def test_metric_to_grid_uses_configurable_origin():
    geometry = GridGeometry(columns=2, rows=2, width=1.0, height=1.0, origin_x=-0.5)

    assert geometry.metric_to_grid(-0.49, 0.01) == (0, 0)
    assert geometry.metric_to_grid(0.49, 0.99) == (1, 1)


def test_target_cell_rejects_boundary_crossing():
    geometry = GridGeometry()

    with pytest.raises(ValueError, match="outside"):
        geometry.target_cell((0, 0), NavigationAction.from_sequence((1, 0)))
