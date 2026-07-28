import numpy as np
import pytest

from active_inference_navigation.models import NavigationAction, Observation


@pytest.mark.parametrize(
    ("values", "delta"),
    [
        ((0, 0), (0, 0)),
        ((1, 0), (-1, 0)),
        ((2, 0), (1, 0)),
        ((0, 1), (0, -1)),
        ((0, 2), (0, 1)),
    ],
)
def test_navigation_action_accepts_cardinal_actions(values, delta):
    action = NavigationAction.from_sequence(values)

    assert action.cell_delta == delta
    assert np.array_equal(action.as_array(), values)


@pytest.mark.parametrize("values", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_navigation_action_rejects_diagonal_actions(values):
    with pytest.raises(ValueError, match="Diagonal"):
        NavigationAction.from_sequence(values)


def test_navigation_action_rejects_unknown_component():
    with pytest.raises(ValueError, match="0, 1, or 2"):
        NavigationAction.from_sequence((3, 0))


def test_observation_converts_to_agent_array():
    observation = Observation(1.0, 2.0, -63.5, timestamp=4.0)

    assert np.array_equal(observation.as_array(), (1.0, 2.0, -63.5))


def test_observation_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        Observation(1.0, float("nan"), -60.0)
