import numpy as np
import pytest

from active_inference_navigation.adapters.simulation import (
    SimulationActionExecutor,
    SimulationObservationSource,
)
from active_inference_navigation.environment import GridNavigationEnvironment
from active_inference_navigation.models import NavigationAction


def test_simulation_adapters_preserve_environment_step_behavior():
    environment = GridNavigationEnvironment(
        model_size=20,
        workspace_size=7.5,
        start=(0.1875, 0.1875),
        goal=(7.3125, 7.3125),
    )
    source = SimulationObservationSource(environment)
    initial = source.reset()
    executor = SimulationActionExecutor(environment, source)

    executor.execute(NavigationAction.from_sequence((2, 0)))
    executor.wait_for_completion()
    moved = source.read_observation()

    assert (initial.x, initial.y) == pytest.approx((0.1875, 0.1875))
    assert (moved.x, moved.y) == pytest.approx((0.5625, 0.1875))
    assert np.array_equal(environment.position, (moved.x, moved.y))


def test_environment_rejects_diagonal_action():
    environment = GridNavigationEnvironment()

    with pytest.raises(ValueError, match="Diagonal"):
        environment.step((1, 2))
