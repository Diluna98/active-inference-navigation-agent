import numpy as np
import pytest

from active_inference_navigation.agent import CardinalNavigationAgent
from active_inference_navigation.config import NavigationConfig, TerminationConfig
from active_inference_navigation.constraints import GridBoundaryConstraint
from active_inference_navigation.geometry import GridGeometry
from active_inference_navigation.models import Observation
from active_inference_navigation.ros_runtime import build_action_constraint


def test_boundary_constraint_masks_outward_actions_at_lower_left():
    constraint = GridBoundaryConstraint(GridGeometry())

    allowed = {
        tuple(action.as_array())
        for action in constraint.allowed_actions(Observation(0.175, 0.175, -70.0))
    }

    assert allowed == {(0, 0), (2, 0), (0, 2)}


def test_boundary_constraint_allows_all_cardinal_actions_inside_grid():
    constraint = GridBoundaryConstraint(GridGeometry())

    allowed = constraint.allowed_actions(Observation(1.0, 1.0, -70.0))

    assert len(allowed) == 5


def test_boundary_constraint_rejects_action_into_blocked_source_cell():
    geometry = GridGeometry()
    constraint = GridBoundaryConstraint(
        geometry,
        blocked_cells=frozenset({(8, 12)}),
    )
    observation = Observation(*geometry.grid_to_metric((8, 11)), -70.0)

    allowed = {
        tuple(action.as_array())
        for action in constraint.allowed_actions(observation)
    }

    assert (0, 2) not in allowed
    assert (0, 0) in allowed


def test_source_footprint_goal_does_not_reject_action_toward_source():
    config = NavigationConfig(
        termination=TerminationConfig(
            provider="source_footprint",
            source_x=2.975,
            source_y=4.375,
            source_body_direction="positive_y",
        )
    )
    geometry = config.grid.geometry()
    constraint = build_action_constraint(config)
    observation = Observation(*geometry.grid_to_metric((8, 11)), -70.0)

    allowed = {
        tuple(action.as_array())
        for action in constraint.allowed_actions(observation)
    }

    assert (0, 2) in allowed


class FakePolicyAgent:
    deep_inference = False

    def __init__(self):
        self.policies = [
            np.asarray([[1, 0, 0]]),
            np.asarray([[2, 0, 0]]),
            np.asarray([[0, 0, 0]]),
        ]
        self.posterior_pi = np.asarray([0.9, 0.8, 0.1])

    def select_action(self):
        return self.policies[np.argmax(self.posterior_pi)][0]


class FixedActionAgent:
    def __init__(self, selected):
        self.selected = selected

    def select_action(self):
        return self.selected


def test_agent_selects_next_best_policy_when_boundary_masks_preferred_action():
    agent = CardinalNavigationAgent(FakePolicyAgent())
    constraint = GridBoundaryConstraint(GridGeometry())
    allowed = constraint.allowed_actions(Observation(0.175, 0.175, -70.0))

    selected = agent.select_action(allowed)

    assert tuple(selected[:2]) == (2, 0)


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        (np.asarray([[2], [0], [0]]), (2, 0)),
        (np.asarray([[0, 2, 0]]), (0, 2)),
    ],
)
def test_agent_normalizes_nested_numeric_action_shapes(selected, expected):
    action = CardinalNavigationAgent(FixedActionAgent(selected)).select_action()

    assert tuple(action) == expected
    assert action.dtype == np.dtype(int)
    assert action.shape == (2,)


def test_agent_normalizes_object_array_of_single_value_arrays():
    selected = np.empty(3, dtype=object)
    selected[:] = [np.asarray([1]), np.asarray([0]), np.asarray([0])]

    action = CardinalNavigationAgent(FixedActionAgent(selected)).select_action()

    assert tuple(action) == (1, 0)


def test_agent_rejects_action_factor_with_multiple_values():
    selected = np.empty(3, dtype=object)
    selected[:] = [np.asarray([1, 2]), np.asarray([0]), np.asarray([0])]

    with pytest.raises(ValueError, match="exactly one value"):
        CardinalNavigationAgent(FixedActionAgent(selected)).select_action()
