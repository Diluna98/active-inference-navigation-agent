import numpy as np

from active_inference_navigation.agent import CardinalNavigationAgent
from active_inference_navigation.constraints import GridBoundaryConstraint
from active_inference_navigation.geometry import GridGeometry
from active_inference_navigation.models import Observation


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


def test_agent_selects_next_best_policy_when_boundary_masks_preferred_action():
    agent = CardinalNavigationAgent(FakePolicyAgent())
    constraint = GridBoundaryConstraint(GridGeometry())
    allowed = constraint.allowed_actions(Observation(0.175, 0.175, -70.0))

    selected = agent.select_action(allowed)

    assert tuple(selected[:2]) == (2, 0)
