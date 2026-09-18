import numpy as np
import pytest
from PyAIF import FilteredRecedingHorizonInference

from active_inference_navigation.mos import (
    TARGET_RESOLUTIONS,
    CollisionOutcome,
    FindOutcome,
    MOSAction,
    MOSAgentConfig,
    MOSEnvironment,
    MOSLayout,
    build_mos_agent,
    detection_distribution,
    sample_mos_instance,
    selected_mos_action,
    target_state,
)


def test_seeded_layout_and_target_are_reproducible():
    first = MOSLayout(map_seed=4)
    second = MOSLayout(map_seed=4)

    assert first.blocked == second.blocked
    assert first.sample_target(8) == second.sample_target(8)
    assert first.is_free(first.sample_target(8))


def test_seeded_robustness_instance_is_reproducible_and_valid():
    first = sample_mos_instance(17)
    second = sample_mos_instance(17)

    assert first == second
    assert first.layout.is_free(first.layout.start)
    assert first.layout.is_free(first.target)
    assert first.layout.start[0] < first.layout.size // 2
    assert first.target[0] >= first.layout.size // 2
    assert first.layout.sensor_range in (5.0, 6.0, 7.0)


def test_robustness_instances_vary_multiple_domain_properties():
    instances = [sample_mos_instance(seed) for seed in range(12)]

    assert len({instance.layout.map_seed for instance in instances}) > 1
    assert len({instance.layout.start for instance in instances}) > 1
    assert len({instance.target for instance in instances}) > 1
    assert len({instance.observation_seed for instance in instances}) > 1
    assert len({instance.layout.sensor_range for instance in instances}) > 1


def test_detection_improves_with_visible_range_and_occlusion():
    layout = MOSLayout(map_seed=0)
    near = detection_distribution((15, 10), (16, 10), layout)
    far = detection_distribution((1, 1), (19, 19), layout)

    assert near[1] > far[1]
    assert np.isclose(near.sum(), 1.0)
    assert np.isclose(far.sum(), 1.0)


def test_find_succeeds_only_when_target_is_visible():
    layout = MOSLayout(map_seed=0, start=(15, 10))
    visible = MOSEnvironment(layout=layout, target=(16, 10))
    hidden = MOSEnvironment(layout=layout, target=(19, 19))

    observation, success = visible.step(MOSAction.FIND, noise_quantile=0.5)
    hidden_observation, hidden_success = hidden.step(MOSAction.FIND, noise_quantile=0.5)

    assert success
    assert observation[3] == FindOutcome.FOUND
    assert not hidden_success
    assert hidden_observation[3] == FindOutcome.FALSE_FIND


def test_blocked_move_is_recorded_without_an_impossible_occupancy_observation():
    layout = MOSLayout(map_seed=0, start=(0, 0))
    environment = MOSEnvironment(layout=layout, target=(19, 19))

    observation, success = environment.step(MOSAction.LEFT, noise_quantile=0.5)

    assert not success
    assert environment.position == (0, 0)
    assert environment.last_collision
    assert observation[4] == CollisionOutcome.FREE


@pytest.mark.parametrize("resolution", TARGET_RESOLUTIONS)
@pytest.mark.parametrize("depth", (1, 2, 3))
def test_fixed_mos_agents_use_requested_resolution_and_depth(resolution, depth):
    agent = build_mos_agent(
        MOSAgentConfig(
            target_resolution=resolution,
            action_depth=depth,
            message_passing_iterations=2,
        )
    )

    assert isinstance(agent.inference, FilteredRecedingHorizonInference)
    assert agent.temporal_horizon == depth + 1
    assert len(agent.policies) == 5**depth
    assert tuple(agent.states_dim) == (20, 20, resolution**2, 2)
    assert agent.pA[2].shape == (2, 20, 20, resolution**2)
    assert agent.pA[3].shape == (3, 20, 20, resolution**2, 2)


def test_agent_completes_one_mos_cycle():
    layout = MOSLayout(map_seed=1)
    environment = MOSEnvironment(layout=layout, target_seed=2)
    agent = build_mos_agent(
        MOSAgentConfig(target_resolution=2, action_depth=1, message_passing_iterations=2),
        layout=layout,
    )
    agent.reset()
    observation = environment.reset(noise_quantile=0.5)
    agent.observe(observation, time_step=0)
    agent.infer_states()
    agent.infer_policies()

    assert selected_mos_action(agent) in MOSAction


def test_coarse_target_state_aliases_more_cells_than_fine_state():
    first = (10, 10)
    second = (11, 10)

    assert target_state(first, 2) == target_state(second, 2)
    assert target_state(first, 20) != target_state(second, 20)


def test_mos_config_rejects_nonpositive_computation_settings():
    with pytest.raises(ValueError, match="message_passing_iterations"):
        MOSAgentConfig(message_passing_iterations=0)
    with pytest.raises(ValueError, match="policy_workers"):
        MOSAgentConfig(policy_workers=0)
