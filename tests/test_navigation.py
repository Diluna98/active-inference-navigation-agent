import numpy as np

from active_inference_navigation import (
    NavigationAgentConfig,
    run_navigation_episode,
)
from active_inference_navigation.animation import README_SCENARIOS, simulate_scenarios
from active_inference_navigation.likelihoods import RssiNavigationLikelihood


def test_continuous_shallow_navigation_approaches_one_cell_from_source():
    result = run_navigation_episode(
        config=NavigationAgentConfig(random_seed=7),
        planning_windows=20,
    )

    assert result.actions.shape == (20, 2)
    assert np.all(np.isfinite(result.distances))
    assert np.all(np.isfinite(result.positions))
    assert result.distances.min() <= 25.0
    assert result.distances[-1] < 0.15 * result.distances[0]


def test_continuous_deep_navigation_approaches_source_at_coarse_resolution():
    result = run_navigation_episode(
        config=NavigationAgentConfig(
            model_size=20,
            goal_resolution=2,
            temporal_horizon=3,
            message_passing_iterations=8,
            policy_samples=300,
            random_seed=7,
        ),
        planning_windows=8,
    )

    assert result.actions.shape == (16, 2)
    assert np.all(np.isfinite(result.distances))
    assert np.all(np.isfinite(result.positions))
    assert result.distances.min() < 0.5 * result.distances[0]
    assert result.distances[-1] <= result.distances.min() + 25.0


def test_readme_animation_scenarios_reach_their_sources():
    results = simulate_scenarios()

    assert len({scenario.start for scenario in README_SCENARIOS}) == len(README_SCENARIOS)
    assert len({scenario.source for scenario in README_SCENARIOS}) == len(README_SCENARIOS)
    assert all(result.reached_goal for result in results)
    assert all(result.distances[-1] <= 18.0 for result in results)


def test_fisher_information_proxy_matches_saved_reference():
    likelihood = RssiNavigationLikelihood(
        (20, 20, 4),
        normalized_signal_preference=True,
    )
    observation = np.array([487.5, 487.5, 1.1112189326234811])

    sensitivity = likelihood.compute_sensitivity(observation)

    assert np.isclose(
        sensitivity,
        0.053184174702422746,
        rtol=0.0,
        atol=2e-10,
    )
