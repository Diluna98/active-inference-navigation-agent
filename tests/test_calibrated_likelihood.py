import numpy as np
import pytest

from active_inference_navigation import NavigationAgentConfig, build_navigation_agent
from active_inference_navigation.likelihoods import CalibratedDbmLikelihood


def test_calibrated_likelihood_matches_fitted_reference_curve():
    likelihood = CalibratedDbmLikelihood((20, 20, 100))

    assert likelihood.expected_rssi(1.0) == pytest.approx(-63.02)
    assert likelihood.expected_rssi(8.6) == pytest.approx(-78.29, abs=0.02)


def test_calibrated_likelihood_clamps_uncalibrated_near_distance():
    likelihood = CalibratedDbmLikelihood((20, 20, 100))

    assert likelihood.expected_rssi(0.1) == pytest.approx(
        likelihood.expected_rssi(1.0)
    )


def test_calibrated_likelihood_uses_negative_dbm_observation_grid():
    likelihood = CalibratedDbmLikelihood((20, 20, 100))

    signal_grid = likelihood.get_o_grid(2)

    assert signal_grid[0] == pytest.approx(-95.0)
    assert signal_grid[-1] == pytest.approx(-55.0)
    assert np.all(np.isfinite(likelihood.likelihoods(-72.0, 2)))


def test_agent_builds_with_calibrated_provider():
    agent = build_navigation_agent(
        NavigationAgentConfig(
            model_size=20,
            workspace_size=7.0,
            workspace_height=7.0,
            likelihood_provider="calibrated_dbm",
        )
    )

    assert len(agent.policies) == 5


def test_agent_rejects_unknown_likelihood_provider():
    with pytest.raises(ValueError, match="Unknown likelihood"):
        build_navigation_agent(NavigationAgentConfig(likelihood_provider="missing"))
