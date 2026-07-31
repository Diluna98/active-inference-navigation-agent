import numpy as np
import pytest

from active_inference_navigation import NavigationAgentConfig, build_navigation_agent
from active_inference_navigation.likelihoods import (
    BearingCalibratedDbmLikelihood,
    CalibratedDbmLikelihood,
)


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


def test_bearing_likelihood_changes_prediction_around_transmitter():
    likelihood = BearingCalibratedDbmLikelihood(
        (20, 20, 100),
        minimum_rssi=-95.0,
        maximum_rssi=-25.0,
        reference_rssi=-63.109,
        path_loss_exponent=3.104,
        minimum_distance=0.35,
        bearing_cosine_coefficient=4.761,
        bearing_sine_coefficient=-9.065,
    )

    east = likelihood.expected_rssi_at(1.0, 0.0, 0.0, 0.0)
    north = likelihood.expected_rssi_at(0.0, 1.0, 0.0, 0.0)

    assert east == pytest.approx(-58.348)
    assert north == pytest.approx(-72.174)


def test_agent_builds_with_bearing_calibrated_provider():
    agent = build_navigation_agent(
        NavigationAgentConfig(
            model_size=20,
            workspace_size=7.0,
            workspace_height=7.0,
            likelihood_provider="bearing_calibrated_dbm",
        )
    )

    assert len(agent.policies) == 5


def test_agent_rejects_unknown_likelihood_provider():
    with pytest.raises(ValueError, match="Unknown likelihood"):
        build_navigation_agent(NavigationAgentConfig(likelihood_provider="missing"))
