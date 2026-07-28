from active_inference_navigation.models import Observation
from active_inference_navigation.termination import PersistentRssiTermination


def observation(rssi):
    return Observation(0.175, 0.175, rssi)


def test_persistent_rssi_requires_consecutive_threshold_hits():
    termination = PersistentRssiTermination(-62.0, consecutive_observations=3)

    assert not termination.is_met(observation(-61.0))
    assert not termination.is_met(observation(-62.0))
    assert termination.is_met(observation(-60.0))


def test_persistent_rssi_resets_count_after_weak_observation():
    termination = PersistentRssiTermination(-62.0, consecutive_observations=2)

    assert not termination.is_met(observation(-61.0))
    assert not termination.is_met(observation(-70.0))
    assert not termination.is_met(observation(-61.0))
    assert termination.is_met(observation(-61.0))
