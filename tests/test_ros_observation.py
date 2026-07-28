from types import SimpleNamespace

import pytest

from active_inference_navigation.adapters.ros_observation import RosObservationSource
from active_inference_navigation.interfaces import (
    ObservationUnavailableError,
    StaleObservationError,
)


def odometry(x, y):
    position = SimpleNamespace(x=x, y=y)
    return SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(position=position)))


def rssi(value):
    return SimpleNamespace(data=value)


def test_ros_observation_combines_odometry_and_median_rssi():
    source = RosObservationSource(rssi_median_window=3, clock=lambda: 10.0)
    source.odometry_callback(odometry(1.25, 2.5), received_at=9.5)
    source.rssi_callback(rssi(-90.0), received_at=9.6)
    source.rssi_callback(rssi(-20.0), received_at=9.7)
    source.rssi_callback(rssi(-60.0), received_at=9.8)

    observation = source.read_observation()

    assert (observation.x, observation.y) == pytest.approx((1.25, 2.5))
    assert observation.rssi == pytest.approx(-60.0)


def test_ros_observation_transforms_raw_odometry_to_arena_coordinates():
    source = RosObservationSource(
        clock=lambda: 10.0,
        position_transform=lambda x, y: (x + 0.175, y + 0.175),
    )
    source.odometry_callback(odometry(0.0, 0.0), received_at=10.0)
    source.rssi_callback(rssi(-63.0), received_at=10.0)

    observation = source.read_observation()

    assert (observation.x, observation.y) == pytest.approx((0.175, 0.175))


def test_rssi_window_keeps_only_configured_number_of_samples():
    source = RosObservationSource(rssi_median_window=3, clock=lambda: 5.0)
    source.odometry_callback(odometry(0.0, 0.0), received_at=5.0)
    for value in (-100.0, -90.0, -80.0, -70.0):
        source.rssi_callback(rssi(value), received_at=5.0)

    assert source.read_observation().rssi == pytest.approx(-80.0)


def test_ros_observation_rejects_stale_rssi():
    source = RosObservationSource(rssi_timeout=0.5, clock=lambda: 10.0)
    source.odometry_callback(odometry(1.0, 2.0), received_at=10.0)
    source.rssi_callback(rssi(-60.0), received_at=9.0)

    with pytest.raises(StaleObservationError, match="RSSI"):
        source.read_observation()


def test_ros_observation_rejects_stale_odometry():
    source = RosObservationSource(odom_timeout=0.5, clock=lambda: 10.0)
    source.odometry_callback(odometry(1.0, 2.0), received_at=9.0)
    source.rssi_callback(rssi(-60.0), received_at=10.0)

    with pytest.raises(StaleObservationError, match="odometry"):
        source.read_observation()


def test_ros_observation_requires_both_sensor_streams():
    source = RosObservationSource(clock=lambda: 10.0)
    source.odometry_callback(odometry(1.0, 2.0), received_at=10.0)

    with pytest.raises(ObservationUnavailableError, match="RSSI"):
        source.read_observation()
