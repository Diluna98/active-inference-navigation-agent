from pathlib import Path

import pytest

from active_inference_navigation.config import (
    NavigationConfig,
    load_default_navigation_config,
    load_navigation_config,
)


def test_default_configuration_matches_real_experiment():
    config = NavigationConfig()

    assert config.grid.geometry().cell_width == pytest.approx(0.35)
    assert config.grid.geometry().cell_height == pytest.approx(0.35)
    assert config.grid.origin_x == pytest.approx(0.0)
    assert config.grid.origin_y == pytest.approx(0.0)
    assert config.frame.arena_x_from_odom == "-y"
    assert config.frame.arena_y_from_odom == "+x"
    assert config.frame.odom_zero_arena_x == pytest.approx(0.175)
    assert config.frame.odom_zero_arena_y == pytest.approx(0.175)
    assert config.active_inference.goal_resolution == 10
    assert config.active_inference.temporal_horizon == 1
    assert config.active_inference.policy_samples == 200
    assert config.experiment.start_column == 0
    assert config.experiment.start_row == 0
    assert config.termination.provider == "persistent_rssi"
    assert config.termination.rssi_threshold == pytest.approx(-62.0)
    assert config.termination.consecutive_observations == 3
    assert config.topics.odom == "/tb4_08/odom"
    assert config.topics.rssi == "/tb4_08/rssi"
    assert config.topics.cmd_vel == "/tb4_08/cmd_vel"
    assert config.motion.final_heading == "positive_x"
    assert config.motion.settling_time == pytest.approx(2.5)
    assert config.likelihood_provider == "calibrated_dbm"
    assert config.rssi_likelihood.reference_rssi == pytest.approx(-63.02)


def test_repository_yaml_loads():
    config = load_navigation_config(Path("config/navigation.yaml"))

    assert config.grid.columns == 20
    assert config.active_inference.goal_resolution == 10
    assert config.sensors.rssi_median_window == 5
    assert config.likelihood_provider == "calibrated_dbm"


def test_packaged_default_matches_repository_yaml():
    assert load_default_navigation_config() == load_navigation_config(
        Path("config/navigation.yaml")
    )


def test_yaml_can_replace_sensor_and_environment_settings(tmp_path):
    path = tmp_path / "navigation.yaml"
    path.write_text(
        """
grid:
  columns: 10
  rows: 5
  width: 4.0
  height: 2.0
topics:
  odom: /robot/pose
  rssi: /radio/rssi
  cmd_vel: /robot/velocity
sensors:
  rssi_median_window: 9
""",
        encoding="utf-8",
    )

    config = load_navigation_config(path)

    assert config.grid.geometry().cell_width == pytest.approx(0.4)
    assert config.grid.geometry().cell_height == pytest.approx(0.4)
    assert config.topics.rssi == "/radio/rssi"
    assert config.sensors.rssi_median_window == 9


def test_configuration_rejects_invalid_timeout(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("sensors:\n  rssi_timeout: 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="timeouts"):
        load_navigation_config(path)
