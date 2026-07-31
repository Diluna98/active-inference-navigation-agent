"""Domain likelihoods for continuous RSSI navigation observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RssiNavigationLikelihood:
    """Continuous position and RSSI likelihood over discrete spatial states.

    Hidden-state factors are current x cell, current y cell, and a flattened
    square grid of possible transmitter/source cells.
    """

    states_dim: Sequence[int]
    workspace_size: float = 500.0
    workspace_height: float | None = None
    minimum_rssi: float = 0.0
    maximum_rssi: float = 30.0
    signal_decay: float = 0.01
    position_sigma: float = 1.0
    signal_sigma: float = 2.0
    preference_midpoint: float = 10.0
    preference_scale: float = 4.0
    grid_size: int = 100
    normalized_signal_preference: bool = False
    master_source_resolution: int = 20
    log_preferences: dict = field(init=False)

    def __post_init__(self) -> None:
        self.states_dim = tuple(int(size) for size in self.states_dim)
        if len(self.states_dim) != 3:
            raise ValueError("states_dim must contain x, y, and flattened source-grid factors.")
        goal_resolution = int(np.sqrt(self.states_dim[2]))
        if goal_resolution**2 != self.states_dim[2]:
            raise ValueError("The source-state dimension must be a perfect square.")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least two.")
        if min(self.position_sigma, self.signal_sigma) <= 0:
            raise ValueError("Likelihood standard deviations must be positive.")

        self.goal_resolution = goal_resolution
        if self.workspace_height is None:
            self.workspace_height = self.workspace_size
        self.x_centers = self._cell_centers(self.states_dim[0], self.workspace_size)
        self.y_centers = self._cell_centers(self.states_dim[1], self.workspace_height)
        goal_x = self._cell_centers(goal_resolution, self.workspace_size)
        goal_y = self._cell_centers(goal_resolution, self.workspace_height)

        current_x, current_y, transmitter_x, transmitter_y = np.meshgrid(
            self.x_centers,
            self.y_centers,
            goal_x,
            goal_y,
            indexing="ij",
        )
        signal_mean = self._signal_from_positions(
            current_x,
            current_y,
            transmitter_x,
            transmitter_y,
        )
        self.signal_mean = np.transpose(signal_mean, (0, 1, 3, 2)).reshape(self.states_dim)
        self._build_master_sensitivity()
        self.log_preferences = self._build_log_preferences()

    @staticmethod
    def _cell_centers(resolution: int, extent: float) -> np.ndarray:
        cell_size = extent / resolution
        return (np.arange(resolution, dtype=float) + 0.5) * cell_size

    def _build_log_preferences(self) -> dict:
        position_grid = self.get_o_grid(0)
        joint_position = np.full(
            (len(position_grid), len(position_grid)),
            1.0 / len(position_grid) ** 2,
        )

        signal_grid = self.get_o_grid(2)
        utility = 1.0 / (
            1.0 + np.exp(-(signal_grid - self.preference_midpoint) / self.preference_scale)
        )
        if self.normalized_signal_preference:
            signal_probability = utility + 0.1
        else:
            signal_probability = np.exp(utility - utility.max())
        signal_probability /= signal_probability.sum()
        return {
            (0, 1): np.log(joint_position),
            2: np.log(signal_probability),
        }

    def _build_master_sensitivity(self) -> None:
        goal_x = self._cell_centers(self.master_source_resolution, self.workspace_size)
        goal_y = self._cell_centers(self.master_source_resolution, self.workspace_height)
        current_x, current_y, transmitter_x, transmitter_y = np.meshgrid(
            self.x_centers,
            self.y_centers,
            goal_x,
            goal_y,
            indexing="ij",
        )
        signal_mean = self._signal_from_positions(
            current_x,
            current_y,
            transmitter_x,
            transmitter_y,
        )
        self.master_signal_mean = np.transpose(
            signal_mean,
            (0, 1, 3, 2),
        ).reshape(
            self.states_dim[0],
            self.states_dim[1],
            self.master_source_resolution**2,
        )
        gradient_x = np.gradient(self.master_signal_mean, axis=0)
        gradient_y = np.gradient(self.master_signal_mean, axis=1)
        self.fisher_map_signal = (
            gradient_x**2 + gradient_y**2
        ) / self.signal_sigma**2

    def compute_sensitivity(self, observation) -> float:
        """Return the paper's master-grid RSSI Fisher-information proxy."""

        signal = float(np.asarray(observation, dtype=float)[2])
        standardized = (signal - self.master_signal_mean) / self.signal_sigma
        likelihood = (
            np.exp(-0.5 * standardized**2)
            / (self.signal_sigma * np.sqrt(2.0 * np.pi))
        )
        normalized = likelihood / (likelihood.sum() + 1e-8)
        return float(np.sum(normalized * self.fisher_map_signal))

    def get_o_grid(self, modality: int, N_grid: int | None = None) -> np.ndarray:
        size = self.grid_size if N_grid is None else int(N_grid)
        if modality in (0, 1):
            extent = self.workspace_size if modality == 0 else self.workspace_height
            return np.linspace(0.0, extent, size)
        if modality == 2:
            return np.linspace(self.minimum_rssi, self.maximum_rssi, size)
        raise ValueError(f"Unknown observation modality: {modality}")

    def likelihoods(self, observation: float, modality: int) -> np.ndarray:
        if modality == 0:
            mean = self.x_centers
            sigma = self.position_sigma
        elif modality == 1:
            mean = self.y_centers
            sigma = self.position_sigma
        elif modality == 2:
            mean = self.signal_mean
            sigma = self.signal_sigma
        else:
            raise ValueError(f"Unknown observation modality: {modality}")

        standardized = (float(observation) - mean) / sigma
        return np.exp(-0.5 * standardized**2) / (sigma * np.sqrt(2.0 * np.pi))

    def likelihoods_grid_vec(
        self,
        observation_grid: np.ndarray,
        modality: int,
        state_samples,
    ) -> np.ndarray:
        observation_grid = np.asarray(observation_grid, dtype=float)
        if modality == 0:
            mean = self.x_centers[np.asarray(state_samples, dtype=int)]
            sigma = self.position_sigma
        elif modality == 1:
            mean = self.y_centers[np.asarray(state_samples, dtype=int)]
            sigma = self.position_sigma
        elif modality == 2:
            x_state, y_state, goal_state = state_samples
            mean = self.signal_mean[
                np.asarray(x_state, dtype=int),
                np.asarray(y_state, dtype=int),
                np.asarray(goal_state, dtype=int),
            ]
            sigma = self.signal_sigma
        else:
            raise ValueError(f"Unknown observation modality: {modality}")

        standardized = (observation_grid[None, :] - mean[:, None]) / sigma
        return np.exp(-0.5 * standardized**2) / (sigma * np.sqrt(2.0 * np.pi))

    def _signal_from_distance(self, distance: np.ndarray) -> np.ndarray:
        """Return the simulated positive RSSI mean at each distance."""

        return self.maximum_rssi * np.exp(-self.signal_decay * distance)

    def _signal_from_positions(
        self,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        transmitter_x: np.ndarray,
        transmitter_y: np.ndarray,
    ) -> np.ndarray:
        """Return expected RSSI for receiver/transmitter arena positions."""

        distance = np.hypot(receiver_x - transmitter_x, receiver_y - transmitter_y)
        return self._signal_from_distance(distance)


@dataclass
class CalibratedDbmLikelihood(RssiNavigationLikelihood):
    """Log-distance RSSI likelihood fitted to real median-aggregated dBm data."""

    workspace_size: float = 7.0
    workspace_height: float | None = 7.0
    minimum_rssi: float = -95.0
    maximum_rssi: float = -55.0
    signal_sigma: float = 3.37
    preference_midpoint: float = -75.0
    preference_scale: float = 5.0
    reference_rssi: float = -63.02
    path_loss_exponent: float = 1.635
    minimum_distance: float = 1.0

    def __post_init__(self) -> None:
        if self.path_loss_exponent <= 0.0:
            raise ValueError("path_loss_exponent must be positive.")
        if self.minimum_distance <= 0.0:
            raise ValueError("minimum_distance must be positive.")
        if self.minimum_rssi >= self.maximum_rssi:
            raise ValueError("minimum_rssi must be lower than maximum_rssi.")
        if self.preference_scale <= 0.0:
            raise ValueError("preference_scale must be positive.")
        super().__post_init__()

    def _signal_from_distance(self, distance: np.ndarray) -> np.ndarray:
        """Return expected dBm using the fitted log-distance path-loss model."""

        return self.expected_rssi(distance)

    def expected_rssi(self, distance: float | np.ndarray) -> np.ndarray:
        """Return calibrated expected dBm for one or more metric distances."""

        calibrated_distance = np.maximum(np.asarray(distance, dtype=float), self.minimum_distance)
        signal = self.reference_rssi - 10.0 * self.path_loss_exponent * np.log10(
            calibrated_distance
        )
        return np.clip(signal, self.minimum_rssi, self.maximum_rssi)


@dataclass
class BearingCalibratedDbmLikelihood(CalibratedDbmLikelihood):
    """Directional dBm likelihood for a repeatable arena-fixed antenna heading.

    Bearing is measured in the arena frame from transmitter to receiver. The
    first-harmonic coefficients therefore model the combined directional
    response of a fixed transmitter and a robot restored to the same heading
    before each RSSI observation.
    """

    minimum_rssi: float = -95.0
    maximum_rssi: float = -25.0
    signal_sigma: float = 7.0
    reference_rssi: float = -63.109
    path_loss_exponent: float = 3.104
    minimum_distance: float = 0.35
    bearing_cosine_coefficient: float = 4.761
    bearing_sine_coefficient: float = -9.065

    def _signal_from_positions(
        self,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        transmitter_x: np.ndarray,
        transmitter_y: np.ndarray,
    ) -> np.ndarray:
        """Return log-distance RSSI adjusted for arena bearing."""

        return self.expected_rssi_at(
            receiver_x,
            receiver_y,
            transmitter_x,
            transmitter_y,
        )

    def expected_rssi_at(
        self,
        receiver_x: float | np.ndarray,
        receiver_y: float | np.ndarray,
        transmitter_x: float | np.ndarray,
        transmitter_y: float | np.ndarray,
    ) -> np.ndarray:
        """Return expected dBm at arena positions with a fixed robot heading."""

        receiver_x = np.asarray(receiver_x, dtype=float)
        receiver_y = np.asarray(receiver_y, dtype=float)
        transmitter_x = np.asarray(transmitter_x, dtype=float)
        transmitter_y = np.asarray(transmitter_y, dtype=float)
        delta_x = receiver_x - transmitter_x
        delta_y = receiver_y - transmitter_y
        distance = np.hypot(delta_x, delta_y)
        bearing = np.arctan2(delta_y, delta_x)
        signal = (
            self.reference_rssi
            - 10.0
            * self.path_loss_exponent
            * np.log10(np.maximum(distance, self.minimum_distance))
            + self.bearing_cosine_coefficient * np.cos(bearing)
            + self.bearing_sine_coefficient * np.sin(bearing)
        )
        return np.clip(signal, self.minimum_rssi, self.maximum_rssi)
