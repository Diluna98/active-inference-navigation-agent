"""Construct a PyAIF continuous-observation navigation agent."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyAIF import (
    ActiveInfAgent,
    ContinuousLikelihood,
    DeepTemporalInference,
    GenerativeModel,
    ShallowInference,
    utils,
)

from .likelihoods import (
    BearingCalibratedDbmLikelihood,
    CalibratedDbmLikelihood,
    RssiNavigationLikelihood,
)
from .models import NavigationAction


def _object_array(*arrays) -> np.ndarray:
    result = np.empty(len(arrays), dtype=object)
    for index, array in enumerate(arrays):
        result[index] = np.asarray(array, dtype=float)
    return result


def _normalize_selected_action(selected: Any) -> np.ndarray:
    """Convert PyAIF's scalar-per-factor action formats to a flat integer array."""

    raw_components = np.asarray(selected, dtype=object).reshape(-1)
    components: list[int] = []
    for raw_component in raw_components:
        component = np.asarray(raw_component)
        if component.size != 1:
            raise ValueError("Each selected action factor must contain exactly one value.")
        value = component.reshape(-1)[0]
        try:
            integer_value = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Selected action factors must be integer values.") from error
        if integer_value != value:
            raise ValueError("Selected action factors must be integer values.")
        components.append(integer_value)
    return np.asarray(components, dtype=int)


def _transition_model(states_dim: tuple[int, int, int]) -> np.ndarray:
    transitions = []
    for factor, state_count in enumerate(states_dim):
        action_count = 3 if factor < 2 else 1
        transition = np.zeros((state_count, state_count, action_count))
        transition[:, :, 0] = np.eye(state_count)
        if factor < 2:
            for state in range(state_count):
                transition[max(0, state - 1), state, 1] = 1.0
                transition[min(state_count - 1, state + 1), state, 2] = 1.0
        transitions.append(transition)
    return _object_array(*transitions)


def _cardinal_policies(
    states_dim: tuple[int, int, int],
    controls_dim: tuple[int, int, int],
    horizon: int,
) -> list[np.ndarray]:
    policy_length = max(1, horizon - 1)
    policies = utils.construct_policies(
        states_dim,
        controls_dim,
        policy_length,
        [0, 1],
    )
    return [
        policy
        for policy in policies
        if not np.any((policy[:, 0] != 0) & (policy[:, 1] != 0))
        and np.all(policy[:, 2:] == 0)
    ]


@dataclass(frozen=True)
class NavigationAgentConfig:
    """Configuration for the continuous-observation navigation agent."""

    model_size: int = 20
    model_rows: int | None = None
    workspace_size: float = 500.0
    workspace_height: float | None = None
    goal_resolution: int = 10
    temporal_horizon: int = 1
    message_passing_iterations: int = 5
    policy_samples: int = 200
    exact_state_limit: int = 100
    random_seed: int = 0
    policy_workers: int = 1
    normalized_signal_preference: bool = False
    likelihood_provider: str = "rssi_navigation"
    reference_rssi: float = -63.109
    path_loss_exponent: float = 3.104
    signal_sigma: float = 7.0
    minimum_calibrated_distance: float = 0.35
    minimum_rssi: float = -95.0
    maximum_rssi: float = -25.0
    bearing_cosine_coefficient: float = 4.761
    bearing_sine_coefficient: float = -9.065


class CardinalNavigationAgent:
    """Expose and validate cardinal actions selected from joint policies."""

    def __init__(self, agent: ActiveInfAgent) -> None:
        self._agent = agent

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    def reset(self) -> None:
        """Reset inference state and cardinal-action arbitration."""

        self._agent.reset()

    def select_action(
        self,
        allowed_actions: Collection[NavigationAction] | None = None,
    ) -> np.ndarray | None:
        """Select the best policy whose next action satisfies constraints."""

        posterior = None
        if allowed_actions is not None:
            allowed = {tuple(action.as_array()) for action in allowed_actions}
            if not allowed:
                raise ValueError("At least one action must be allowed.")
            policy_time = 0
            if getattr(self._agent, "deep_inference", False):
                policy_time = (
                    int(getattr(self._agent, "_current_time", 0))
                    % int(self._agent.temporal_horizon)
                )
            if policy_time < self._agent.policies[0].shape[0]:
                posterior = np.asarray(self._agent.posterior_pi).copy()
                for index, policy in enumerate(self._agent.policies):
                    if tuple(np.asarray(policy[policy_time, :2], dtype=int)) not in allowed:
                        self._agent.posterior_pi[index] = -1.0
        try:
            selected = self._agent.select_action()
        finally:
            if posterior is not None:
                self._agent.posterior_pi[...] = posterior
        if selected is None:
            return None
        action = _normalize_selected_action(selected)
        navigation_action = NavigationAction.from_sequence(action)
        if allowed_actions is not None and navigation_action not in allowed_actions:
            raise RuntimeError("The inference agent selected a constrained action.")
        return navigation_action.as_array()

    @property
    def navigation_actions(self) -> tuple[NavigationAction, ...]:
        """Return the complete public action space."""

        return tuple(
            NavigationAction.from_sequence(values)
            for values in ((0, 0), (1, 0), (2, 0), (0, 1), (0, 2))
        )


def build_navigation_agent(
    config: NavigationAgentConfig | None = None,
) -> CardinalNavigationAgent:
    """Build the navigation agent using PyAIF's component API."""

    if config is None:
        config = NavigationAgentConfig()
    states_dim = (
        config.model_size,
        config.model_size if config.model_rows is None else config.model_rows,
        config.goal_resolution**2,
    )
    controls_dim = (3, 3, 1)
    horizon = config.temporal_horizon
    policies = _cardinal_policies(states_dim, controls_dim, horizon)
    model = GenerativeModel(
        B=_transition_model(states_dim),
        D=_object_array(*(np.ones(state_count) for state_count in states_dim)),
        controls_dim=controls_dim,
        controllable_factors=[0, 1],
        policies=policies,
    )

    if config.likelihood_provider == "rssi_navigation":
        domain_likelihood = RssiNavigationLikelihood(
            states_dim,
            workspace_size=config.workspace_size,
            workspace_height=config.workspace_height,
            normalized_signal_preference=config.normalized_signal_preference,
        )
    elif config.likelihood_provider in {"calibrated_dbm", "bearing_calibrated_dbm"}:
        likelihood_type = (
            BearingCalibratedDbmLikelihood
            if config.likelihood_provider == "bearing_calibrated_dbm"
            else CalibratedDbmLikelihood
        )
        likelihood_kwargs = {}
        if likelihood_type is BearingCalibratedDbmLikelihood:
            likelihood_kwargs = {
                "bearing_cosine_coefficient": config.bearing_cosine_coefficient,
                "bearing_sine_coefficient": config.bearing_sine_coefficient,
            }
        domain_likelihood = likelihood_type(
            states_dim,
            workspace_size=config.workspace_size,
            workspace_height=config.workspace_height,
            reference_rssi=config.reference_rssi,
            path_loss_exponent=config.path_loss_exponent,
            signal_sigma=config.signal_sigma,
            minimum_distance=config.minimum_calibrated_distance,
            minimum_rssi=config.minimum_rssi,
            maximum_rssi=config.maximum_rssi,
            normalized_signal_preference=config.normalized_signal_preference,
            **likelihood_kwargs,
        )
    else:
        raise ValueError(f"Unknown likelihood provider: {config.likelihood_provider}")
    likelihood = ContinuousLikelihood.from_model(
        domain_likelihood,
        modality_dependencies=[[0], [1], [0, 1, 2]],
        grid_size=domain_likelihood.grid_size,
        policy_samples=config.policy_samples,
        exact_state_limit=config.exact_state_limit,
        random_seed=config.random_seed,
    )
    if horizon > 1:
        inference = DeepTemporalInference(
            horizon=horizon,
            message_passing_iterations=config.message_passing_iterations,
            policy_workers=config.policy_workers,
        )
    else:
        inference = ShallowInference(
            message_passing_iterations=config.message_passing_iterations,
            policy_workers=config.policy_workers,
        )

    return CardinalNavigationAgent(
        ActiveInfAgent(
            model=model,
            likelihood=likelihood,
            inference=inference,
            action_selection="deterministic",
        )
    )
