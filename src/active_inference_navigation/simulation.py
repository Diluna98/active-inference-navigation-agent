"""Episode runner for the continuous-observation navigation agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapters.simulation import (
    SimulationActionExecutor,
    SimulationGoalTermination,
    SimulationObservationSource,
)
from .agent import NavigationAgentConfig, build_navigation_agent
from .environment import GridNavigationEnvironment
from .runtime import NavigationRuntime


@dataclass(frozen=True)
class NavigationEpisodeResult:
    """Trajectory data returned by a navigation episode."""

    distances: np.ndarray
    positions: np.ndarray
    actions: np.ndarray
    reached_goal: bool


def run_navigation_episode(
    *,
    config: NavigationAgentConfig | None = None,
    environment: GridNavigationEnvironment | None = None,
    planning_windows: int = 8,
) -> NavigationEpisodeResult:
    """Run a deterministic continuous-observation navigation episode."""

    if config is None:
        config = NavigationAgentConfig()
    if planning_windows < 1:
        raise ValueError("planning_windows must be positive.")
    if environment is None:
        environment = GridNavigationEnvironment(
            model_size=config.model_size,
            random_seed=config.random_seed,
        )

    observation_source = SimulationObservationSource(environment)
    observation_source.reset()
    runtime = NavigationRuntime(
        agent=build_navigation_agent(config),
        observation_source=observation_source,
        action_executor=SimulationActionExecutor(environment, observation_source),
        termination_condition=SimulationGoalTermination(
            goal=environment.goal,
            threshold=environment.goal_threshold,
        ),
        temporal_horizon=config.temporal_horizon,
    )
    runtime_result = runtime.run(planning_windows=planning_windows)
    positions = np.asarray(
        [(observation.x, observation.y) for observation in runtime_result.observations],
        dtype=float,
    )
    distances = np.linalg.norm(positions - np.asarray(environment.goal, dtype=float), axis=1)
    actions = np.asarray(
        [action.as_array() for action in runtime_result.actions],
        dtype=int,
    ).reshape(-1, 2)

    return NavigationEpisodeResult(
        distances=distances,
        positions=positions,
        actions=actions,
        reached_goal=runtime_result.terminated,
    )

