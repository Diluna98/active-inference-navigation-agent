"""Generic orchestration for Active Inference navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .interfaces import (
    ActionConstraint,
    ActionExecutor,
    ObservationSource,
    TerminationCondition,
)
from .models import NavigationAction, Observation


class ActiveInferenceController(Protocol):
    """Subset of the PyAIF agent API required by the runtime."""

    def reset(self) -> None: ...

    def observe(self, observation: Any, *, time_step: int) -> None: ...

    def infer_states(self) -> Any: ...

    def infer_policies(self) -> Any: ...

    def select_action(self) -> Any: ...


@dataclass(frozen=True)
class NavigationRuntimeResult:
    """Hardware-independent observations and actions recorded by a run."""

    observations: tuple[Observation, ...]
    actions: tuple[NavigationAction, ...]
    terminated: bool


@dataclass
class NavigationRuntime:
    """Connect an Active Inference controller to replaceable I/O adapters."""

    agent: ActiveInferenceController
    observation_source: ObservationSource
    action_executor: ActionExecutor
    termination_condition: TerminationCondition
    action_constraint: ActionConstraint | None = None
    temporal_horizon: int = 1

    def __post_init__(self) -> None:
        if self.temporal_horizon < 1:
            raise ValueError("temporal_horizon must be positive.")

    def run(self, *, planning_windows: int) -> NavigationRuntimeResult:
        """Run navigation for a bounded number of planning windows."""

        if planning_windows < 1:
            raise ValueError("planning_windows must be positive.")

        observation = self.observation_source.read_observation()
        observations = [observation]
        actions: list[NavigationAction] = []
        terminated = self.termination_condition.is_met(observation)
        self.agent.reset()

        for window in range(planning_windows):
            if terminated:
                break
            if self.temporal_horizon > 1:
                self.agent.reset()
                time_steps = range(self.temporal_horizon)
            else:
                time_steps = (window,)

            for time_step in time_steps:
                self.agent.observe(observation.as_array(), time_step=time_step)
                self.agent.infer_states()
                self.agent.infer_policies()
                if self.action_constraint is None:
                    selected_action = self.agent.select_action()
                else:
                    allowed_actions = self.action_constraint.allowed_actions(observation)
                    selected_action = self.agent.select_action(allowed_actions)
                if selected_action is None:
                    continue

                action = NavigationAction.from_sequence(selected_action)
                self.action_executor.execute(action)
                self.action_executor.wait_for_completion()
                actions.append(action)

                observation = self.observation_source.read_observation()
                observations.append(observation)
                terminated = self.termination_condition.is_met(observation)
                if terminated:
                    break

        return NavigationRuntimeResult(
            observations=tuple(observations),
            actions=tuple(actions),
            terminated=terminated,
        )
