"""Hardware-independent interfaces used by the navigation runtime."""

from __future__ import annotations

from collections.abc import Collection
from typing import Protocol, runtime_checkable

from .models import NavigationAction, Observation


class NavigationError(RuntimeError):
    """Base error for navigation runtime failures."""


class ObservationUnavailableError(NavigationError):
    """Raised when a complete, current observation cannot be produced."""


class StaleObservationError(ObservationUnavailableError):
    """Raised when sensor measurements are older than their allowed age."""


class ActionExecutionError(NavigationError):
    """Raised when an actuator cannot complete a navigation action."""


@runtime_checkable
class ObservationSource(Protocol):
    """Provide observations without exposing a particular sensor API."""

    def read_observation(self) -> Observation:
        """Return the latest complete observation or raise a navigation error."""


@runtime_checkable
class ActionExecutor(Protocol):
    """Execute grid navigation actions using any actuator implementation."""

    def execute(self, action: NavigationAction) -> None:
        """Start executing an action."""

    def wait_for_completion(self) -> None:
        """Block until the current action completes or fails."""


@runtime_checkable
class TerminationCondition(Protocol):
    """Decide whether a navigation episode is complete."""

    def is_met(self, observation: Observation) -> bool:
        """Return whether navigation should stop after an observation."""


@runtime_checkable
class ActionConstraint(Protocol):
    """Provide actions that are safe to execute for an observation."""

    def allowed_actions(self, observation: Observation) -> Collection[NavigationAction]:
        """Return the currently executable navigation actions."""
