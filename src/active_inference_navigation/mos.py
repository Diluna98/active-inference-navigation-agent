"""Two-dimensional object-search POMDP adapted to factorized Active Inference.

The domain follows the established MOS structure: robot pose is known, object
location is hidden, observations are range- and occlusion-dependent, and FIND
terminates successfully only when the object is visible.  This first benchmark
uses one object so belief resolution and planning depth can be diagnosed before
adding multi-object coupling or any adaptive controller.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from math import exp, hypot
from typing import Any

import numpy as np
from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    FilteredRecedingHorizonInference,
    GenerativeModel,
    utils,
)

GRID_SIZE = 20
TARGET_RESOLUTIONS = (2, 5, 10, 20)


class MOSAction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    FIND = 4


class DetectionOutcome(IntEnum):
    NO_DETECTION = 0
    DETECTION = 1


class FindOutcome(IntEnum):
    SEARCHING = 0
    FOUND = 1
    FALSE_FIND = 2


class CollisionOutcome(IntEnum):
    FREE = 0
    BLOCKED = 1


class OperationState(IntEnum):
    MOVE = 0
    FIND = 1


def _object_array(*arrays: np.ndarray) -> np.ndarray:
    result = np.empty(len(arrays), dtype=object)
    for index, array in enumerate(arrays):
        result[index] = np.asarray(array, dtype=float)
    return result


@dataclass(frozen=True)
class MOSLayout:
    """Known 2D search map with deterministic room-like occluders."""

    size: int = GRID_SIZE
    map_seed: int = 0
    start: tuple[int, int] = (1, 1)
    sensor_range: float = 6.0

    @cached_property
    def blocked(self) -> frozenset[tuple[int, int]]:
        rng = np.random.default_rng(self.map_seed)
        first_x = int(rng.integers(6, 9))
        second_x = int(rng.integers(12, 15))
        first_door = int(rng.integers(4, 9))
        second_door = int(rng.integers(10, 16))
        cells = {
            (first_x, y) for y in range(1, self.size - 1) if y not in (first_door, first_door + 1)
        }
        cells |= {
            (second_x, y)
            for y in range(1, self.size - 1)
            if y not in (second_door, second_door + 1)
        }
        return frozenset(cells)

    def is_free(self, position: tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.size and 0 <= y < self.size and position not in self.blocked

    def move(self, position: tuple[int, int], action: MOSAction) -> tuple[int, int]:
        if action is MOSAction.FIND:
            return position
        dx, dy = {
            MOSAction.UP: (0, 1),
            MOSAction.DOWN: (0, -1),
            MOSAction.LEFT: (-1, 0),
            MOSAction.RIGHT: (1, 0),
        }[action]
        candidate = (position[0] + dx, position[1] + dy)
        return candidate if self.is_free(candidate) else position

    def visibility(self, robot: tuple[int, int], target: tuple[int, int]) -> float:
        """Return continuous detection quality from range and ray occlusion."""

        distance = hypot(robot[0] - target[0], robot[1] - target[1])
        if distance > self.sensor_range + 3.0:
            return 0.0
        ray_steps = max(abs(target[0] - robot[0]), abs(target[1] - robot[1]))
        occluded = 0
        for index in range(1, ray_steps):
            fraction = index / ray_steps
            point = (
                round(robot[0] + fraction * (target[0] - robot[0])),
                round(robot[1] + fraction * (target[1] - robot[1])),
            )
            occluded += point in self.blocked
        range_quality = 1.0 / (1.0 + exp((distance - self.sensor_range) / 0.8))
        return float(range_quality * exp(-2.0 * occluded))

    def target_is_visible(self, robot: tuple[int, int], target: tuple[int, int]) -> bool:
        return self.visibility(robot, target) >= 0.5

    def sample_target(self, seed: int) -> tuple[int, int]:
        candidates = [
            (x, y)
            for x in range(self.size // 2, self.size)
            for y in range(self.size)
            if self.is_free((x, y))
        ]
        return candidates[int(np.random.default_rng(seed).integers(len(candidates)))]

    def sample_start(self, seed: int) -> tuple[int, int]:
        """Sample a reproducible free start from the map's left half."""

        candidates = [
            (x, y) for x in range(self.size // 2) for y in range(self.size) if self.is_free((x, y))
        ]
        return candidates[int(np.random.default_rng(seed).integers(len(candidates)))]


@dataclass(frozen=True)
class MOSInstance:
    """One paired benchmark instance shared by every allocation setting."""

    instance_seed: int
    layout: MOSLayout
    target_seed: int
    observation_seed: int
    start_seed: int
    target: tuple[int, int]


def sample_mos_instance(seed: int) -> MOSInstance:
    """Generate a deterministic map, start, target, and sensor realization."""

    rng = np.random.default_rng(seed)
    map_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    start_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    target_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    observation_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    sensor_range = float(rng.choice((5.0, 6.0, 7.0)))
    base_layout = MOSLayout(map_seed=map_seed, sensor_range=sensor_range)
    layout = MOSLayout(
        map_seed=map_seed,
        start=base_layout.sample_start(start_seed),
        sensor_range=sensor_range,
    )
    return MOSInstance(
        instance_seed=seed,
        layout=layout,
        target_seed=target_seed,
        observation_seed=observation_seed,
        start_seed=start_seed,
        target=layout.sample_target(target_seed),
    )


def target_state(
    position: tuple[int, int],
    resolution: int,
    grid_size: int = GRID_SIZE,
) -> int:
    cell_size = grid_size / resolution
    x_index = min(resolution - 1, int(position[0] / cell_size))
    y_index = min(resolution - 1, int(position[1] / cell_size))
    return y_index * resolution + x_index


def target_state_position(
    state: int,
    resolution: int,
    grid_size: int = GRID_SIZE,
) -> tuple[int, int]:
    cell_size = grid_size / resolution
    x_index, y_index = state % resolution, state // resolution
    return (
        min(grid_size - 1, int((x_index + 0.5) * cell_size)),
        min(grid_size - 1, int((y_index + 0.5) * cell_size)),
    )


def detection_distribution(
    robot: tuple[int, int],
    target: tuple[int, int],
    layout: MOSLayout,
) -> np.ndarray:
    quality = layout.visibility(robot, target)
    hit_probability = 0.02 + 0.93 * quality
    return np.asarray([1.0 - hit_probability, hit_probability], dtype=float)


class MOSEnvironment:
    """Single-object MOS simulator with common-random-number detections."""

    def __init__(
        self,
        *,
        layout: MOSLayout | None = None,
        target: tuple[int, int] | None = None,
        target_seed: int = 0,
        observation_seed: int = 0,
    ) -> None:
        self.layout = layout or MOSLayout()
        self.target = target or self.layout.sample_target(target_seed)
        self.observation_seed = observation_seed
        if not self.layout.is_free(self.layout.start):
            raise ValueError("start must be a free cell")
        if not self.layout.is_free(self.target):
            raise ValueError("target must be a free cell")
        self.position = self.layout.start
        self._operation = OperationState.MOVE
        self._collision = False
        self._rng = np.random.default_rng(observation_seed)

    @property
    def last_collision(self) -> bool:
        """Whether the most recently attempted movement was blocked."""

        return self._collision

    def reset(self, *, noise_quantile: float | None = None) -> tuple[int, int, int, int, int]:
        self.position = self.layout.start
        self._operation = OperationState.MOVE
        self._collision = False
        self._rng = np.random.default_rng(self.observation_seed)
        return self.observe(noise_quantile=noise_quantile)

    def observe(self, *, noise_quantile: float | None = None) -> tuple[int, int, int, int, int]:
        if noise_quantile is None:
            noise_quantile = float(self._rng.random())
        distribution = detection_distribution(self.position, self.target, self.layout)
        detection = int(
            min(
                np.searchsorted(np.cumsum(distribution), noise_quantile, side="right"),
                len(distribution) - 1,
            )
        )
        if self._operation is OperationState.FIND:
            find = (
                FindOutcome.FOUND
                if self.layout.target_is_visible(self.position, self.target)
                else FindOutcome.FALSE_FIND
            )
        else:
            find = FindOutcome.SEARCHING
        return (
            self.position[0],
            self.position[1],
            detection,
            int(find),
            int(
                CollisionOutcome.FREE
                if self.layout.is_free(self.position)
                else CollisionOutcome.BLOCKED
            ),
        )

    def step(
        self,
        action: MOSAction | int,
        *,
        noise_quantile: float | None = None,
    ) -> tuple[tuple[int, int, int, int, int], bool]:
        action = MOSAction(action)
        previous = self.position
        self.position = self.layout.move(previous, action)
        self._operation = OperationState.FIND if action is MOSAction.FIND else OperationState.MOVE
        self._collision = action is not MOSAction.FIND and self.position == previous
        observation = self.observe(noise_quantile=noise_quantile)
        return observation, observation[3] == FindOutcome.FOUND


def _axis_transitions(size: int) -> np.ndarray:
    transitions = np.zeros((size, size, 3), dtype=float)
    for source in range(size):
        transitions[source, source, 0] = 1.0
        transitions[max(0, source - 1), source, 1] = 1.0
        transitions[min(size - 1, source + 1), source, 2] = 1.0
    return transitions


def _target_transitions(resolution: int) -> np.ndarray:
    return np.eye(resolution**2, dtype=float)[:, :, None]


def _operation_transitions() -> np.ndarray:
    result = np.zeros((len(OperationState), len(OperationState), len(OperationState)))
    result[OperationState.MOVE, :, OperationState.MOVE] = 1.0
    result[OperationState.FIND, :, OperationState.FIND] = 1.0
    return result


def _target_prior(resolution: int, layout: MOSLayout) -> np.ndarray:
    prior = np.zeros(resolution**2, dtype=float)
    for x in range(layout.size):
        for y in range(layout.size):
            if layout.is_free((x, y)):
                prior[target_state((x, y), resolution, layout.size)] += 1.0
    return prior / prior.sum()


def _detection_likelihood(resolution: int, layout: MOSLayout) -> np.ndarray:
    result = np.empty(
        (len(DetectionOutcome), layout.size, layout.size, resolution**2),
        dtype=float,
    )
    for x in range(layout.size):
        for y in range(layout.size):
            for state in range(resolution**2):
                result[:, x, y, state] = detection_distribution(
                    (x, y),
                    target_state_position(state, resolution, layout.size),
                    layout,
                )
    return result


def _find_likelihood(resolution: int, layout: MOSLayout) -> np.ndarray:
    result = np.zeros(
        (
            len(FindOutcome),
            layout.size,
            layout.size,
            resolution**2,
            len(OperationState),
        ),
        dtype=float,
    )
    for x in range(layout.size):
        for y in range(layout.size):
            for state in range(resolution**2):
                target = target_state_position(state, resolution, layout.size)
                result[FindOutcome.SEARCHING, x, y, state, OperationState.MOVE] = 1.0
                find = (
                    FindOutcome.FOUND
                    if layout.target_is_visible((x, y), target)
                    else FindOutcome.FALSE_FIND
                )
                result[find, x, y, state, OperationState.FIND] = 1.0
    return result


def _collision_likelihood(layout: MOSLayout) -> np.ndarray:
    result = np.zeros((len(CollisionOutcome), layout.size, layout.size), dtype=float)
    result[CollisionOutcome.FREE, :, :] = 1.0
    for x, y in layout.blocked:
        result[CollisionOutcome.FREE, x, y] = 0.0
        result[CollisionOutcome.BLOCKED, x, y] = 1.0
    return result


def _policies(action_depth: int, grid_size: int) -> list[np.ndarray]:
    candidates = utils.construct_policies(
        (grid_size, grid_size, 1, len(OperationState)),
        (3, 3, 1, len(OperationState)),
        action_depth,
        [0, 1, 3],
    )
    valid_controls = {
        (0, 2, OperationState.MOVE),
        (0, 1, OperationState.MOVE),
        (1, 0, OperationState.MOVE),
        (2, 0, OperationState.MOVE),
        (0, 0, OperationState.FIND),
    }
    return [
        policy
        for policy in candidates
        if all((int(row[0]), int(row[1]), int(row[3])) in valid_controls for row in policy)
    ]


@dataclass(frozen=True)
class MOSAgentConfig:
    target_resolution: int = 10
    action_depth: int = 2
    message_passing_iterations: int = 10
    policy_workers: int = 1

    def __post_init__(self) -> None:
        if self.target_resolution not in TARGET_RESOLUTIONS:
            raise ValueError(f"target_resolution must be one of {TARGET_RESOLUTIONS}")
        if self.action_depth not in (1, 2, 3):
            raise ValueError("action_depth must be 1, 2, or 3")
        if self.message_passing_iterations < 1:
            raise ValueError("message_passing_iterations must be positive")
        if self.policy_workers < 1:
            raise ValueError("policy_workers must be positive")


class MOSAgent:
    """Small wrapper that initializes generated categorical arrays efficiently."""

    def __init__(self, agent: ActiveInfAgent) -> None:
        self._agent = agent

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    def __deepcopy__(self, memo: dict[int, Any]) -> MOSAgent:
        clone = type(self)(copy.deepcopy(self._agent, memo))
        memo[id(self)] = clone
        return clone

    def reset(self) -> None:
        if not hasattr(self._agent, "D"):
            self._initialize_normalized_model()
        self._agent.reset(normalize=False)

    def _initialize_normalized_model(self) -> None:
        def normalize_axis_zero(values: np.ndarray) -> np.ndarray:
            array = np.asarray(values, dtype=float).copy()
            denominator = array.sum(axis=0, keepdims=True)
            return array / np.where(denominator > 0, denominator, 1.0)

        self._agent.A = _object_array(*(normalize_axis_zero(values) for values in self._agent.pA))
        self._agent.B = _object_array(*(normalize_axis_zero(values) for values in self._agent.pB))
        self._agent.D = _object_array(*(normalize_axis_zero(values) for values in self._agent.pD))
        self._agent.E = normalize_axis_zero(self._agent.pE)
        preferences = []
        for values in self._agent.pC:
            shifted = np.asarray(values, dtype=float) - np.max(values, axis=0, keepdims=True)
            probability = np.exp(shifted)
            probability /= probability.sum(axis=0, keepdims=True)
            preferences.append(np.log(np.clip(probability, 1e-16, 1.0)))
        self._agent.C = _object_array(*preferences)
        self._agent.transposed_B = self._agent._transpose_B_matrix()


def build_mos_agent(
    config: MOSAgentConfig | None = None,
    *,
    layout: MOSLayout | None = None,
) -> MOSAgent:
    config = config or MOSAgentConfig()
    layout = layout or MOSLayout()
    horizon = config.action_depth + 1
    x_prior = np.zeros(layout.size)
    y_prior = np.zeros(layout.size)
    x_prior[layout.start[0]] = 1.0
    y_prior[layout.start[1]] = 1.0
    operation_prior = np.zeros(len(OperationState))
    operation_prior[OperationState.MOVE] = 1.0

    model = GenerativeModel(
        B=_object_array(
            _axis_transitions(layout.size),
            _axis_transitions(layout.size),
            _target_transitions(config.target_resolution),
            _operation_transitions(),
        ),
        D=_object_array(
            x_prior,
            y_prior,
            _target_prior(config.target_resolution, layout),
            operation_prior,
        ),
        controls_dim=(3, 3, 1, len(OperationState)),
        controllable_factors=(0, 1, 3),
        policies=_policies(config.action_depth, layout.size),
    )
    find_preferences = np.zeros((len(FindOutcome), horizon))
    find_preferences[FindOutcome.FOUND, :] = 9.0
    find_preferences[FindOutcome.FALSE_FIND, :] = -12.0
    collision_preferences = np.zeros((len(CollisionOutcome), horizon))
    collision_preferences[CollisionOutcome.BLOCKED, :] = -12.0
    likelihood = CategoricalLikelihood(
        A=_object_array(
            np.eye(layout.size),
            np.eye(layout.size),
            _detection_likelihood(config.target_resolution, layout),
            _find_likelihood(config.target_resolution, layout),
            _collision_likelihood(layout),
        ),
        preferences=_object_array(
            np.zeros((layout.size, horizon)),
            np.zeros((layout.size, horizon)),
            np.zeros((len(DetectionOutcome), horizon)),
            find_preferences,
            collision_preferences,
        ),
        modality_dependencies=((0,), (1,), (0, 1, 2), (0, 1, 2, 3), (0, 1)),
    )
    inference = FilteredRecedingHorizonInference(
        horizon=horizon,
        message_passing_iterations=config.message_passing_iterations,
        policy_workers=config.policy_workers,
    )
    return MOSAgent(
        ActiveInfAgent(
            model=model,
            likelihood=likelihood,
            inference=inference,
            action_selection="deterministic",
        )
    )


def selected_mos_action(agent: MOSAgent) -> MOSAction:
    selected = np.asarray(agent.select_action(), dtype=object).reshape(-1)
    controls = tuple(int(np.asarray(selected[index]).reshape(-1)[0]) for index in (0, 1, 3))
    mapping = {
        (0, 2, OperationState.MOVE): MOSAction.UP,
        (0, 1, OperationState.MOVE): MOSAction.DOWN,
        (1, 0, OperationState.MOVE): MOSAction.LEFT,
        (2, 0, OperationState.MOVE): MOSAction.RIGHT,
        (0, 0, OperationState.FIND): MOSAction.FIND,
    }
    try:
        return mapping[controls]
    except KeyError as error:
        raise RuntimeError(f"PyAIF selected an invalid MOS control tuple: {controls}") from error


def mos_action_controls(action: MOSAction | int) -> np.ndarray:
    return np.asarray(
        {
            MOSAction.UP: (0, 2, 0, OperationState.MOVE),
            MOSAction.DOWN: (0, 1, 0, OperationState.MOVE),
            MOSAction.LEFT: (1, 0, 0, OperationState.MOVE),
            MOSAction.RIGHT: (2, 0, 0, OperationState.MOVE),
            MOSAction.FIND: (0, 0, 0, OperationState.FIND),
        }[MOSAction(action)],
        dtype=int,
    )
