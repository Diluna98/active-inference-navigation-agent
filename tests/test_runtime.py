from collections import deque

import numpy as np
import pytest

from active_inference_navigation.models import NavigationAction, Observation
from active_inference_navigation.runtime import NavigationRuntime


class FakeAgent:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.observed = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def observe(self, observation, *, time_step):
        self.observed.append((np.asarray(observation), time_step))

    def infer_states(self):
        return None

    def infer_policies(self):
        return None

    def select_action(self):
        return next(self.actions)


class FakeObservationSource:
    def __init__(self, observations, events):
        self.observations = deque(observations)
        self.events = events

    def read_observation(self):
        observation = self.observations.popleft()
        self.events.append(("read", observation.x))
        return observation


class FakeExecutor:
    def __init__(self, events):
        self.events = events

    def execute(self, action):
        self.events.append(("execute", action.cell_delta))

    def wait_for_completion(self):
        self.events.append(("wait", None))


class XTermination:
    def __init__(self, target):
        self.target = target

    def is_met(self, observation):
        return observation.x >= self.target


def test_runtime_reads_real_observation_after_action_completion():
    events = []
    source = FakeObservationSource(
        [Observation(0.0, 0.0, -70.0), Observation(1.0, 0.0, -60.0)],
        events,
    )
    runtime = NavigationRuntime(
        agent=FakeAgent([(2, 0)]),
        observation_source=source,
        action_executor=FakeExecutor(events),
        termination_condition=XTermination(1.0),
    )

    result = runtime.run(planning_windows=3)

    assert events == [
        ("read", 0.0),
        ("execute", (1, 0)),
        ("wait", None),
        ("read", 1.0),
    ]
    assert result.terminated
    assert result.actions == (NavigationAction.from_sequence((2, 0)),)


def test_runtime_rejects_invalid_agent_action_before_execution():
    events = []
    runtime = NavigationRuntime(
        agent=FakeAgent([(2, 2)]),
        observation_source=FakeObservationSource([Observation(0.0, 0.0, -70.0)], events),
        action_executor=FakeExecutor(events),
        termination_condition=XTermination(2.0),
    )

    with pytest.raises(ValueError, match="Diagonal"):
        runtime.run(planning_windows=1)

    assert not any(event[0] == "execute" for event in events)


def test_runtime_resets_agent_for_each_deep_planning_window():
    events = []
    agent = FakeAgent([(0, 0), (0, 0), (0, 0), (0, 0)])
    runtime = NavigationRuntime(
        agent=agent,
        observation_source=FakeObservationSource(
            [Observation(float(index), 0.0, -70.0) for index in range(5)],
            events,
        ),
        action_executor=FakeExecutor(events),
        termination_condition=XTermination(99.0),
        temporal_horizon=2,
    )

    runtime.run(planning_windows=2)

    assert agent.reset_count == 3
    assert [time_step for _, time_step in agent.observed] == [0, 1, 0, 1]
