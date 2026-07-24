# Active-Inference Navigation Agent

[![CI](https://github.com/Diluna98/active-inference-navigation-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Diluna98/active-inference-navigation-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A continuous-observation active-inference agent that navigates toward an
unknown radio-frequency source using position and RSSI measurements. The agent
is built with [PyAIF](https://github.com/Diluna98/python_active_inference).

![Continuous active-inference navigation from different starting and source locations](docs/results/active_inference_navigation.gif)

The animation shows four deterministic navigation episodes. Each agent starts
at a different position and searches for a different RSSI source. The panels
display the travelled path, source distance, received signal strength, and
completion status.

## Overview

The environment provides a continuous observation at every time step:

```text
observation = [x position, y position, RSSI]
```

The generative model represents three categorical hidden-state factors:

```text
state = [x cell, y cell, source cell]
```

The agent updates its beliefs about its current position and the unknown source
location, evaluates candidate movement policies, and selects actions that are
expected to produce preferred high-strength RSSI observations.

The implementation supports:

- Continuous Gaussian likelihoods for position and RSSI
- Discrete spatial hidden states
- Cardinal movement policies on a bounded grid
- Shallow state and policy inference
- Deep temporal inference over multi-step policies
- Deterministic, reproducible simulation scenarios
- Optional parallel policy evaluation through PyAIF

The reusable navigation package keeps the domain likelihood in
`RssiNavigationLikelihood`. Set `paper_compatible_likelihood=True` to reproduce
the RSSI preference and fixed `20×20` Fisher-information reference grid used by
the paper's meta-inference experiments:

```python
config = NavigationAgentConfig(
    goal_resolution=2,
    temporal_horizon=3,
    message_passing_iterations=10,
    policy_samples=500,
    exact_state_limit=1,
    random_seed=7,
    paper_compatible_likelihood=True,
)
```

For real sensors, keep the task inference configuration and replace or
calibrate the likelihood's observation model, noise parameters, and sensor
preprocessing.

## Installation

Clone the repository and install it in a virtual environment:

```bash
git clone https://github.com/Diluna98/active-inference-navigation-agent.git
cd active-inference-navigation-agent
python -m venv .venv
python -m pip install -e .
```

Install the development tools when running tests or building distributions:

```bash
python -m pip install -e ".[dev]"
```

## Command-line usage

Run a shallow-inference episode:

```bash
active-inference-navigate --seed 7 --planning-windows 20
```

Run deep temporal inference with a three-step horizon:

```bash
active-inference-navigate \
  --seed 7 \
  --temporal-horizon 3 \
  --goal-resolution 2 \
  --planning-windows 8 \
  --policy-samples 300
```

The command reports the initial, minimum, and final source distances together
with the number of movements and completion status.

## Python API

```python
from active_inference_navigation import (
    GridNavigationEnvironment,
    NavigationAgentConfig,
    run_navigation_episode,
)

config = NavigationAgentConfig(
    model_size=20,
    goal_resolution=10,
    temporal_horizon=1,
    random_seed=7,
)
environment = GridNavigationEnvironment(
    start=(487.5, 487.5),
    goal=(212.5, 312.5),
    random_seed=7,
)

result = run_navigation_episode(
    config=config,
    environment=environment,
    planning_windows=20,
)

print(result.positions)
print(result.distances)
print(result.reached_goal)
```

Set `temporal_horizon=1` for shallow inference. Values greater than one enable
deep temporal inference.

## Recreate the animation

Generate the four-scenario GIF:

```bash
active-inference-navigation-gif
```

The animation is written to:

```text
docs/results/active_inference_navigation.gif
```

## Project structure

```text
active-inference-navigation-agent/
├── src/active_inference_navigation/
│   ├── agent.py
│   ├── animation.py
│   ├── cli.py
│   ├── environment.py
│   ├── likelihoods.py
│   └── simulation.py
├── tests/
├── docs/results/
├── pyproject.toml
└── LICENSE
```

## Development

Run the quality checks locally:

```bash
ruff check .
pytest -q
python -m build
```

GitHub Actions runs linting, tests, and distribution builds on Python 3.10 and
3.11.

## License

This project is available under the [MIT License](LICENSE).

Copyright © 2026 Diluna A. Warnakulasuriya.
