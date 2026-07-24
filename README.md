# Active-Inference Navigation Agent

Continuous-observation RSSI source navigation built with
[PyAIF](https://github.com/Diluna98/python_active_inference).

![Four active-inference navigation episodes with different starts and RSSI sources](docs/results/active_inference_navigation.gif)

The animation shows four deterministic episodes running simultaneously. In
each panel, the agent begins at a different location and navigates toward a
different RSSI source while its continuous signal observation increases.

## What this repository contains

- A continuous RSSI likelihood over discrete spatial hidden states
- Shallow and deep-temporal active-inference configurations
- A deterministic grid-navigation environment
- Reproducible multi-scenario simulation and animation
- Automated linting, tests, and package builds

PyAIF provides the reusable inference machinery. This repository contains only
the navigation-specific likelihood, environment, configuration, and experiment.
It does not contain the neural-network RSSI localization project or the
hierarchical model-selection experiments from the ICRA study.

## Installation

Create a virtual environment and install the package:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

During PyAIF v0.2 validation, the dependency points to the tested continuous
observations feature branch. After PyAIF v0.2.0 is published, it will be
replaced with `pyaif-toolkit>=0.2,<0.3`.

## Run a navigation episode

```bash
active-inference-navigate --seed 7 --planning-windows 20
```

From Python:

```python
from active_inference_navigation import (
    NavigationAgentConfig,
    run_navigation_episode,
)

result = run_navigation_episode(
    config=NavigationAgentConfig(random_seed=7),
    planning_windows=20,
)
print(result.distances)
```

Use `--temporal-horizon 1` for shallow inference or a value greater than one
for deep temporal inference.

## Recreate the animation

```bash
active-inference-navigation-gif
```

The command writes
`docs/results/active_inference_navigation.gif`.

## Validation

```bash
ruff check .
pytest -q
python -m build
```

## License

MIT License. Copyright (c) 2026 Diluna A. Warnakulasuriya.

