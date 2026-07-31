# Changelog

All notable changes to this project are documented here.

## [0.2.2] - 2026-07-31

### Added

- Configurable final arena heading for repeatable robot orientation after every
  grid action.
- Configurable post-motion settling interval before the next sensor
  observation.

### Changed

- TurtleBot actions now restore arena `+y` by default, including stay actions,
  so PCB-antenna orientation does not depend on the preceding movement.
- ROS navigation, actuator testing, and packaged YAML propagate the new motion
  settings.

## [0.2.1] - 2026-07-28

### Fixed

- Normalize nested and object-array action outputs returned by PyAIF before
  converting them to hardware-independent navigation actions.
- Preserve the public two-component cardinal action shape across supported
  Python and NumPy combinations.

## [0.2.0] - 2026-07-28

### Added

- Hardware-independent `Observation`, `NavigationAction`, and `AxisAction`
  models.
- `ObservationSource`, `ActionExecutor`, `TerminationCondition`, and
  `ActionConstraint` protocols.
- Generic `NavigationRuntime` for simulation and physical robots.
- Simulation observation/action adapters that preserve the existing public
  episode API.
- Configurable rectangular grid geometry and bidirectional arena/odometry frame
  transforms.
- ROS 2 odometry/RSSI observation adapter with freshness checks and configurable
  median aggregation.
- Closed-loop TurtleBot `Twist` executor with boundary enforcement, exact
  cell-centre targets, proportional heading control, and stop-on-error behavior.
- Real-world calibrated dBm likelihood based on measurements from 1.0 m to
  8.6 m.
- Persistent RSSI goal termination and a configurable planning-window safety
  limit.
- Explicit actuator-test and deterministic odometry-only return-home commands.
- Typed YAML configuration for geometry, frames, topics, inference, sensors,
  motion, likelihoods, experiments, and termination.
- Packaged default YAML so installed commands work outside a source checkout.
- Boundary-aware policy masking that selects the next-best valid policy at arena
  edges.

### Changed

- The real-world default arena is 7 m by 7 m with a 20 by 20 movement grid and
  0.35 m cells.
- The default source-belief grid is 10 by 10 with 0.7 m cells.
- Shallow inference now uses the same five joint cardinal policies as deep
  inference.
- `run_navigation_episode()` now delegates orchestration to
  `NavigationRuntime`.
- Real-world coordinates are transformed consistently between raw odometry and
  continuous arena observations.

### Compatibility

- `GridNavigationEnvironment`, `NavigationAgentConfig`,
  `NavigationEpisodeResult`, `build_navigation_agent()`, and
  `run_navigation_episode()` remain publicly available.
- Diagonal actions that were previously accepted by the simulator now raise a
  validation error.
- Cardinal-only policies change trajectories and may require more movements than
  the diagonal behavior present in version 0.1.0.

### Known limitations

- ROS 2 packages must be supplied by a sourced ROS installation.
- The calibrated likelihood is specific to the measured radio and arena setup.
- Wheel odometry requires a repeatable physical start pose after the robot is
  lifted.
- Return-home follows an obstacle-free Manhattan path and does not perform
  obstacle avoidance.

## [0.1.0] - 2026-07-25

- Initial continuous-observation Active Inference navigation release.

[0.2.0]: https://github.com/Diluna98/active-inference-navigation-agent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Diluna98/active-inference-navigation-agent/releases/tag/v0.1.0
