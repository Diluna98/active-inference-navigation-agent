# Two-dimensional MOS benchmark

This benchmark replaces further development of the custom inspection task. It
adapts the established Multi-Object Search POMDP structure to factorized Active
Inference and borrows the spatial abstraction idea from Multi-Resolution POMDP
Planning for Multi-Object Search in 3D.

References:

- [Multi-Resolution POMDP Planning for Multi-Object Search in 3D](https://h2r.cs.brown.edu/wp-content/uploads/zhengmulti21.pdf)
- [3D-MOS project and implementation](https://kaiyuzheng.me/3D-MOS/html/)
- [Object-Oriented POMDP Multi-Object Search](https://h2r.cs.brown.edu/object-oriented-pomdps/)

## Initial scope

The first milestone is a single-object 2D-MOS problem. This is the simplest
controlled member of the MOS family and is intended to validate fixed belief
resolution and planning-depth effects before adding multiple hidden objects.

The robot position `(x, y)` is observed exactly. The target location is hidden
and represented at resolution `gamma**2`, with `gamma` in `{2, 5, 10, 20}`.
Range and ray-cast occlusion determine the probability of detecting the target.
Seeded room-like maps provide known obstacles, and target positions are sampled
from free cells. The action space contains four cardinal MOVE actions and FIND.
A successful FIND requires the real target to be visible; premature FIND and
collisions incur task cost.

The spatial-safety observation describes whether the resulting `(x, y)` state
is free or blocked and supplies obstacle preferences during policy evaluation.
An attempted blocked movement leaves the robot in its previous free state, so
that event is recorded separately as an environment collision metric rather
than emitted as a contradictory blocked-occupancy observation.

All agents use `FilteredRecedingHorizonInference`. Action depth `T=1,2,3`
produces 5, 25, and 125 policies. No adaptive gamma or T controller is present.

## Fixed diagnostic

Run a quick smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_mos_fixed.py `
  --target-seeds 0 --resolutions 2 --depths 1 --max-steps 5 `
  --output-dir docs/results/mos_fixed_smoke
```

Run one complete 12-configuration instance:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_mos_fixed.py `
  --target-seeds 0 --max-steps 50 `
  --output-dir docs/results/mos_fixed_first_instance
```

Run the initial multi-instance diagnostic across four target placements:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_mos_fixed.py `
  --map-seeds 0 --target-seeds 0 1 2 3 --observation-seeds 0 `
  --max-steps 50 --output-dir docs/results/mos_fixed
```

The output includes episode behavior, decision-level state and policy timing,
VFE, target-belief entropy, policy EFE/posterior, initial-action disagreement,
false FIND attempts, collisions, cumulative inference time, and task cost.
Each decision row distinguishes the observation used for inference from the
post-action observation and position. The summary records the exact policy
count, software versions, and pairwise `T=1` versus `T=3` and `T=2` versus
`T=3` action disagreements at each resolution. Trajectory-prefix comparisons
stop at the first differing action, ensuring that the compared decision was
reached through the same action and observation history.

The four-target fixed pilot showed the following success rates:

| `gamma` | `T=1` | `T=2` | `T=3` |
| ---: | ---: | ---: | ---: |
| 2 | 0% | 0% | 0% |
| 5 | 0% | 50% | 50% |
| 10 | 25% | 50% | 75% |
| 20 | 50% | 75% | 100% |

The optimized PyAIF contraction path preserved every aggregate success rate
and mean step count from the original run. Policy evaluation speedups ranged
from 1.5x to 55.0x across the 12 configurations. At `(gamma=20,T=3)`, mean
policy time fell from about 9.44 s to 171 ms per decision in the original
comparison. Stable deterministic tie handling prevents contraction-order
roundoff from changing actions when policy posterior values agree to numerical
precision. The auditable final outputs are in
`docs/results/mos_fixed_optimized_stable/`.

The pilot showed the following qualitative pattern:

- `gamma=2` failed for every tested T.
- `gamma=5` failed at `T=1` and succeeded at `T=2,3`.
- `gamma=10,20` succeeded at every T.
- `gamma=10,T=2` gave the lowest observed successful task cost.
- All horizons selected the same first action in the four initial instances at
  every resolution. Nevertheless, `T=1` and `T=3` later diverged from an
  identical trajectory prefix in every instance; `T=2` and `T=3` diverged in
  three or four of four instances, depending on resolution.
- In the committed reproducibility run, mean decision time ranged from about
  2.7 ms at `(2,1)` to 448 ms at `(10,3)`. Wall-clock timings are
  machine-dependent; action traces and behavioral outputs are the
  reproducibility criterion.

These are diagnostic observations from one map and four target placements, not
benchmark conclusions.
The fixed sweep must be repeated across maps, target locations, and observation
seeds before defining matched-state interventions or any adaptive controller.

## Paired robustness protocol

The robustness runner generates one deterministic domain instance per master
seed. Each seed jointly determines obstacle geometry, a free robot start in the
left half of the map, a target in the right half, sensor range, and the stochastic
observation realization. Every `(gamma, T)` allocation is evaluated on exactly
the same generated instance, providing paired comparisons without a Cartesian
explosion of independent seed lists.

Run a one-configuration smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_mos_robustness.py `
  --instance-seeds 0 --resolutions 2 --depths 1 --max-steps 5 `
  --message-passing-iterations 2 --output-dir docs/results/mos_robustness_smoke
```

Run the planned 30-instance fixed-allocation experiment:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_mos_robustness.py `
  --num-instances 30 --seed-offset 0 --max-steps 50 `
  --output-dir docs/results/mos_robustness_30
```

This evaluates 360 paired episodes. Keep `--policy-workers 1` for interpretable
latency comparisons; increasing it changes the execution regime. The output
contains `instances.csv`, `episodes.csv`, `decisions.csv`, and `summary.json`.
In addition to the fixed diagnostic fields, the summary reports Wilson 95%
success intervals, median and 95th-percentile decision latency, paired depth and
resolution effects, and per-instance allocation regimes. A positive
`mean_task_cost_left_minus_right` means the right-hand, more expensive allocation
reduced task cost. The runner remains strictly diagnostic and contains no adaptive
controller.

## Deliberate differences from 3D-MOS

The published 3D-MOS implementation uses octree beliefs, MOVE/LOOK/FIND
actions, frustum observations, and POUCT. This initial PyAIF adaptation uses a
dense 2D categorical target factor, automatic range/occlusion observations,
MOVE/FIND actions, and enumerated receding-horizon policies. The simplification
keeps `gamma=20` at 400 target states; a dense 3D factor would contain 8,000
states and would require sparse or on-demand likelihood evaluation.

The intended research question is not whether multi-resolution POMDP planning
works. It is whether representational resolution and temporal planning depth
have distinct, context-dependent value under Active Inference. That question
will be evaluated only after the fixed diagnostic is sufficiently replicated.
