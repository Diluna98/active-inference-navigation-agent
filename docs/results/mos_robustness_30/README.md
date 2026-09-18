# Paired 2D-MOS robustness result

This directory contains the completed 30-instance fixed-allocation diagnostic:

- `instances.csv`: the 30 deterministic randomized domain instances;
- `episodes.csv`: 360 episode-level outcomes, one for every
  `(instance, gamma, T)` combination;
- `decisions.csv`: 12,080 decision-level belief, action, VFE, entropy, and
  timing records;
- `summary.json`: aggregate confidence intervals, paired comparisons, action
  disagreements, and allocation-regime classifications.

## Integrity checks

- All instance seeds from 0 through 29 are present.
- Each instance has exactly 12 episodes.
- All 360 episode identifiers are unique.
- The run used PyAIF 0.4.1, 10 message-passing iterations, one policy worker,
  and a 50-step episode limit.

## Main interpretation

The randomized benchmark retains meaningful resolution and temporal-depth
effects. Depth is most valuable at coarse and intermediate resolution, whereas
maximum depth does not improve aggregate success at `gamma=10` or `gamma=20`.
Resolution has a large effect at shallow depth. The cheapest successful
allocation also varies across instances, so the benchmark provides a genuine
allocation problem instead of merely rewarding the largest fixed model.

The post-hoc least-compute successful oracle reaches 30/30 successes with mean
cumulative inference time 774.5 ms, mean task cost 29.7, and mean 27.5 steps.
Its selected allocations were:

| Allocation | Instances |
| --- | ---: |
| `(gamma=5, T=1)` | 13 |
| `(gamma=5, T=2)` | 7 |
| `(gamma=10, T=1)` | 3 |
| `(gamma=10, T=2)` | 2 |
| `(gamma=2, T=1)` | 2 |
| `(gamma=20, T=1)` | 2 |
| `(gamma=2, T=2)` | 1 |

This oracle uses completed outcomes and therefore serves only as an upper bound
for evaluating a future online allocator.
