"""Run a paired randomized robustness diagnostic on the 2D MOS benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from diagnose_mos_fixed import run_episode, summarize, write_csv

from active_inference_navigation.mos import TARGET_RESOLUTIONS, sample_mos_instance


def paired_effects(episodes: list[dict]) -> dict:
    """Compare allocations within instances so task randomness cancels out."""

    by_instance = {
        instance_seed: {
            (row["resolution"], row["action_depth"]): row
            for row in episodes
            if row["instance_seed"] == instance_seed
        }
        for instance_seed in sorted({row["instance_seed"] for row in episodes})
    }

    def compare(left, right):
        pairs = [
            (rows[left], rows[right])
            for rows in by_instance.values()
            if left in rows and right in rows
        ]
        return {
            "instances": len(pairs),
            "right_success_only": sum(not a["success"] and b["success"] for a, b in pairs),
            "left_success_only": sum(a["success"] and not b["success"] for a, b in pairs),
            "both_succeeded": sum(a["success"] and b["success"] for a, b in pairs),
            "both_failed": sum(not a["success"] and not b["success"] for a, b in pairs),
            "mean_task_cost_left_minus_right": (
                statistics.fmean(a["task_cost"] - b["task_cost"] for a, b in pairs)
                if pairs
                else None
            ),
            "mean_inference_ms_left_minus_right": (
                statistics.fmean(
                    a["cumulative_inference_ms"] - b["cumulative_inference_ms"] for a, b in pairs
                )
                if pairs
                else None
            ),
        }

    depth = []
    for resolution in TARGET_RESOLUTIONS:
        for shallow in (1, 2):
            depth.append(
                {
                    "resolution": resolution,
                    "left_depth": shallow,
                    "right_depth": 3,
                    **compare((resolution, shallow), (resolution, 3)),
                }
            )

    resolution = []
    for depth_value in (1, 2, 3):
        for coarse in (2, 5, 10):
            resolution.append(
                {
                    "action_depth": depth_value,
                    "left_resolution": coarse,
                    "right_resolution": 20,
                    **compare((coarse, depth_value), (20, depth_value)),
                }
            )

    regimes = []
    regime_counts = Counter()
    for instance_seed, rows in by_instance.items():
        successful = sorted(configuration for configuration, row in rows.items() if row["success"])
        resolution_required = bool(successful) and not any(gamma == 2 for gamma, _ in successful)
        depth_required = bool(successful) and not any(
            depth_value == 1 for _, depth_value in successful
        )
        if not successful:
            regime = "no_configuration_succeeded"
        elif (2, 1) in successful:
            regime = "coarse_shallow_sufficient"
        elif resolution_required and depth_required:
            regime = "joint_resolution_and_depth_required"
        elif resolution_required:
            regime = "resolution_required"
        elif depth_required:
            regime = "depth_required"
        else:
            regime = "mixed_tradeoff"
        regime_counts[regime] += 1
        regimes.append(
            {
                "instance_seed": instance_seed,
                "regime": regime,
                "successful_configurations": [
                    {"resolution": gamma, "action_depth": depth_value}
                    for gamma, depth_value in successful
                ],
            }
        )

    return {
        "paired_depth_effects": depth,
        "paired_resolution_effects": resolution,
        "allocation_regime_counts": dict(sorted(regime_counts.items())),
        "allocation_regimes": regimes,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--instance-seeds", nargs="+", type=int)
    seeds.add_argument("--num-instances", type=int, default=30)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--resolutions", nargs="+", type=int, choices=TARGET_RESOLUTIONS)
    parser.add_argument("--depths", nargs="+", type=int, choices=(1, 2, 3))
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/mos_robustness"),
    )
    args = parser.parse_args()
    if args.num_instances is not None and args.num_instances < 1:
        parser.error("--num-instances must be positive")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instance_seeds = (
        args.instance_seeds
        if args.instance_seeds is not None
        else list(range(args.seed_offset, args.seed_offset + args.num_instances))
    )
    resolutions = args.resolutions or list(TARGET_RESOLUTIONS)
    depths = args.depths or [1, 2, 3]
    instances = [sample_mos_instance(seed) for seed in instance_seeds]
    instance_rows = [
        {
            "instance_seed": instance.instance_seed,
            "map_seed": instance.layout.map_seed,
            "start_seed": instance.start_seed,
            "target_seed": instance.target_seed,
            "observation_seed": instance.observation_seed,
            "start_x": instance.layout.start[0],
            "start_y": instance.layout.start[1],
            "target_x": instance.target[0],
            "target_y": instance.target[1],
            "sensor_range": instance.layout.sensor_range,
            "blocked_cells": len(instance.layout.blocked),
        }
        for instance in instances
    ]
    episodes = []
    decisions = []
    total = len(instances) * len(resolutions) * len(depths)
    completed = 0
    for instance in instances:
        for resolution in resolutions:
            for depth in depths:
                episode = run_episode(
                    map_seed=instance.layout.map_seed,
                    target_seed=instance.target_seed,
                    observation_seed=instance.observation_seed,
                    resolution=resolution,
                    depth=depth,
                    max_steps=args.max_steps,
                    decision_rows=decisions,
                    layout=instance.layout,
                    instance_seed=instance.instance_seed,
                    start_seed=instance.start_seed,
                    message_passing_iterations=args.message_passing_iterations,
                    policy_workers=args.policy_workers,
                )
                episodes.append(episode)
                completed += 1
                print(
                    f"[{completed}/{total}] instance={instance.instance_seed} "
                    f"gamma={resolution} T={depth} success={episode['success']} "
                    f"steps={episode['steps']}",
                    flush=True,
                )

    write_csv(args.output_dir / "instances.csv", instance_rows)
    write_csv(args.output_dir / "episodes.csv", episodes)
    write_csv(args.output_dir / "decisions.csv", decisions)
    summary = {
        "protocol": {
            "domain": "paired randomized single-object 2D MOS",
            "instance_seeds": instance_seeds,
            "resolutions": resolutions,
            "depths": depths,
            "max_steps": args.max_steps,
            "message_passing_iterations": args.message_passing_iterations,
            "policy_workers": args.policy_workers,
            "randomized_properties": [
                "obstacle geometry",
                "robot start",
                "target location",
                "sensor range",
                "observation realization",
            ],
            "paired_across_allocations": True,
            "adaptive_controller": False,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyaif_toolkit": importlib.metadata.version("pyaif-toolkit"),
            "active_inference_navigation_agent": importlib.metadata.version(
                "active-inference-navigation-agent"
            ),
        },
        **summarize(episodes, decisions),
        **paired_effects(episodes),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
