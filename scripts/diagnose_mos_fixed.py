"""Run fixed-resolution, fixed-depth diagnostics on the 2D MOS benchmark."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from active_inference_navigation.mos import (
    TARGET_RESOLUTIONS,
    FindOutcome,
    MOSAgentConfig,
    MOSEnvironment,
    MOSLayout,
    build_mos_agent,
    mos_action_controls,
    selected_mos_action,
)

CONFIGURATIONS = tuple(
    (resolution, depth) for resolution in TARGET_RESOLUTIONS for depth in (1, 2, 3)
)


def timed(function):
    started = time.perf_counter_ns()
    value = function()
    return value, (time.perf_counter_ns() - started) / 1e6


def entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0]
    return float(-values.dot(np.log(values)))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_episode(
    *,
    map_seed,
    target_seed,
    observation_seed,
    resolution,
    depth,
    max_steps,
    decision_rows,
):
    layout = MOSLayout(map_seed=map_seed)
    target = layout.sample_target(target_seed)
    environment = MOSEnvironment(
        layout=layout,
        target=target,
        observation_seed=observation_seed,
    )
    agent = build_mos_agent(
        MOSAgentConfig(target_resolution=resolution, action_depth=depth),
        layout=layout,
    )
    agent.reset()
    num_policies = len(agent.policies)
    quantile_seed = map_seed * 1_000_003 + target_seed * 1009 + observation_seed
    quantiles = np.random.default_rng(quantile_seed).random(max_steps + 1)
    observation = environment.reset(noise_quantile=float(quantiles[0]))
    episode_id = f"map{map_seed}-target{target_seed}-obs{observation_seed}-g{resolution}-T{depth}"
    previous_action = None
    false_finds = 0
    collisions = 0
    task_cost = 0.0
    cumulative_inference_ms = 0.0
    success = False
    steps = 0

    for decision in range(max_steps):
        current_observation = observation
        controls = None if previous_action is None else mos_action_controls(previous_action)
        agent.observe(current_observation, time_step=decision, executed_action=controls)
        _, state_ms = timed(agent.infer_states)
        _, policy_ms = timed(agent.infer_policies)
        action, action_ms = timed(lambda: selected_mos_action(agent))
        total_ms = state_ms + policy_ms + action_ms
        cumulative_inference_ms += total_ms
        posterior = np.asarray(agent.posterior_pi, dtype=float)
        selected_policy = int(np.argmax(posterior))
        target_belief = np.asarray(agent.filtered_posteriors[2], dtype=float)
        position = environment.position
        next_observation, success = environment.step(
            action,
            noise_quantile=float(quantiles[decision + 1]),
        )
        false_find = next_observation[3] == FindOutcome.FALSE_FIND
        # The observation reports occupancy of the resulting spatial state so
        # it remains consistent with A(o_c | x, y).  A blocked movement is a
        # separate environment/task diagnostic because the robot stays in its
        # previous free state.
        collision = environment.last_collision
        false_finds += int(false_find)
        collisions += int(collision)
        task_cost += 1.0 + 5.0 * false_find + 0.5 * collision
        decision_rows.append(
            {
                "episode_id": episode_id,
                "map_seed": map_seed,
                "target_seed": target_seed,
                "observation_seed": observation_seed,
                "resolution": resolution,
                "action_depth": depth,
                "decision": decision,
                "x": position[0],
                "y": position[1],
                "target_x": target[0],
                "target_y": target[1],
                "observation": json.dumps([int(value) for value in current_observation]),
                "next_observation": json.dumps([int(value) for value in next_observation]),
                "next_x": environment.position[0],
                "next_y": environment.position[1],
                "selected_action": action.name,
                "state_ms": state_ms,
                "policy_ms": policy_ms,
                "action_ms": action_ms,
                "total_ms": total_ms,
                "cumulative_inference_ms": cumulative_inference_ms,
                "state_vfe": float(agent.last_state_inference.variational_free_energy),
                "target_belief_entropy": entropy(target_belief),
                "selected_policy": selected_policy,
                "selected_policy_posterior": float(posterior[selected_policy]),
                "selected_policy_efe": float(agent.G_policy[selected_policy]),
                "false_find": false_find,
                "collision": collision,
                "success": success,
            }
        )
        steps = decision + 1
        previous_action = action
        observation = next_observation
        if success:
            break

    return {
        "episode_id": episode_id,
        "map_seed": map_seed,
        "target_seed": target_seed,
        "observation_seed": observation_seed,
        "resolution": resolution,
        "action_depth": depth,
        "num_policies": num_policies,
        "target_x": target[0],
        "target_y": target[1],
        "success": success,
        "steps": steps,
        "task_cost": task_cost + (0.0 if success else 2.0 * max_steps),
        "false_finds": false_finds,
        "collisions": collisions,
        "cumulative_inference_ms": cumulative_inference_ms,
        "final_x": environment.position[0],
        "final_y": environment.position[1],
        "final_target_distance": float(
            np.hypot(environment.position[0] - target[0], environment.position[1] - target[1])
        ),
    }


def summarize(episodes: list[dict], decisions: list[dict]) -> dict:
    grouped_episodes = defaultdict(list)
    grouped_decisions = defaultdict(list)
    for row in episodes:
        grouped_episodes[(row["resolution"], row["action_depth"])].append(row)
    for row in decisions:
        grouped_decisions[(row["resolution"], row["action_depth"])].append(row)
    configurations = []
    for configuration in sorted(grouped_episodes):
        episode_values = grouped_episodes[configuration]
        decision_values = grouped_decisions[configuration]
        configurations.append(
            {
                "resolution": configuration[0],
                "action_depth": configuration[1],
                "num_policies": episode_values[0]["num_policies"],
                "episodes": len(episode_values),
                "success_rate": statistics.fmean(row["success"] for row in episode_values),
                "mean_steps": statistics.fmean(row["steps"] for row in episode_values),
                "mean_task_cost": statistics.fmean(row["task_cost"] for row in episode_values),
                "mean_decision_ms": statistics.fmean(row["total_ms"] for row in decision_values),
                "mean_state_ms": statistics.fmean(row["state_ms"] for row in decision_values),
                "mean_policy_ms": statistics.fmean(row["policy_ms"] for row in decision_values),
                "mean_cumulative_inference_ms": statistics.fmean(
                    row["cumulative_inference_ms"] for row in episode_values
                ),
            }
        )

    initial_actions = defaultdict(lambda: defaultdict(dict))
    for row in decisions:
        if row["decision"] == 0:
            instance = (row["map_seed"], row["target_seed"], row["observation_seed"])
            initial_actions[instance][row["resolution"]][row["action_depth"]] = row[
                "selected_action"
            ]
    disagreements = []
    for resolution in sorted({row["resolution"] for row in episodes}):
        matched = [
            actions[resolution]
            for actions in initial_actions.values()
            if resolution in actions and all(depth in actions[resolution] for depth in (1, 2, 3))
        ]
        t1_t3 = sum(actions[1] != actions[3] for actions in matched)
        t2_t3 = sum(actions[2] != actions[3] for actions in matched)
        disagreements.append(
            {
                "resolution": resolution,
                "instances": len(matched),
                "t1_vs_t3_count": t1_t3,
                "t1_vs_t3_rate": t1_t3 / len(matched) if matched else None,
                "t2_vs_t3_count": t2_t3,
                "t2_vs_t3_rate": t2_t3 / len(matched) if matched else None,
            }
        )

    action_sequences = defaultdict(lambda: defaultdict(list))
    for row in decisions:
        instance = (
            row["map_seed"],
            row["target_seed"],
            row["observation_seed"],
            row["resolution"],
        )
        action_sequences[instance][row["action_depth"]].append(row["selected_action"])
    prefix_disagreements = []
    for resolution in sorted({row["resolution"] for row in episodes}):
        matched = [
            sequences
            for instance, sequences in action_sequences.items()
            if instance[3] == resolution and all(depth in sequences for depth in (1, 2, 3))
        ]
        row = {"resolution": resolution, "instances": len(matched)}
        for left_depth, right_depth, label in ((1, 3, "t1_vs_t3"), (2, 3, "t2_vs_t3")):
            first_disagreements = []
            for sequences in matched:
                first = next(
                    (
                        decision
                        for decision, (left, right) in enumerate(
                            zip(sequences[left_depth], sequences[right_depth], strict=False)
                        )
                        if left != right
                    ),
                    None,
                )
                if first is not None:
                    first_disagreements.append(first)
            row[f"{label}_count"] = len(first_disagreements)
            row[f"{label}_rate"] = len(first_disagreements) / len(matched) if matched else None
            row[f"{label}_mean_first_decision"] = (
                statistics.fmean(first_disagreements) if first_disagreements else None
            )
        prefix_disagreements.append(row)
    return {
        "configurations": configurations,
        "initial_action_instances": len(initial_actions),
        "initial_action_disagreement_by_resolution": disagreements,
        "trajectory_prefix_disagreement_by_resolution": prefix_disagreements,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--target-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--observation-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--resolutions", nargs="+", type=int, choices=TARGET_RESOLUTIONS)
    parser.add_argument("--depths", nargs="+", type=int, choices=(1, 2, 3))
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/mos_fixed"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolutions = args.resolutions or list(TARGET_RESOLUTIONS)
    depths = args.depths or [1, 2, 3]
    episodes = []
    decisions = []
    total = (
        len(args.map_seeds)
        * len(args.target_seeds)
        * len(args.observation_seeds)
        * len(resolutions)
        * len(depths)
    )
    completed = 0
    for map_seed in args.map_seeds:
        for target_seed in args.target_seeds:
            for observation_seed in args.observation_seeds:
                for resolution in resolutions:
                    for depth in depths:
                        episode = run_episode(
                            map_seed=map_seed,
                            target_seed=target_seed,
                            observation_seed=observation_seed,
                            resolution=resolution,
                            depth=depth,
                            max_steps=args.max_steps,
                            decision_rows=decisions,
                        )
                        episodes.append(episode)
                        completed += 1
                        print(
                            f"[{completed}/{total}] map={map_seed} target={target_seed} "
                            f"obs={observation_seed} gamma={resolution} T={depth} "
                            f"success={episode['success']} steps={episode['steps']}",
                            flush=True,
                        )

    write_csv(args.output_dir / "episodes.csv", episodes)
    write_csv(args.output_dir / "decisions.csv", decisions)
    summary = {
        "protocol": {
            "domain": "single-object 2D MOS",
            "map_seeds": args.map_seeds,
            "target_seeds": args.target_seeds,
            "observation_seeds": args.observation_seeds,
            "resolutions": resolutions,
            "depths": depths,
            "max_steps": args.max_steps,
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
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
