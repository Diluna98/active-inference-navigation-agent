"""Fit distance-only and bearing-aware RSSI models from calibration CSV files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import atan2, cos, hypot, log10, sin, sqrt
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AggregatedObservation:
    """One median-aggregated observation from a stationary collection batch."""

    batch: tuple[int, int]
    distance: float
    bearing: float
    rssi: float


def _stationary_batches(
    paths: list[Path],
    *,
    position_tolerance: float,
) -> dict[tuple[int, int], list[tuple[float, float, float, float, float]]]:
    """Read CSV rows and split them when the robot moves to another position."""

    batches: dict[tuple[int, int], list[tuple[float, float, float, float, float]]] = {}
    for file_index, path in enumerate(paths):
        batch_index = -1
        anchor: tuple[float, float] | None = None
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {
                "rssi_dbm",
                "arena_x_m",
                "arena_y_m",
                "source_x_m",
                "source_y_m",
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for row in reader:
                x = float(row["arena_x_m"])
                y = float(row["arena_y_m"])
                if anchor is None or hypot(x - anchor[0], y - anchor[1]) > position_tolerance:
                    batch_index += 1
                    anchor = (x, y)
                batches.setdefault((file_index, batch_index), []).append(
                    (
                        x,
                        y,
                        float(row["source_x_m"]),
                        float(row["source_y_m"]),
                        float(row["rssi_dbm"]),
                    )
                )
    return batches


def aggregate_median_windows(
    batches: dict[tuple[int, int], list[tuple[float, float, float, float, float]]],
    *,
    window: int,
) -> list[AggregatedObservation]:
    """Convert stationary raw packets into non-overlapping median windows."""

    observations: list[AggregatedObservation] = []
    for batch, rows in batches.items():
        for start in range(0, len(rows) - window + 1, window):
            block = np.asarray(rows[start : start + window], dtype=float)
            receiver_x, receiver_y, source_x, source_y, rssi = np.median(block, axis=0)
            delta_x = receiver_x - source_x
            delta_y = receiver_y - source_y
            observations.append(
                AggregatedObservation(
                    batch=batch,
                    distance=hypot(delta_x, delta_y),
                    bearing=atan2(delta_y, delta_x),
                    rssi=float(rssi),
                )
            )
    return observations


def _design(
    observations: list[AggregatedObservation],
    *,
    minimum_distance: float,
    include_bearing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_counts: dict[tuple[int, int], int] = {}
    for observation in observations:
        batch_counts[observation.batch] = batch_counts.get(observation.batch, 0) + 1
    rows = []
    for observation in observations:
        row = [1.0, log10(max(observation.distance, minimum_distance))]
        if include_bearing:
            row.extend((cos(observation.bearing), sin(observation.bearing)))
        rows.append(row)
    response = np.asarray([observation.rssi for observation in observations])
    weights = np.asarray(
        [1.0 / batch_counts[observation.batch] for observation in observations]
    )
    return np.asarray(rows), response, weights


def fit(
    observations: list[AggregatedObservation],
    *,
    minimum_distance: float,
    include_bearing: bool,
) -> tuple[np.ndarray, float]:
    """Return equal-batch-weighted least-squares coefficients and fit RMSE."""

    design, response, weights = _design(
        observations,
        minimum_distance=minimum_distance,
        include_bearing=include_bearing,
    )
    root_weights = np.sqrt(weights)
    coefficients = np.linalg.lstsq(
        design * root_weights[:, None],
        response * root_weights,
        rcond=None,
    )[0]
    residual = response - design @ coefficients
    rmse = sqrt(float(np.sum(weights * residual**2) / np.sum(weights)))
    return coefficients, rmse


def leave_one_batch_out_rmse(
    observations: list[AggregatedObservation],
    *,
    minimum_distance: float,
    include_bearing: bool,
) -> float:
    """Return prediction RMSE when each stationary batch is held out once."""

    errors: list[float] = []
    for held_out in sorted({observation.batch for observation in observations}):
        training = [observation for observation in observations if observation.batch != held_out]
        validation = [
            observation for observation in observations if observation.batch == held_out
        ]
        coefficients, _ = fit(
            training,
            minimum_distance=minimum_distance,
            include_bearing=include_bearing,
        )
        design, response, _ = _design(
            validation,
            minimum_distance=minimum_distance,
            include_bearing=include_bearing,
        )
        errors.extend((response - design @ coefficients).tolist())
    return sqrt(float(np.mean(np.square(errors))))


def main() -> None:
    """Fit both models and print configuration-ready parameters."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="Calibration CSV file(s).")
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--minimum-distance", type=float, default=0.35)
    parser.add_argument("--position-tolerance", type=float, default=0.1)
    args = parser.parse_args()
    if args.median_window < 1:
        parser.error("--median-window must be positive")
    if min(args.minimum_distance, args.position_tolerance) <= 0.0:
        parser.error("distance and position tolerances must be positive")

    batches = _stationary_batches(
        args.csv,
        position_tolerance=args.position_tolerance,
    )
    observations = aggregate_median_windows(batches, window=args.median_window)
    if len(batches) < 5 or len(observations) < 8:
        parser.error("too few stationary batches or aggregated observations")

    print(
        f"stationary_batches={len(batches)} "
        f"median_observations={len(observations)}"
    )
    for include_bearing, label in ((False, "distance"), (True, "bearing")):
        coefficients, training_rmse = fit(
            observations,
            minimum_distance=args.minimum_distance,
            include_bearing=include_bearing,
        )
        validation_rmse = leave_one_batch_out_rmse(
            observations,
            minimum_distance=args.minimum_distance,
            include_bearing=include_bearing,
        )
        print(
            f"{label}: coefficients={coefficients.tolist()} "
            f"training_rmse={training_rmse:.3f} "
            f"leave_one_batch_out_rmse={validation_rmse:.3f}"
        )
        if include_bearing:
            print(f"reference_rssi: {coefficients[0]:.3f}")
            print(f"path_loss_exponent: {-coefficients[1] / 10.0:.3f}")
            print(f"signal_sigma: {validation_rmse:.3f}")
            print(f"minimum_calibrated_distance: {args.minimum_distance:.3f}")
            print(f"bearing_cosine_coefficient: {coefficients[2]:.3f}")
            print(f"bearing_sine_coefficient: {coefficients[3]:.3f}")


if __name__ == "__main__":
    main()
