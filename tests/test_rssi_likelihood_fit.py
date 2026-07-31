import numpy as np

from scripts.fit_rssi_likelihood import AggregatedObservation, fit


def test_bearing_fitter_recovers_synthetic_coefficients():
    expected = np.asarray([-63.0, -30.0, 5.0, -9.0])
    observations = []
    for batch, (distance, bearing) in enumerate(
        (
            (0.35, -2.8),
            (0.5, -1.5),
            (0.7, -0.5),
            (1.0, 0.0),
            (1.4, 0.7),
            (2.0, 1.5),
            (3.0, 2.5),
            (5.0, 3.0),
        )
    ):
        design = np.asarray(
            [1.0, np.log10(max(distance, 0.35)), np.cos(bearing), np.sin(bearing)]
        )
        observations.append(
            AggregatedObservation(
                batch=(0, batch),
                distance=distance,
                bearing=bearing,
                rssi=float(design @ expected),
            )
        )

    coefficients, rmse = fit(
        observations,
        minimum_distance=0.35,
        include_bearing=True,
    )

    np.testing.assert_allclose(coefficients, expected, atol=1e-10)
    assert rmse < 1e-10
