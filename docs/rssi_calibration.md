# RSSI calibration

The `calibrated_dbm` likelihood was fitted from stationary measurements at
1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, and 8.6 metres.

The runtime's default observation is the median of five recent RSSI samples.
Bootstrap distributions of five-sample medians were therefore used instead of
fitting individual packets. The resulting initial log-distance model is:

```text
expected_rssi(distance_m) =
    -63.02 - 10 * 1.635 * log10(distance_m)
```

The pooled residual standard deviation of the aggregated measurements is
approximately 3.37 dB. These values are exposed in `config/navigation.yaml`;
they are not embedded in the ROS sensor adapter.

Distances below 1.0 m are clamped to 1.0 m because no closer measurements were
provided. Distances beyond 8.6 m are extrapolations. The 5 m and 7 m datasets
had substantially larger spread than the other distances, so this initial
single-Gaussian model should be reviewed after collecting measurements across
robot orientations and additional arena locations.

The fitted model parameters are:

| Parameter | Value |
|---|---:|
| Reference RSSI at 1 m | -63.02 dBm |
| Path-loss exponent | 1.635 |
| Aggregated signal sigma | 3.37 dB |
| Minimum calibrated distance | 1.0 m |
| Observation grid minimum | -95 dBm |
| Observation grid maximum | -55 dBm |

Changing radios, antennas, mounting, transmit power, arena materials, or RSSI
aggregation requires recalibration. Sensor adapters should continue returning
the measured dBm value unchanged.
