# RSSI calibration

The real-robot configuration selects `bearing_calibrated_dbm`. This provider
models the repeatable directional response of the TurtleBot 4 and nRF DK PCB
antennas in addition to log-distance path loss:

```text
theta = atan2(robot_y - source_y, robot_x - source_x)
expected_rssi =
    reference_rssi
    - 10 * path_loss_exponent * log10(max(distance_m, minimum_distance))
    + bearing_cosine_coefficient * cos(theta)
    + bearing_sine_coefficient * sin(theta)
```

`theta` is the arena-frame bearing from the source to the robot. The fit is
valid only when the transmitter orientation is unchanged and the robot is
restored to arena positive x before every RSSI observation. The actuator does
this through `motion.final_heading: positive_x`.

The fit used 720 stationary packets from 24 location batches in the two
directional calibration runs. Raw packets were converted to 144 non-overlapping
median-of-five observations to match the runtime. Each location batch received
equal total weight. Leave-one-batch-out RMSE improved from 9.62 dB for a
distance-only model to 7.02 dB for the bearing model. The configured 7.0 dB
sigma therefore reflects held-out prediction error rather than training error.

| Parameter | Value |
|---|---:|
| Reference RSSI at 1 m | -63.109 dBm |
| Path-loss exponent | 3.104 |
| Bearing cosine coefficient | 4.761 dB |
| Bearing sine coefficient | -9.065 dB |
| Aggregated signal sigma | 7.0 dB |
| Minimum calibrated distance | 0.35 m |
| Observation grid minimum | -95 dBm |
| Observation grid maximum | -25 dBm |

Reproduce the fit with:

```bash
python scripts/fit_rssi_likelihood.py \
  rssi_directional_test_part_1.csv \
  rssi_directional_test_part_2.csv
```

The older distance-only `calibrated_dbm` provider remains available. Changing
radios, antenna mounting/orientation, transmit power, arena materials, or RSSI
aggregation requires recalibration. Sensor adapters continue returning
measured dBm unchanged; likelihood assumptions remain outside ROS code.
