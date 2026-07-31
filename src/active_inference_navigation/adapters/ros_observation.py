"""ROS 2 position/RSSI observation adapter.

The aggregation logic is importable and testable without ROS. The factory at
the bottom is the only place that imports ROS message classes.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import median
from threading import Lock
from time import monotonic
from typing import Any

from ..interfaces import ObservationUnavailableError, StaleObservationError
from ..models import Observation


def identity_position(x: float, y: float) -> tuple[float, float]:
    """Return an unchanged position for already-aligned sensor coordinates."""

    return x, y


@dataclass
class RosObservationSource:
    """Combine latest odometry with a median window of RSSI samples."""

    rssi_median_window: int = 5
    odom_timeout: float = 1.0
    rssi_timeout: float = 1.0
    clock: Callable[[], float] = monotonic
    position_transform: Callable[[float, float], tuple[float, float]] = identity_position
    _position: tuple[float, float] | None = field(default=None, init=False, repr=False)
    _odom_time: float | None = field(default=None, init=False, repr=False)
    _rssi_samples: deque[tuple[float, float]] = field(init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rssi_median_window < 1:
            raise ValueError("rssi_median_window must be positive.")
        if self.odom_timeout <= 0.0 or self.rssi_timeout <= 0.0:
            raise ValueError("Sensor timeouts must be positive.")
        self._rssi_samples = deque(maxlen=self.rssi_median_window)

    def odometry_callback(self, message: Any, *, received_at: float | None = None) -> None:
        """Store x and y from a ``nav_msgs/msg/Odometry``-compatible message."""

        timestamp = self.clock() if received_at is None else float(received_at)
        try:
            position = message.pose.pose.position
            value = float(position.x), float(position.y)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Odometry message does not contain a valid x/y position.") from error
        with self._lock:
            self._position = value
            self._odom_time = timestamp

    def rssi_callback(self, message: Any, *, received_at: float | None = None) -> None:
        """Store a ``std_msgs/msg/Float32``-compatible RSSI sample."""

        timestamp = self.clock() if received_at is None else float(received_at)
        try:
            value = float(message.data)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("RSSI message does not contain a valid Float32 value.") from error
        with self._lock:
            self._rssi_samples.append((timestamp, value))

    def read_observation(self) -> Observation:
        """Return a current position and median RSSI observation."""

        now = self.clock()
        with self._lock:
            if self._position is None or self._odom_time is None:
                raise ObservationUnavailableError("No odometry measurement has been received.")
            if now - self._odom_time > self.odom_timeout:
                raise StaleObservationError("The latest odometry measurement is stale.")
            if not self._rssi_samples:
                raise ObservationUnavailableError("No RSSI measurement has been received.")
            latest_rssi_time = self._rssi_samples[-1][0]
            if now - latest_rssi_time > self.rssi_timeout:
                raise StaleObservationError("The latest RSSI measurement is stale.")
            window_rssi = [value for _, value in self._rssi_samples]
            x, y = self.position_transform(*self._position)
            measurement_time = max(self._odom_time, latest_rssi_time)
        return Observation(x=x, y=y, rssi=float(median(window_rssi)), timestamp=measurement_time)


def attach_ros_observation_subscriptions(
    node: Any,
    source: RosObservationSource,
    *,
    odom_topic: str,
    rssi_topic: str,
    qos_depth: int = 10,
) -> tuple[Any, Any]:
    """Attach ROS subscriptions while keeping ROS imports outside the core."""

    try:
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Float32
    except ImportError as error:
        raise RuntimeError("ROS 2 nav_msgs and std_msgs are required for ROS observation I/O.") from error

    odom_subscription = node.create_subscription(
        Odometry,
        odom_topic,
        source.odometry_callback,
        qos_depth,
    )
    rssi_subscription = node.create_subscription(
        Float32,
        rssi_topic,
        source.rssi_callback,
        qos_depth,
    )
    return odom_subscription, rssi_subscription
