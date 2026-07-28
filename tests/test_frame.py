import pytest

from active_inference_navigation.frame import ArenaFrameTransform


def test_confirmed_robot_frame_maps_raw_odometry_to_arena_axes():
    transform = ArenaFrameTransform(
        arena_x_from_odom="-y",
        arena_y_from_odom="+x",
        odom_zero_arena_x=0.175,
        odom_zero_arena_y=0.175,
    )

    assert transform.position_to_arena(0.0, 0.0) == pytest.approx((0.175, 0.175))
    assert transform.position_to_arena(0.0, -0.35) == pytest.approx((0.525, 0.175))
    assert transform.position_to_arena(0.35, 0.0) == pytest.approx((0.175, 0.525))


def test_confirmed_robot_frame_inverse_maps_arena_targets_to_odometry():
    transform = ArenaFrameTransform(
        arena_x_from_odom="-y",
        arena_y_from_odom="+x",
        odom_zero_arena_x=0.175,
        odom_zero_arena_y=0.175,
    )

    assert transform.position_to_odom(0.525, 0.175) == pytest.approx((0.0, -0.35))
    assert transform.position_to_odom(0.175, 0.525) == pytest.approx((0.35, 0.0))


def test_frame_rejects_reusing_same_odometry_axis():
    with pytest.raises(ValueError, match="different"):
        ArenaFrameTransform(arena_x_from_odom="+x", arena_y_from_odom="-x")
