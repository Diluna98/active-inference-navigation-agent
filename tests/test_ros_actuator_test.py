import pytest

from active_inference_navigation.models import NavigationAction
from active_inference_navigation.ros_actuator_test import build_parser, parse_action


def test_parse_positive_x_action():
    assert parse_action("positive_x") == NavigationAction.from_sequence((2, 0))


def test_parse_action_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown"):
        parse_action("diagonal")


def test_actuator_test_parser_requires_explicit_action():
    args = build_parser().parse_args(["--action", "positive_y"])

    assert args.action == "positive_y"
    assert args.config is None
