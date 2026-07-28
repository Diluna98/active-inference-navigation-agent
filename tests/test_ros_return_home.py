from active_inference_navigation.models import NavigationAction
from active_inference_navigation.ros_return_home import action_name, plan_return_actions


def test_return_path_moves_x_then_y_to_start():
    actions = plan_return_actions((3, 2), (0, 0))

    assert [action_name(action) for action in actions] == [
        "negative_x",
        "negative_x",
        "negative_x",
        "negative_y",
        "negative_y",
    ]


def test_return_path_supports_positive_directions():
    actions = plan_return_actions((1, 1), (3, 2))

    assert actions == (
        NavigationAction.from_sequence((2, 0)),
        NavigationAction.from_sequence((2, 0)),
        NavigationAction.from_sequence((0, 2)),
    )


def test_return_path_is_empty_when_already_home():
    assert plan_return_actions((0, 0), (0, 0)) == ()
