import numpy as np

from ask_professor_g.gripper_axes import local_gripper_axes, rotation_from_normal_and_closing


def _assert_axis_mapping(gripper, normal, closing):
    rotation, debug = rotation_from_normal_and_closing(normal, closing, gripper)
    axes = local_gripper_axes(gripper)
    local_closing = np.asarray(axes["local_closing_axis"], dtype=float)
    local_tcp = np.asarray(axes["local_tcp_axis"], dtype=float)
    target_closing = np.asarray(debug["target_closing_axis"], dtype=float)
    target_tcp = np.asarray(debug["target_tcp_axis"], dtype=float)

    assert np.dot(rotation @ local_closing, target_closing) > 0.995
    assert np.dot(rotation @ local_tcp, target_tcp) > 0.995
    assert debug["closing_alignment_dot"] > 0.995
    assert debug["tcp_axis_alignment_dot"] > 0.995


def test_wsg50_maps_local_x_closing_to_world_narrow_axis():
    gripper = {
        "closing_direction": "+X",
        "palm_normal": "+Z",
        "tcp_offset": [0.0, 0.0, 0.1335],
    }
    _assert_axis_mapping(gripper, normal=[0, 0, 1], closing=[0, 1, 0])


def test_franka_maps_local_y_closing_to_world_narrow_axis():
    gripper = {
        "closing_direction": "+Y",
        "palm_normal": "+Z",
        "tcp_offset": [0.0, 0.0, 0.095],
    }
    _assert_axis_mapping(gripper, normal=[1, 0, 0], closing=[0, 0, 1])


def test_sawyer_uses_tcp_offset_direction_even_when_offset_has_small_x_component():
    gripper = {
        "closing_direction": "+Y",
        "palm_normal": "+Z",
        "tcp_offset": [0.003, 0.0, 0.0455],
    }
    _assert_axis_mapping(gripper, normal=[0, 1, 0], closing=[1, 0, 0])
