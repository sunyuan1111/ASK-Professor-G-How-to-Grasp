import json

import numpy as np

from ask_professor_g.geometry.probing import normalized_to_pixel, simple_geometry_probe


def test_normalized_to_pixel_top_left():
    assert normalized_to_pixel([0.0, 0.0], 800, 600) == (0, 0)
    assert normalized_to_pixel([1.0, 1.0], 800, 600) == (599, 799)
    assert normalized_to_pixel([1.5, -0.5], 800, 600) == (0, 799)


def test_normalized_to_pixel_bottom_left():
    assert normalized_to_pixel([0.0, 0.0], 800, 600, origin="bottom_left") == (599, 0)


def test_pyrender_depth_backprojection(tmp_path):
    camera = {
        "projection_type": "pyrender_perspective",
        "image_size": [5, 5],
        "image_width": 5,
        "image_height": 5,
        "intrinsic": {"fx": 2.0, "fy": 2.0, "cx": 2.0, "cy": 2.0},
        "camera_pose": np.eye(4).tolist(),
        "extrinsic": np.eye(4).tolist(),
    }
    camera_path = tmp_path / "camera.json"
    camera_path.write_text(json.dumps(camera), encoding="utf-8")
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 2.0
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, depth)
    points = np.array(
        [
            [-0.02, -0.02, -2.0],
            [0.02, -0.02, -2.0],
            [0.02, 0.02, -2.0],
            [-0.02, 0.02, -2.0],
            [0.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )
    cloud_path = tmp_path / "points.xyz"
    np.savetxt(cloud_path, points)
    stage0 = {
        "object_analysis": "synthetic centered plane",
        "proposals": [
            {
                "id": 1,
                "strategy": "CenterPoint",
                "priority": "High",
                "candidate_points": [[0.5, 0.5]],
                "closing_direction": [1, 0, 0],
            }
        ],
    }

    result = simple_geometry_probe(
        stage0,
        cloud_path,
        output_path=tmp_path / "geometry.json",
        camera_path=camera_path,
        depth_path=depth_path,
        gripper_limits={"max_width": 1.0, "min_width": 0.0},
    )

    selected = result["audit_results"][0]
    assert selected["projection_source"] == "pyrender_depth"
    assert selected["audit_status"] == "VALID"
    np.testing.assert_allclose(selected["adjusted_3d_point"], [0.0, 0.0, -2.0], atol=1e-6)
