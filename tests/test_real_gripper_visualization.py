from pathlib import Path

import numpy as np

from ask_professor_g.config import load_config
from ask_professor_g.data_registry import DataRegistry
from ask_professor_g.visualization.real_gripper import build_real_gripper_meshes, load_urdf_visual_model


ROOT = Path(__file__).resolve().parents[1]


def test_wsg50_real_gripper_meshes_load_from_urdf():
    config = load_config(ROOT / "configs" / "default.yaml")
    gripper = DataRegistry(config).gripper()
    state = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0], dtype=float)

    result = build_real_gripper_meshes(
        state=state,
        gripper=gripper.raw,
        urdf_path=gripper.path("urdf"),
        glpca_path=gripper.path("glpca"),
    )

    assert result.meshes
    assert result.source == "glpca"
    assert result.joint_config["base_joint_gripper_left"] < 0
    assert result.joint_config["base_joint_gripper_right"] > 0


def test_urdf_visual_model_discovers_wsg50_links_and_visuals():
    config = load_config(ROOT / "configs" / "default.yaml")
    gripper = DataRegistry(config).gripper()
    model = load_urdf_visual_model(gripper.path("urdf"))

    assert model.root_link == "base_link"
    assert {"base_link", "gripper_left", "gripper_right", "finger_left", "finger_right"}.issubset(model.links)
    assert len(model.visuals) >= 5
