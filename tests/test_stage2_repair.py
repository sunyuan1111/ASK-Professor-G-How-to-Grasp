import json

import numpy as np

from ask_professor_g.cli import _run_stage2_with_repair
from ask_professor_g.llm.base import LLMClient
from ask_professor_g.optimization.loss_loader import load_loss_function, validate_loss_function


class FakeStage2Client(LLMClient):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_text(self, system_prompt, user_prompt, *, image_path=None):
        self.calls += 1
        if not self.responses:
            raise RuntimeError("No fake responses left")
        return self.responses.pop(0)


def write_stage1_fixture(path):
    payload = {
        "grasps": [
            {
                "type": "stem grasp",
                "category": "primary",
                "source_3d_point": [0.01, 0.02, 0.03],
                "target_part": "stem",
                "source_stage0_strategy": "pinch stem",
                "wrist_pose_relative": {
                    "pos_xyz_m": {"x": [0.0, 0.02], "y": [0.01, 0.03], "z": [0.02, 0.04]},
                    "orn_rpy_deg": {"r": [80, 100], "p": [-5, 5], "y": [-10, 10]},
                },
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_point_cloud(path):
    np.save(path, np.array([[0.0, 0.0, 0.0], [0.01, 0.02, 0.03], [0.02, 0.0, 0.0]], dtype=float))


def test_validate_loss_function_catches_runtime_name_error(tmp_path):
    bad_loss = tmp_path / "bad_runtime_loss.py"
    bad_loss.write_text(
        "import numpy as np\n\n"
        "def calculate_loss(pose_mat: np.ndarray, point_cloud: np.ndarray) -> float:\n"
        "    return float(_MISSING_CONSTANT)\n",
        encoding="utf-8",
    )
    try:
        validate_loss_function(bad_loss, point_cloud=np.zeros((4, 3)))
    except NameError as exc:
        assert "_MISSING_CONSTANT" in str(exc)
    else:
        raise AssertionError("Expected NameError")


def test_stage2_repair_loop_accepts_second_valid_response(tmp_path):
    stage1_path = tmp_path / "stage1_output_processed.json"
    point_cloud_path = tmp_path / "object.npy"
    prompts_dir = tmp_path / "prompts"
    step2_path = tmp_path / "step2_loss.py"
    write_stage1_fixture(stage1_path)
    write_point_cloud(point_cloud_path)
    bad_code = "import numpy as np\n\ndef calculate_loss(pose_mat, point_cloud):\n    return float(_STAGE1_TARGETS[0])\n"
    good_code = "import numpy as np\n\ndef calculate_loss(pose_mat, point_cloud):\n    return float(np.linalg.norm(pose_mat[:3, 3]))\n"
    client = FakeStage2Client([bad_code, good_code])

    metadata = _run_stage2_with_repair(
        client=client,
        gripper={"name": "test"},
        stage1_processed_path=stage1_path,
        point_cloud_path=point_cloud_path,
        run_dir=tmp_path,
        prompts_dir=prompts_dir,
        step2_path=step2_path,
        max_attempts=2,
    )

    assert client.calls == 2
    assert metadata["source"] == "llm"
    assert metadata["status"] == "loaded_and_smoke_tested"
    assert (tmp_path / "step2_loss.invalid.py").exists()
    assert (tmp_path / "step2_loss.repair1.raw.txt").exists()
    assert "_STAGE1_TARGETS" not in step2_path.read_text(encoding="utf-8")


def test_stage2_repair_loop_falls_back_after_failed_attempts(tmp_path):
    stage1_path = tmp_path / "stage1_output_processed.json"
    point_cloud_path = tmp_path / "object.npy"
    prompts_dir = tmp_path / "prompts"
    step2_path = tmp_path / "step2_loss.py"
    write_stage1_fixture(stage1_path)
    write_point_cloud(point_cloud_path)
    bad_code = "import numpy as np\n\ndef calculate_loss(pose_mat, point_cloud):\n    return float(_STAGE1_TARGETS[0])\n"
    client = FakeStage2Client([bad_code, bad_code])

    metadata = _run_stage2_with_repair(
        client=client,
        gripper={"name": "test"},
        stage1_processed_path=stage1_path,
        point_cloud_path=point_cloud_path,
        run_dir=tmp_path,
        prompts_dir=prompts_dir,
        step2_path=step2_path,
        max_attempts=2,
    )

    assert client.calls == 2
    assert metadata["source"] == "fallback_from_stage1"
    assert metadata["status"] == "llm_stage2_failed_after_repair_attempts"
    assert (tmp_path / "step2_loss.invalid.py").exists()
    assert (tmp_path / "step2_loss.repair1.invalid.py").exists()
    loss_func = load_loss_function(step2_path)
    assert np.isfinite(loss_func(np.eye(4), np.zeros((4, 3))))