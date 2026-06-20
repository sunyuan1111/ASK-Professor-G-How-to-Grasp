import numpy as np

from ask_professor_g.optimization.cem import CEMOptimizer
from ask_professor_g.optimization.loss_loader import load_loss_function
from ask_professor_g.optimization.run_optimization import apply_fixed_synergy_center


def test_cem_optimizer_converges_on_position_loss():
    target = np.array([0.1, -0.2, 0.3])

    def loss(pose, point_cloud):
        return float(np.linalg.norm(pose[:3, 3] - target))

    optimizer = CEMOptimizer(num_samples=256, num_elites=32, max_iterations=8, seed=3)
    mean = np.zeros(9)
    std = np.ones(9) * 0.4
    state, best_loss, history = optimizer.optimize(mean, std, np.zeros((10, 3)), loss)
    assert best_loss < 0.15
    assert np.linalg.norm(state[:3] - target) < 0.15
    assert len(history) == 8


def test_loss_loader_requires_calculate_loss(tmp_path):
    bad_loss = tmp_path / "bad_loss.py"
    bad_loss.write_text("x = 1\n", encoding="utf-8")
    try:
        load_loss_function(bad_loss)
    except AttributeError as exc:
        assert "calculate_loss" in str(exc)
    else:
        raise AssertionError("Expected AttributeError")



def test_apply_fixed_synergy_center_uses_stage1_calibrated_bounds():
    state = np.array([0.1, -0.2, 0.3, 0.01, 0.02, 0.03, 0.12, 0.9, -0.7])
    grasp = {
        "synergy_config": {
            "s0": [0.46, 0.54],
            "s1": [0.1, 0.3],
            "s2": [-0.2, 0.2],
        }
    }

    fixed = apply_fixed_synergy_center(state, grasp)

    np.testing.assert_allclose(fixed[:6], state[:6])
    np.testing.assert_allclose(fixed[6:], [0.5, 0.2, 0.0])
    np.testing.assert_allclose(state[6:], [0.12, 0.9, -0.7])
