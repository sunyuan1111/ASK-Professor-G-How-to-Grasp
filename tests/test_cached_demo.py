from argparse import Namespace
import json

from ask_professor_g.cli import run_pipeline


def test_cached_demo_runs(tmp_path):
    run_dir = tmp_path / "cached_run"
    result_dir = run_pipeline(
        Namespace(
            config="examples/demo_config.yaml",
            gripper=None,
            object=None,
            run_dir=str(run_dir),
            stages=None,
            cached=True,
            example="examples/cached",
        )
    )
    assert result_dir == run_dir
    assert (run_dir / "stage0_output.json").exists()
    assert (run_dir / "geometry_probing_results.json").exists()
    assert (run_dir / "step3_output.json").exists()
    assert (run_dir / "final_grasp_report.json").exists()
    assert (run_dir / "visualizations" / "final_grasp_render.png").exists()
    assert (run_dir / "visualizations" / "final_grasp_real_views.png").exists()
    assert (run_dir / "visualizations" / "final_grasp_diagnostics.png").exists()
    assert (run_dir / "visualizations" / "obj_scene" / "final_grasp_real.obj").exists()

    report = json.loads((run_dir / "final_grasp_report.json").read_text(encoding="utf-8"))
    assert report["diagnostics"]["target_alignment"] == "OK"
    assert report["diagnostics"]["opening"] == "OK"
    assert report["diagnostics"]["surface_distance"] == "OK"
