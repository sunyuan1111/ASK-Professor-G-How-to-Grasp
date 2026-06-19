from pathlib import Path

from ask_professor_g.config import load_config
from ask_professor_g.data_registry import DataRegistry


ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_ignores_oldcode_and_env_example_is_kept():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "oldcode/" in gitignore
    assert ".env" in gitignore
    assert "!.env.example" in gitignore


def test_default_config_and_registry_paths_exist():
    config = load_config(ROOT / "configs" / "default.yaml")
    registry = DataRegistry(config)
    gripper = registry.gripper()
    obj = registry.object()
    assert gripper.key == "wsg_50"
    assert obj.key == "3D_Dollhouse_Lamp"
    assert registry.validate_paths(gripper, obj) == []

