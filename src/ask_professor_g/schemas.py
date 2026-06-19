from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    run_dir: Path
    gripper_name: str
    object_name: str

    @property
    def render_dir(self) -> Path:
        return self.run_dir / "render"

    @property
    def prompts_dir(self) -> Path:
        return self.run_dir / "prompts"

    @property
    def visualizations_dir(self) -> Path:
        return self.run_dir / "visualizations"

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)

