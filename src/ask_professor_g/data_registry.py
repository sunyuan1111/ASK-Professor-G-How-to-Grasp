from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig, load_yaml


@dataclass(frozen=True)
class GripperSpec:
    key: str
    raw: dict[str, Any]
    repo_root: Path

    def path(self, field: str) -> Path:
        return self.repo_root / self.raw[field]

    @property
    def max_opening_width(self) -> float:
        return float(self.raw.get("max_opening_width", 0.08))

    @property
    def min_opening_width(self) -> float:
        return float(self.raw.get("min_opening_width", 0.0))

    @property
    def tcp_offset(self) -> list[float]:
        return list(self.raw.get("tcp_offset", self.raw.get("tcp_wrap", [0.0, 0.0, 0.0])))


@dataclass(frozen=True)
class ObjectSpec:
    key: str
    raw: dict[str, Any]
    repo_root: Path

    def path(self, field: str) -> Path:
        return self.repo_root / self.raw[field]

    @property
    def mesh_path(self) -> Path:
        return self.path("mesh")

    @property
    def point_cloud_path(self) -> Path:
        if self.raw.get("point_cloud"):
            return self.path("point_cloud")
        return self.mesh_path


class DataRegistry:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.grippers = load_yaml(config.grippers_path).get("grippers", {})
        self.objects = load_yaml(config.objects_path).get("objects", {})

    def gripper(self, name: str | None = None) -> GripperSpec:
        key = name or self.config.gripper_name
        if key not in self.grippers:
            raise KeyError(f"Unknown gripper '{key}'. Available: {', '.join(sorted(self.grippers))}")
        return GripperSpec(key=key, raw=self.grippers[key], repo_root=self.config.repo_root)

    def object(self, name: str | None = None) -> ObjectSpec:
        key = name or self.config.object_name
        if key not in self.objects:
            raise KeyError(f"Unknown object '{key}'. Available: {', '.join(sorted(self.objects))}")
        return ObjectSpec(key=key, raw=self.objects[key], repo_root=self.config.repo_root)

    def validate_paths(self, gripper: GripperSpec, obj: ObjectSpec) -> list[str]:
        missing: list[str] = []
        for label, path in [
            ("gripper.urdf", gripper.path("urdf")),
            ("gripper.config", gripper.path("config")),
            ("object.mesh", obj.mesh_path),
            ("object.point_cloud", obj.point_cloud_path),
        ]:
            if not path.exists():
                missing.append(f"{label}: {path}")
        return missing

