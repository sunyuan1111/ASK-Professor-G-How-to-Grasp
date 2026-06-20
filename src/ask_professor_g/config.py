from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"^\$\{([^}:]+)(?::-([^}]*))?\}$")


def load_dotenv(path: str | Path = ".env") -> None:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if not isinstance(value, str):
        return value

    match = _ENV_PATTERN.match(value)
    if not match:
        return value
    name, default = match.groups()
    return os.environ.get(name, default or "")


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _resolve_env(data)


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]
    repo_root: Path

    @property
    def gripper_name(self) -> str:
        return self.raw["selection"]["gripper"]

    @property
    def object_name(self) -> str:
        return self.raw["selection"]["object"]

    @property
    def output_root(self) -> Path:
        return self.repo_root / self.raw.get("run", {}).get("output_root", "runs")

    @property
    def top_k(self) -> int:
        return int(self.raw.get("run", {}).get("top_k", 1))

    @property
    def seed(self) -> int:
        return int(self.raw.get("run", {}).get("seed", 7))

    @property
    def cem(self) -> dict[str, Any]:
        return self.raw.get("run", {}).get("cem", {})

    @property
    def llm(self) -> dict[str, Any]:
        return self.raw.get("llm", {})

    @property
    def grippers_path(self) -> Path:
        return self.repo_root / self.raw["paths"]["grippers"]

    @property
    def objects_path(self) -> Path:
        return self.repo_root / self.raw["paths"]["objects"]


def load_config(
    config_path: str | Path,
    *,
    gripper: str | None = None,
    obj: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    config_path = Path(config_path)
    repo_root = config_path.resolve().parents[1] if config_path.parent.name == "configs" else Path.cwd()
    if not (repo_root / ".git").exists():
        repo_root = Path.cwd()
    load_dotenv(repo_root / ".env")

    raw = load_yaml(config_path)
    if overrides:
        raw = _deep_update(raw, overrides)
    if gripper:
        raw.setdefault("selection", {})["gripper"] = gripper
    if obj:
        raw.setdefault("selection", {})["object"] = obj

    return PipelineConfig(raw=raw, repo_root=repo_root.resolve())

