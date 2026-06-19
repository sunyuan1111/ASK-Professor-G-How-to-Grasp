from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


LossFunction = Callable[[np.ndarray, np.ndarray], float]


def state_to_pose_matrix(state: np.ndarray) -> np.ndarray:
    x, y, z, roll, pitch, yaw = state[:6]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rz @ ry @ rx
    pose[:3, 3] = [x, y, z]
    return pose


@dataclass
class CEMOptimizer:
    num_samples: int = 128
    num_elites: int = 16
    max_iterations: int = 5
    seed: int = 7

    def optimize(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        point_cloud: np.ndarray,
        loss_func: LossFunction,
        *,
        clamp_min: np.ndarray | None = None,
        clamp_max: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, list[float]]:
        rng = np.random.default_rng(self.seed)
        mean = mean.astype(np.float64)
        std = np.maximum(std.astype(np.float64), 1e-6)
        history: list[float] = []
        best_state = mean.copy()
        best_loss = float("inf")

        for _ in range(self.max_iterations):
            samples = rng.normal(loc=mean, scale=std, size=(self.num_samples, mean.size))
            if clamp_min is not None:
                samples = np.maximum(samples, clamp_min)
            if clamp_max is not None:
                samples = np.minimum(samples, clamp_max)

            losses = np.array([loss_func(state_to_pose_matrix(sample), point_cloud) for sample in samples])
            elite_indices = np.argsort(losses)[: self.num_elites]
            elites = samples[elite_indices]
            if losses[elite_indices[0]] < best_loss:
                best_loss = float(losses[elite_indices[0]])
                best_state = samples[elite_indices[0]].copy()
            mean = 0.7 * elites.mean(axis=0) + 0.3 * mean
            std = np.maximum(0.7 * elites.std(axis=0) + 0.3 * std, 1e-6)
            history.append(best_loss)

        return best_state, best_loss, history

