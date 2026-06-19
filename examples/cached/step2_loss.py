import numpy as np


TARGETS = [
    {
        "name": "SideY_Stem_Pinch",
        "point": np.array([0.0016, 0.0070, 0.0902], dtype=float),
        "category_weight": 0.0,
    },
    {
        "name": "SideX_Head_Grasp",
        "point": np.array([0.0004, 0.0072, 0.1235], dtype=float),
        "category_weight": 0.35,
    },
]


def calculate_loss(pose_mat: np.ndarray, point_cloud: np.ndarray) -> float:
    if pose_mat.shape != (4, 4):
        return 1e6
    if not np.all(np.isfinite(pose_mat)):
        return 1e6

    position = np.asarray(pose_mat[:3, 3], dtype=float)
    points = np.asarray(point_cloud[:, :3], dtype=float)
    if len(points) == 0:
        return 1e6

    nearest_dist = float(np.min(np.linalg.norm(points - position[None, :], axis=1)))
    z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
    centerline = np.array([0.0, 0.0, position[2]], dtype=float)
    radial = float(np.linalg.norm(position[:2] - centerline[:2]))

    target_terms = []
    for target in TARGETS:
        target_distance = float(np.linalg.norm(position - target["point"]))
        target_terms.append(target_distance + float(target["category_weight"]))
    semantic_target = min(target_terms)

    contact_term = abs(nearest_dist - 0.006)
    penetration_penalty = max(0.0, 0.002 - nearest_dist) * 10.0
    height_penalty = max(0.0, z_min - position[2]) + max(0.0, position[2] - z_max)
    radial_penalty = max(0.0, radial - 0.035)
    approach_bonus = -0.015 * abs(float(pose_mat[2, 2]))

    return float(
        1.0 * semantic_target
        + 0.45 * contact_term
        + 0.4 * penetration_penalty
        + 0.5 * height_penalty
        + 0.2 * radial_penalty
        + approach_bonus
    )
