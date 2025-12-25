"""Camera geometry helpers for MuJoCo environments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None  # type: ignore


@dataclass(frozen=True)
class CameraParameters:
    name: str
    position: np.ndarray
    quaternion: np.ndarray
    fovy: float
    resolution: Tuple[int, int]


def get_camera_parameters(model: "mujoco.MjModel", camera_name: str, resolution: Tuple[int, int] = (224, 224)) -> CameraParameters:
    if mujoco is None:  # pragma: no cover
        raise ImportError("MuJoCo is required for camera utilities")

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found in model")

    pos = model.cam_pos[cam_id].copy()
    quat = model.cam_quat[cam_id].copy()
    fovy = float(model.cam_fovy[cam_id])
    # Use provided resolution as cam_resolution may not be set in XML
    return CameraParameters(camera_name, pos, quat, fovy, resolution)


def project_point(point: np.ndarray, params: CameraParameters) -> np.ndarray:
    if mujoco is None:  # pragma: no cover
        raise ImportError("MuJoCo is required for camera utilities")

    point = np.asarray(point, dtype=np.float64)
    cam_pos = params.position
    cam_quat = params.quaternion

    rot = np.zeros((3, 3), dtype=np.float64)
    mujoco.mju_quat2Mat(rot.ravel(), cam_quat)
    cam_frame = rot.T @ (point - cam_pos)
    if cam_frame[2] <= 1e-6:
        raise ValueError("Point is behind the camera")

    height, width = params.resolution[1], params.resolution[0]
    fovy_rad = np.deg2rad(params.fovy)
    focal = 0.5 * height / np.tan(0.5 * fovy_rad)
    u = width * 0.5 + (cam_frame[0] / cam_frame[2]) * focal
    v = height * 0.5 - (cam_frame[1] / cam_frame[2]) * focal
    return np.array([u, v], dtype=np.float64)


__all__ = ["CameraParameters", "get_camera_parameters", "project_point"]
