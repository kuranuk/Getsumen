"""座標系ユーティリティ.

世界座標: x=東, y=北, z=上 (右手系)
車体座標: x=前, y=左, z=上
姿勢は R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。pitch は「機首下げ」が正 (y軸まわりの右手系回転)。
センサ取り付け角の pitch は設定ファイル上「上向きが正」(内部で符号反転)。
"""
from __future__ import annotations

import math

import numpy as np


def rot_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def mount_rotation(mount: dict) -> np.ndarray:
    """設定の mount(yaw_deg, pitch_deg[上向き正]) -> 車体座標系での回転行列."""
    yaw = math.radians(mount.get("yaw_deg", 0.0))
    pitch_up = math.radians(mount.get("pitch_deg", 0.0))
    return rot_zyx(yaw, -pitch_up, 0.0)


def mount_position(mount: dict) -> np.ndarray:
    return np.array([mount.get("x", 0.0), mount.get("y", 0.0), mount.get("z", 0.0)], float)


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
