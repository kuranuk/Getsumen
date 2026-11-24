"""有人与圧ローバーの簡易運動モデル (ユニサイクル + 地形追従)."""
from __future__ import annotations

import math
import threading

import numpy as np

from .geometry import wrap_pi
from .world import World


class Rover:
    def __init__(self, cfg: dict, world: World):
        self.c = cfg
        self.world = world
        self.L = float(cfg["length"])
        self.W = float(cfg["width"])
        self.wheel_r = float(cfg["wheel_radius"])
        self.vmax = float(cfg["max_speed"])
        self.wmax = math.radians(cfg["max_yaw_rate_deg"])
        self.acc = float(cfg["accel"])
        self.wacc = math.radians(cfg["yaw_accel_deg"])
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        s = self.c["start"]
        with self._lock:
            self.x, self.y = float(s["x"]), float(s["y"])
            self.yaw = math.radians(s.get("yaw_deg", 0.0))
            self.v = self.w = 0.0
            self.odo = 0.0
            z, pitch, roll = self._terrain_pose(self.x, self.y, self.yaw)
            self.z, self.pitch, self.roll = z, pitch, roll

    # ---- 地形追従 ----
    def _terrain_pose(self, x, y, yaw):
        a, b = self.L * 0.4, self.W / 2.0
        cy, sy = math.cos(yaw), math.sin(yaw)
        pts = np.array([[a, 0], [-a, 0], [0, b], [0, -b]], float)
        wx = x + cy * pts[:, 0] - sy * pts[:, 1]
        wy = y + sy * pts[:, 0] + cy * pts[:, 1]
        hf, hr, hl, hrt = self.world.height(wx, wy)
        slope_f = (hf - hr) / (2 * a)
        slope_l = (hl - hrt) / (2 * b)
        z = (hf + hr + hl + hrt) / 4.0 + self.wheel_r
        return float(z), float(-math.atan(slope_f)), float(math.atan(slope_l))

    # ---- 運動 ----
    def step(self, dt: float, cmd_v: float, cmd_w: float):
        with self._lock:
            tv = max(-1.0, min(1.0, cmd_v)) * self.vmax
            tw = max(-1.0, min(1.0, cmd_w)) * self.wmax
            self.v += max(-self.acc * dt, min(self.acc * dt, tv - self.v))
            self.w += max(-self.wacc * dt, min(self.wacc * dt, tw - self.w))

            nyaw = self.yaw + self.w * dt
            mid = 0.5 * (self.yaw + nyaw)
            nx = self.x + self.v * math.cos(mid) * dt
            ny = self.y + self.v * math.sin(mid) * dt

            # HLS との衝突 (円近似) -> 進入を止める
            h = self.world.hls
            keep = h["radius"] + self.L / 2.0
            if math.hypot(nx - h["x"], ny - h["y"]) < keep and \
               math.hypot(nx - h["x"], ny - h["y"]) < math.hypot(self.x - h["x"], self.y - h["y"]):
                nx, ny, self.v = self.x, self.y, 0.0
            half = self.world.size / 2.0 - 2.0
            nx, ny = max(-half, min(half, nx)), max(-half, min(half, ny))

            self.odo += math.hypot(nx - self.x, ny - self.y)
            self.x, self.y, self.yaw = nx, ny, wrap_pi(nyaw)

            z, pitch, roll = self._terrain_pose(self.x, self.y, self.yaw)
            k = min(1.0, dt * 8.0)  # サスペンションの簡易ローパス
            self.z += (z - self.z) * k
            self.pitch += (pitch - self.pitch) * k
            self.roll += (roll - self.roll) * k

    def pose(self) -> tuple[float, float, float, float, float, float]:
        with self._lock:
            return (self.x, self.y, self.z, self.yaw, self.pitch, self.roll)

    def state(self, sim_t: float) -> dict:
        with self._lock:
            return {
                "t": "state", "sim_t": round(sim_t, 3),
                "x": self.x, "y": self.y, "z": self.z,
                "yaw": self.yaw, "pitch": self.pitch, "roll": self.roll,
                "v": self.v, "w": self.w, "odo": self.odo,
            }
