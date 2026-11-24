"""前面フラッシュライダー: 設定した視野/画素数の格子状に一括照射し、距離画像→点群を生成."""
from __future__ import annotations

import math

import numpy as np

from .geometry import mount_position, mount_rotation, rot_zyx
from .world import World


class FlashLidar:
    def __init__(self, cfg: dict, world: World, seed: int = 0):
        self.cfg = cfg
        self.world = world
        self.name = cfg["name"]
        self.rate = float(cfg["rate_hz"])
        self.tmin = float(cfg["min_range"])
        self.tmax = float(cfg["max_range"])
        self.noise = float(cfg["range_noise_std"])
        self.dropout = float(cfg["dropout"])
        # 検出限界: max_range は「基準反射率 reference_reflectivity の面に垂直入射したとき」の到達距離。
        # 受信強度 ∝ 反射率 x cos(入射角) / R^2 なので、地面すれすれ(cosが小さい)や暗い面ほど短くなる。0 で無効。
        self.rho_ref = float(cfg.get("reference_reflectivity", 0.10))
        self.rho = {1: float(cfg.get("ground_reflectivity", 0.12)), 2: float(cfg.get("rock_reflectivity", 0.12)),
                    3: float(cfg.get("hls_reflectivity", 0.70))}
        self.rng = np.random.default_rng(seed)

        hp, vp = int(cfg["h_pixels"]), int(cfg["v_pixels"])
        az = np.radians(np.linspace(cfg["h_fov_deg"] / 2, -cfg["h_fov_deg"] / 2, hp))
        el = np.radians(np.linspace(cfg["v_fov_deg"] / 2, -cfg["v_fov_deg"] / 2, vp))
        AZ, EL = np.meshgrid(az, el)
        d = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
        self.m_pos = mount_position(cfg["mount"])
        self.m_rot = mount_rotation(cfg["mount"])
        self.dirs_body = d @ self.m_rot.T  # 車体座標系での各画素の方向

    def scan(self, pose) -> np.ndarray:
        """pose=(x,y,z,yaw,pitch,roll). 戻り値: (n,4) float32 [x,y,z,range] (車体座標系)."""
        x, y, z, yaw, pitch, roll = pose
        R = rot_zyx(yaw, pitch, roll)
        origin = np.array([x, y, z]) + R @ self.m_pos
        d_world = self.dirs_body @ R.T
        if self.rho_ref > 0:
            t, kind, rid = self.world.raycast(origin, d_world, self.tmin, self.tmax, detail=True)
            hit = np.isfinite(t)
            tt = np.where(hit, t, 0.0)
            n = self.world.surface_normals(origin + d_world * tt[:, None], kind, rid)
            cos_inc = np.abs((n * d_world).sum(1))
            rho = np.choose(np.clip(kind, 0, 3), [0.0, self.rho[1], self.rho[2], self.rho[3]])
            r_eff = np.minimum(self.tmax, self.tmax * np.sqrt(rho * cos_inc / self.rho_ref))
            t = np.where(hit & (t <= r_eff), t, np.inf)
        else:
            t = self.world.raycast(origin, d_world, self.tmin, self.tmax)
        valid = np.isfinite(t)
        if self.dropout > 0:
            valid &= self.rng.random(len(t)) >= self.dropout
        t = t[valid]
        if self.noise > 0:
            t = t + self.rng.normal(0.0, self.noise, len(t))
        pts = self.m_pos + self.dirs_body[valid] * t[:, None]
        return np.column_stack([pts, t]).astype(np.float32)
