"""前面フラッシュライダー: 設定した視野/画素数の格子状に一括照射し、距離画像→点群を生成."""
from __future__ import annotations

import numpy as np

from .geometry import mount_position, mount_rotation, rot_zyx
from .world import World


class FlashLidar:
    def __init__(self, cfg: dict, world: World, seed: int = 0):
        self.world = world
        self.name = cfg["name"]
        self.rate = float(cfg["rate_hz"])
        self.tmin = float(cfg["min_range"])
        self.tmax = float(cfg["max_range"])
        self.noise = float(cfg["range_noise_std"])
        self.dropout = float(cfg["dropout"])
        # 検出限界 (max_range の基準反射率と、面の種別 [なし, 地形, 岩, HLS] ごとの反射率)
        self.rho_ref = float(cfg["reference_reflectivity"])
        self.rho = np.array([0.0, cfg["ground_reflectivity"], cfg["rock_reflectivity"], cfg["hls_reflectivity"]])
        self.rng = np.random.default_rng(seed)

        # 画素ごとの方向 (センサ座標: x=前, y=左, z=上) を車体座標へ
        az = np.radians(np.linspace(cfg["h_fov_deg"] / 2, -cfg["h_fov_deg"] / 2, int(cfg["h_pixels"])))
        el = np.radians(np.linspace(cfg["v_fov_deg"] / 2, -cfg["v_fov_deg"] / 2, int(cfg["v_pixels"])))
        AZ, EL = np.meshgrid(az, el)
        d = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
        self.m_pos = mount_position(cfg["mount"])
        self.dirs_body = d @ mount_rotation(cfg["mount"]).T

    def scan(self, pose) -> np.ndarray:
        """pose=(x,y,z,yaw,pitch,roll). 戻り値: (n,4) float32 [x,y,z,range] (車体座標系)."""
        R = rot_zyx(*pose[3:])
        origin = np.array(pose[:3]) + R @ self.m_pos
        d_world = self.dirs_body @ R.T
        t, kind, rid = self.world.raycast(origin, d_world, self.tmin, self.tmax)

        # 検出限界: 受信強度 ∝ 反射率 x cos(入射角) / 距離^2。地面すれすれや暗い面ほど短い距離で検出できなくなる
        hit = np.isfinite(t)
        normals = self.world.surface_normals(origin + d_world * np.where(hit, t, 0.0)[:, None], kind, rid)
        cos_inc = np.abs((normals * d_world).sum(1))
        r_eff = self.tmax * np.sqrt(self.rho[kind] * cos_inc / self.rho_ref)
        valid = hit & (t <= np.minimum(r_eff, self.tmax))
        valid &= self.rng.random(len(t)) >= self.dropout

        t = t[valid] + self.rng.normal(0.0, self.noise, valid.sum())
        pts = self.m_pos + self.dirs_body[valid] * t[:, None]
        return np.column_stack([pts, t]).astype(np.float32)
