"""設定ファイルの読み込み (デフォルト値とのディープマージ)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "server": {"host": "0.0.0.0", "port": 8080},
    "simulation": {"physics_hz": 50, "state_hz": 30},
    "world": {
        "seed": 7,
        "size": 240.0,          # 地形一辺 [m]
        "resolution": 0.5,      # 高さマップ格子間隔 [m]
        "undulation": {"amplitude": 0.35, "wavelength": 45.0, "roughness": 0.04},
        "craters": {"count": 22, "radius_min": 2.0, "radius_max": 14.0, "depth_ratio": 0.10},
        "rocks": {"count": 450, "radius_min": 0.08, "radius_max": 0.7},
        "sun": {"azimuth_deg": 215.0, "elevation_deg": 14.0},
    },
    "hls": {
        "x": 0.0, "y": 0.0, "radius": 4.5, "height": 20.0,
        "pad_radius": 18.0,        # HLS周囲の平坦な着陸パッド半径 [m]
        "hatch_azimuth_deg": 180.0,  # ハッチ(乗降口)の向き (世界座標, x軸から反時計回り)
        "hatch_height": 4.0,
    },
    "rover": {
        "start": {"x": -55.0, "y": -25.0, "yaw_deg": 25.0},
        "length": 6.0, "width": 4.0, "wheel_radius": 0.6,
        "max_speed": 2.0,          # [m/s]
        "max_yaw_rate_deg": 25.0,  # [deg/s]
        "accel": 0.6,              # [m/s^2]
        "yaw_accel_deg": 30.0,     # [deg/s^2]
        "auto_standoff": 3.0,      # 自動アプローチの停止距離 (車体前端-HLS表面) [m]
    },
    "sensors": {"cameras": [], "lidars": []},
    "ui": {"lidar_view_range": 40.0},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None) -> dict:
    user = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    names = set()
    for c in cfg["sensors"]["cameras"]:
        for key in ("name", "width", "height", "fov_deg"):
            if key not in c:
                raise ValueError(f"camera に '{key}' がありません: {c}")
        c.setdefault("label", c["name"])
        c.setdefault("rate_hz", 10.0)
        c.setdefault("mount", {})
        if c["name"] in names:
            raise ValueError(f"センサ名が重複しています: {c['name']}")
        names.add(c["name"])
    for c in cfg["sensors"]["lidars"]:
        for key in ("name", "h_fov_deg", "v_fov_deg", "h_pixels", "v_pixels", "max_range"):
            if key not in c:
                raise ValueError(f"lidar に '{key}' がありません: {c}")
        c.setdefault("label", c["name"])
        c.setdefault("min_range", 0.5)
        c.setdefault("rate_hz", 5.0)
        c.setdefault("range_noise_std", 0.0)
        c.setdefault("dropout", 0.0)
        c.setdefault("mount", {})
        if c["name"] in names:
            raise ValueError(f"センサ名が重複しています: {c['name']}")
        names.add(c["name"])
