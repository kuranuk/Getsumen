"""設定ファイルの読み込み (デフォルト値とのディープマージ)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "server": {"host": "0.0.0.0", "port": 8080},
    "simulation": {"physics_hz": 50, "state_hz": 30, "workers": 0},  # workers: 0=CPU数
    "world": {
        "seed": 7,
        "size": 240.0,          # 地形一辺 [m]
        "resolution": 0.5,      # 高さマップ格子間隔 [m]
        "undulation": {"amplitude": 0.35, "wavelength": 45.0, "roughness": 0.04},
        # 月極域ハイランドの統計に基づく (LOLA/LROC/Diviner の公開値)
        "craters": {
            "diameter_min": 4.0,     # [m] 生成する最小直径
            "diameter_max": 80.0,    # [m] 最大直径
            "saturation": 0.03,      # 幾何学的飽和に対する割合 (累積 N(>D)=1.54*sat*D^-2 /m^2)
            "depth_ratio": 0.05,     # 平均 深さ/直径 (劣化クレータを含む。新鮮=0.1+)
        },
        "rocks": {
            "abundance": 0.02,       # 岩の面積被覆率 k (極域の典型 1-3%, 新しいクレータ周辺は高い)
            "min_diameter": 0.25,    # [m]
            "max_diameter": 2.0,     # [m]
            "rim_fraction": 0.35,    # クレータ縁に集中配置する割合
            "max_count": 2500,
        },
        # 南極: 太陽は常に地平線近く (仰角 0-1.5° 程度, 軸傾斜 1.54°)。方位角は約29.5日で一周
        "sun": {"azimuth_deg": 110.0, "elevation_deg": 1.5, "ambient": 0.03},
    },
    "hls": {
        "x": 0.0, "y": 0.0, "radius": 4.5, "height": 20.0,
        "pad_radius": 18.0,        # HLS周囲の平坦な着陸パッド半径 [m]
        "model": "cylinder",       # cylinder=単純な円柱 / detailed=ノーズ・ハッチ付き
        "color": "#d8dade",
        "hatch_azimuth_deg": 180.0,  # ハッチ(乗降口)の向き (detailed のみ描画)
        "hatch_height": 4.0,
    },
    "rover": {
        "start": {"x": -55.0, "y": -25.0, "yaw_deg": 25.0},
        "model": "box",            # box=単純な箱 / detailed=キャビン+6輪
        "color": "#e9eaec",
        "length": 6.0, "width": 4.0, "height": 2.4,   # 箱の寸法 [m] (車体座標 x前/y左/z上)
        "clearance": 0.6,          # 地面から箱の下面まで [m]
        "wheel_radius": 0.6,       # 接地高さ・車輪径 (運動モデル/ライダー高さの基準)
        "max_speed": 2.0,          # [m/s]
        "max_yaw_rate_deg": 25.0,  # [deg/s]
        "accel": 0.6,              # [m/s^2]
        "yaw_accel_deg": 30.0,     # [deg/s^2]
        "auto_standoff": 3.0,      # 自動アプローチの停止距離 (車体前端-HLS表面) [m]
    },
    "sensors": {"cameras": [], "lidars": []},
    "ui": {"lidar_view_range": 40.0, "exposure": 2.5, "view_fill": 0.25, "camera_grid_columns": 4},
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
        # 実機カメラ風の画像特性
        c["exposure"] = c.get("exposure", "auto")  # "auto" か 固定倍率
        c.setdefault("ev_bias", 0.0)
        c.setdefault("auto_exposure_max", 40.0)
        c.setdefault("read_noise", 0.006)    # 読み出しノイズ (飽和=1.0 に対する標準偏差)
        c.setdefault("shot_noise", 0.012)    # ショットノイズ係数
        c.setdefault("vignette", 0.35)       # 周辺減光 0-1
        c.setdefault("glare", 0.8)           # 太陽直視時のグレア強度 0-1
        c.setdefault("blur_px", 0.6)         # 光学ボケ [px]
        # 画像生成: raytrace=サーバ側(numpy)のレイトレーシング / raster=ブラウザのCG描画 (高速・近似)
        c.setdefault("render", "raytrace")
        c.setdefault("supersample", 1)       # レイトレの 1画素あたり n x n サンプル (エッジの滑らかさ。n^2倍の計算)
        c.setdefault("stars", True)          # 恒星 (等級に基づく放射輝度。露出が低いと写らない)
        c.setdefault("star_limit_mag", 7.0)  # 描く限界等級
        # 地面距離の目盛 (カメラ位置から平坦な接地面を仮定した推定値)
        c.setdefault("distance_overlay", True)
        c.setdefault("overlay_distances", [2, 4, 6, 8, 10, 15, 20, 30, 50])
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
