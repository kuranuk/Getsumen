"""重い計算 (レイトレーシング / ライダー / 影の再計算) を別プロセスで実行する.

メインプロセス (asyncio) の GIL を塞がないよう ProcessPoolExecutor で並列化する。
各ワーカーは同じ設定(seed)から同じ World を決定論的に再生成するので、地形データの転送は不要。
"""
from __future__ import annotations

_S: dict = {}


def init(cfg: dict) -> None:
    from .lidar import FlashLidar
    from .raytrace import RayTracer
    from .world import World

    world = World(cfg)
    _S["world"] = world
    _S["rt"] = RayTracer(cfg, world)
    _S["lidars"] = [FlashLidar(c, world, seed=100 + i) for i, c in enumerate(cfg["sensors"]["lidars"])]


def _set_sun(az: float, el: float) -> None:
    w = _S["world"]
    w.sun_azimuth, w.sun_elevation = az, el


def render_camera(name: str, pose, az: float, el: float, shade, rock_shade):
    _set_sun(az, el)
    return _S["rt"].render(name, pose, shade, rock_shade)


def scan_lidar(idx: int, pose):
    return _S["lidars"][idx].scan(pose)


def bake(az_deg: float, el_deg: float):
    w = _S["world"]
    w.set_sun(az_deg, el_deg)
    return w.sun_azimuth, w.sun_elevation, w.bake_shade(), w.bake_rock_shade()
