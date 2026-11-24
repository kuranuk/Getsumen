"""重い計算 (レイトレーシング / ライダー / 影の再計算) を別プロセスで実行する.

メインプロセス (asyncio) の GIL を塞がないよう ProcessPoolExecutor で並列化する。
各ワーカーは同じ設定(seed)から同じ World を決定論的に再生成するので、地形データの転送は不要。
"""
from __future__ import annotations

from .lidar import FlashLidar
from .raytrace import RayTracer
from .world import World

_S: dict = {}  # ワーカープロセスごとの World / RayTracer / FlashLidar


def init(cfg: dict) -> None:
    world = World(cfg)
    _S["world"] = world
    _S["rt"] = RayTracer(cfg, world)
    _S["lidars"] = [FlashLidar(c, world, seed=100 + i) for i, c in enumerate(cfg["sensors"]["lidars"])]


def render_camera(name: str, pose, sun_az: float, sun_el: float, shade, rock_shade):
    world = _S["world"]
    world.sun_azimuth, world.sun_elevation = sun_az, sun_el  # 太陽はメイン側の値に合わせる (ラジアン)
    return _S["rt"].render(name, pose, shade, rock_shade)


def scan_lidar(idx: int, pose):
    return _S["lidars"][idx].scan(pose)


def bake(az_deg: float, el_deg: float):
    """太陽の方位/仰角 [deg] を設定して影を再計算. 戻り値: (方位rad, 仰角rad, 地形の影, 岩の影)."""
    world = _S["world"]
    world.set_sun(az_deg, el_deg)
    return world.sun_azimuth, world.sun_elevation, world.bake_shade(), world.bake_rock_shade()
