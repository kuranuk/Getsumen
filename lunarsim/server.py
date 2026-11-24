"""aiohttp サーバ: 物理ループ, ライダーループ, WebSocket 配信, 静的ファイル."""
from __future__ import annotations

import asyncio
import json
import math
import multiprocessing as mp
import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from aiohttp import WSMsgType, web

from . import workers
from .raytrace import RayTracer
from .rover import Rover
from .world import World

WEB_DIR = Path(__file__).parent / "web"
MSG_LIDAR = 1
MSG_CAMERA = 2


class SimServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.world = World(cfg)
        self.rover = Rover(cfg["rover"], self.world)
        self.cameras = RayTracer(cfg, self.world).cams  # 名前 -> CameraModel (画像は workers で生成)
        self.lidars = cfg["sensors"]["lidars"]
        self.clients: set[web.WebSocketResponse] = set()
        self.cmd = (0.0, 0.0)  # (前進, 旋回) の指令 -1..1
        self.cmd_time = 0.0
        self.sim_t = 0.0
        # 太陽が決める影 (方位/仰角を変えると再計算する)
        self.shade = self.world.bake_shade()
        self.rock_shade = self.world.bake_rock_shade()
        self._sun_pending = None
        self._sun_busy = False

        # 重い計算 (レイトレ/ライダー/影) は別プロセスで。BLAS等のスレッドは1に固定して競合を避ける
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(k, "1")
        n_workers = int(cfg["simulation"]["workers"] or os.cpu_count() or 2)
        self.pool = ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn"),
                                        initializer=workers.init, initargs=(cfg,))

    # ---------------- 太陽 (方位/仰角の変更 -> 地形の影を再計算) ----------------
    async def set_sun(self, az_deg: float, el_deg: float):
        self._sun_pending = (az_deg, el_deg)
        if self._sun_busy:
            return  # 計算中なら最新値だけ保持
        self._sun_busy = True
        loop = asyncio.get_running_loop()
        try:
            while self._sun_pending is not None:
                az, el = self._sun_pending
                self._sun_pending = None
                az_r, el_r, shade, rock_shade = await loop.run_in_executor(self.pool, workers.bake, az, el)
                self.world.sun_azimuth, self.world.sun_elevation = az_r, el_r
                self.shade, self.rock_shade = shade, rock_shade
                await self._broadcast(json.dumps(self.sun_message()))
        finally:
            self._sun_busy = False

    def sun_message(self) -> dict:
        az, el = self.world.sun_azimuth, self.world.sun_elevation
        return {"t": "sun", "az": az, "el": el, "az_deg": math.degrees(az), "el_deg": math.degrees(el)}

    # ---------------- 配信用データ ----------------
    def world_json(self) -> dict:
        c = self.cfg
        return {
            "terrain": self.world.describe(),
            "rocks": [[round(float(v), 3) for v in r] for r in self.world.rocks],
            "hls": self.world.hls,
            "rover": c["rover"],
            "cameras": c["sensors"]["cameras"],
            "lidars": c["sensors"]["lidars"],
            "ui": c["ui"],
            "sun": {**self.sun_message(), "ambient": self.world.ambient},
        }

    def heightmap_bytes(self) -> bytes:
        return self.world.heights.astype("<f4").tobytes()

    # ---------------- ループ ----------------
    async def physics_loop(self):
        hz = float(self.cfg["simulation"]["physics_hz"])
        last = time.perf_counter()
        while True:
            await asyncio.sleep(1.0 / hz)
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now
            v, w = self.cmd
            if now - self.cmd_time > 1.0:  # 指令途絶 -> 停止
                v = w = 0.0
            self.rover.step(dt, v, w)
            self.sim_t += dt

    async def state_loop(self):
        period = 1.0 / float(self.cfg["simulation"]["state_hz"])
        while True:
            await asyncio.sleep(period)
            if self.clients:
                await self._broadcast(json.dumps(self.rover.state(self.sim_t)))

    async def _every(self, hz: float, job):
        """job() を約 hz で繰り返す (接続中のクライアントがいる間だけ)。"""
        loop = asyncio.get_running_loop()
        while True:
            t0 = loop.time()
            if self.clients:
                await job()
            await asyncio.sleep(max(0.005, 1.0 / hz - (loop.time() - t0)))

    async def lidar_job(self, idx: int):
        pts = await asyncio.get_running_loop().run_in_executor(self.pool, workers.scan_lidar, idx, self.rover.pose())
        await self._broadcast(struct.pack("<BBxxIf", MSG_LIDAR, idx, len(pts), self.sim_t) + pts.tobytes())

    async def camera_job(self, name: str):
        """レイトレーシングで画像を生成して配信 (HDR float16, 線形). 露出/ノイズ等はブラウザ側で加える."""
        cam, pose = self.cameras[name], self.rover.pose()
        img = await asyncio.get_running_loop().run_in_executor(
            self.pool, workers.render_camera, name, pose,
            self.world.sun_azimuth, self.world.sun_elevation, self.shade, self.rock_shade)
        header = struct.pack("<BBHHxxf6f", MSG_CAMERA, cam.index, cam.w, cam.h, self.sim_t, *pose)
        await self._broadcast(header + img.tobytes())

    async def _broadcast(self, data):
        async def send(ws):
            try:
                if isinstance(data, str):
                    await ws.send_str(data)
                else:
                    await ws.send_bytes(data)
            except Exception:
                self.clients.discard(ws)

        if self.clients:
            await asyncio.gather(*(send(ws) for ws in list(self.clients)))

    # ---------------- ハンドラ ----------------
    def handle_message(self, msg: dict):
        t = msg.get("t")
        if t == "drive":
            self.cmd = (float(msg.get("v", 0.0)), float(msg.get("w", 0.0)))
            self.cmd_time = time.perf_counter()
        elif t == "sun":
            asyncio.get_running_loop().create_task(
                self.set_sun(float(msg.get("az_deg", 0.0)), float(msg.get("el_deg", 1.5))))
        elif t == "reset":
            self.rover.reset()
            self.cmd = (0.0, 0.0)


def create_app(cfg: dict) -> web.Application:
    sim = SimServer(cfg)
    app = web.Application()
    app["sim"] = sim

    async def index(_):
        return web.FileResponse(WEB_DIR / "index.html")

    async def api_world(_):
        return web.json_response(sim.world_json())

    async def api_heightmap(_):
        return web.Response(body=sim.heightmap_bytes(), content_type="application/octet-stream")

    async def api_shade(_):
        return web.Response(body=sim.shade.tobytes(), content_type="application/octet-stream")

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)
        sim.clients.add(ws)
        try:
            async for m in ws:
                if m.type == WSMsgType.TEXT:
                    try:
                        sim.handle_message(json.loads(m.data))
                    except (ValueError, TypeError):
                        pass
        finally:
            sim.clients.discard(ws)
            if not sim.clients:
                sim.cmd = (0.0, 0.0)
        return ws

    async def on_startup(app):
        app["tasks"] = [asyncio.create_task(sim.physics_loop()), asyncio.create_task(sim.state_loop())]
        for i, c in enumerate(sim.lidars):
            app["tasks"].append(asyncio.create_task(sim._every(c["rate_hz"], lambda i=i: sim.lidar_job(i))))
        for name, cam in sim.cameras.items():
            app["tasks"].append(asyncio.create_task(sim._every(cam.cfg["rate_hz"], lambda n=name: sim.camera_job(n))))

    async def on_cleanup(app):
        for t in app["tasks"]:
            t.cancel()
        sim.pool.shutdown(wait=False, cancel_futures=True)

    @web.middleware
    async def no_cache(request, handler):
        resp = await handler(request)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    app.middlewares.append(no_cache)
    app.router.add_get("/", index)
    app.router.add_get("/api/world", api_world)
    app.router.add_get("/api/heightmap", api_heightmap)
    app.router.add_get("/api/shade", api_shade)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/", WEB_DIR, show_index=False)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app
