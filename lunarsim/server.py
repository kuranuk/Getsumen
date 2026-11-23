"""aiohttp サーバ: 物理ループ, ライダーループ, WebSocket 配信, 静的ファイル."""
from __future__ import annotations

import asyncio
import json
import struct
import time
from pathlib import Path

from aiohttp import WSMsgType, web

from .lidar import FlashLidar
from .rover import Rover
from .world import World

WEB_DIR = Path(__file__).parent / "web"
MSG_LIDAR = 1


class SimServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.world = World(cfg)
        self.rover = Rover(cfg["rover"], self.world)
        self.lidars = [FlashLidar(c, self.world, seed=i) for i, c in enumerate(cfg["sensors"]["lidars"])]
        self.clients: set[web.WebSocketResponse] = set()
        self.cmd = (0.0, 0.0)
        self.cmd_time = 0.0
        self.sim_t = 0.0
        self.shade = self.world.bake_shade()
        self.sun_version = 0
        self._sun_pending = None
        self._sun_busy = False

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
                self.world.set_sun(az, el)
                self.shade = await loop.run_in_executor(None, self.world.bake_shade)
                self.sun_version += 1
                await self._broadcast(json.dumps(self.sun_message()))
        finally:
            self._sun_busy = False

    def sun_message(self) -> dict:
        import math
        return {"t": "sun", "az": self.world.sun_azimuth, "el": self.world.sun_elevation,
                "az_deg": math.degrees(self.world.sun_azimuth), "el_deg": math.degrees(self.world.sun_elevation),
                "v": self.sun_version}

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
            if now - self.cmd_time > 0.5:  # 指令途絶 -> 停止
                v = w = 0.0
            self.rover.step(dt, v, w)
            self.sim_t += dt

    async def state_loop(self):
        period = 1.0 / float(self.cfg["simulation"]["state_hz"])
        while True:
            await asyncio.sleep(period)
            if self.clients:
                await self._broadcast(json.dumps(self.rover.state(self.sim_t)))

    async def lidar_loop(self, idx: int):
        lidar = self.lidars[idx]
        loop = asyncio.get_running_loop()
        period = 1.0 / lidar.rate
        while True:
            t0 = loop.time()
            if self.clients:
                pose = self.rover.pose()
                pts = await loop.run_in_executor(None, lidar.scan, pose)
                header = struct.pack("<BBxxIf", MSG_LIDAR, idx, len(pts), self.sim_t)
                await self._broadcast(header + pts.tobytes())
            await asyncio.sleep(max(0.005, period - (loop.time() - t0)))

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
            if self.cmd != (0.0, 0.0):
                self.rover.auto = False
        elif t == "auto":
            self.rover.auto = bool(msg.get("on", False))
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
        app["tasks"] += [asyncio.create_task(sim.lidar_loop(i)) for i in range(len(sim.lidars))]

    async def on_cleanup(app):
        for t in app["tasks"]:
            t.cancel()

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
