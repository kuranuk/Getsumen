import math
import unittest

import numpy as np

from lunarsim.config import load_config
from lunarsim.lidar import FlashLidar
from lunarsim.rover import Rover
from lunarsim.world import World


class WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config("config/default.yaml")
        cls.world = World(cls.cfg)

    def test_terrain_is_mild(self):
        h = self.world.heights
        self.assertLess(float(np.abs(h).max()), 9.0)
        gy, gx = np.gradient(h, self.world.res)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        self.assertLess(float(np.percentile(slope, 99)), 22.0)

    def test_deterministic(self):
        w2 = World(self.cfg)
        self.assertTrue(np.array_equal(w2.heights, self.world.heights))

    def test_ray_hits_flat_ground(self):
        o = np.array([-60.0, 60.0, 10.0])  # 平坦に近い遠方ではなく単純に下向き
        d = np.array([[0.0, 0.0, -1.0]])
        t = self.world.raycast(o, d, 0.1, 100.0)
        self.assertAlmostEqual(float(t[0]), 10.0 - float(self.world.height(o[0], o[1])), delta=0.02)

    def test_ray_hits_hls(self):
        h = self.world.hls
        o = np.array([h["x"] - 30.0, h["y"], self.world.hls_z0 + 2.0])
        t = self.world.raycast(o, np.array([[1.0, 0.0, 0.0]]), 0.1, 100.0)
        self.assertAlmostEqual(float(t[0]), 30.0 - h["radius"], delta=0.05)

    def test_lidar_scan(self):
        lidar = FlashLidar(self.cfg["sensors"]["lidars"][0], self.world)
        rover = Rover(self.cfg["rover"], self.world)
        pts = lidar.scan(rover.pose())
        self.assertGreater(len(pts), 1000)
        self.assertEqual(pts.shape[1], 4)
        self.assertTrue(np.all(pts[:, 3] <= lidar.tmax + 0.5))

    def test_rover_drives(self):
        rover = Rover(self.cfg["rover"], self.world)
        x0, y0 = rover.x, rover.y
        for _ in range(200):
            rover.step(0.02, 1.0, 0.0)
        self.assertGreater(math.hypot(rover.x - x0, rover.y - y0), 1.0)


if __name__ == "__main__":
    unittest.main()


class RayTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lunarsim.raytrace import RayTracer
        cls.cfg = load_config("config/default.yaml")
        cls.world = World(cls.cfg)
        cls.rt = RayTracer(cls.cfg, cls.world)
        cls.shade = cls.world.bake_shade()
        cls.rock = cls.world.bake_rock_shade()

    def test_render_shape_and_hdr(self):
        rover = Rover(self.cfg["rover"], self.world)
        img = self.rt.render("front", rover.pose(), self.shade, self.rock).astype(np.float32)
        c = self.cfg["sensors"]["cameras"][0]
        self.assertEqual(img.shape, (c["height"], c["width"], 3))
        self.assertTrue(np.isfinite(img).all())
        self.assertLess(float(img[:5].max()), 1e-3)      # 上端は空 (黒)
        self.assertGreater(float(img[-20:].mean()), 0.0)  # 下は地面

    def test_sun_disc_visible_when_facing_sun(self):
        rover = Rover(self.cfg["rover"], self.world)
        x, y, z, _, p, r = rover.pose()
        # 太陽方位を向いたローバー: 前方カメラの中央に太陽円盤 (極端に高い輝度) が写る
        pose = (x, y, z, self.world.sun_azimuth, 0.0, 0.0)
        img = self.rt.render("front", pose, self.shade, self.rock).astype(np.float32)
        self.assertGreater(float(img.max()), 1e3)

    def test_shadow_of_hls_is_dark(self):
        w = self.world
        h = w.hls
        L = w.sun_dir()
        # HLSの影の中の地面点
        d = -L[:2] / np.linalg.norm(L[:2])
        P = np.array([[h["x"] + d[0] * 60.0, h["y"] + d[1] * 60.0, 0.0]])
        self.assertTrue(bool(self.rt._hls_shadow(P, L)[0]))
        P2 = np.array([[h["x"] - d[0] * 60.0, h["y"] - d[1] * 60.0, 0.0]])
        self.assertFalse(bool(self.rt._hls_shadow(P2, L)[0]))
