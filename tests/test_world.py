import math
import unittest

import numpy as np

from server import FlashLidar, RayTracer, Rover, World, load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_applied_to_sensors(self):
        cfg = load_config()
        cam = cfg["sensors"]["cameras"][0]
        self.assertEqual(cam["width"], 320)              # camera_defaults から
        self.assertEqual(cam["label"], "前 (Front)")
        self.assertIn("reference_reflectivity", cfg["sensors"]["lidars"][0])

    def test_override_is_a_diff(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write("world: {seed: 3}\nsensors:\n  cameras:\n    - {name: only, mount: {x: 1.0}, width: 64, height: 48}\n")
        cfg = load_config(f.name)
        self.assertEqual(cfg["world"]["seed"], 3)
        self.assertEqual(cfg["world"]["size"], 240.0)    # 書いていない項目は既定値
        self.assertEqual([c["name"] for c in cfg["sensors"]["cameras"]], ["only"])
        self.assertEqual(cfg["sensors"]["cameras"][0]["fov_deg"], 90)


class WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.world = World(cls.cfg)

    def test_terrain_is_mild(self):
        h = self.world.heights
        self.assertLess(float(np.abs(h).max()), 9.0)
        gy, gx = np.gradient(h, self.world.res)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        self.assertLess(float(np.percentile(slope, 99)), 22.0)

    def test_deterministic(self):
        self.assertTrue(np.array_equal(World(self.cfg).heights, self.world.heights))

    def test_ray_hits_ground(self):
        o = np.array([-60.0, 60.0, 10.0])
        t, kind = self.world.raycast(o, np.array([[0.0, 0.0, -1.0]]), 0.1, 100.0)
        self.assertAlmostEqual(float(t[0]), 10.0 - float(self.world.height(o[0], o[1])), delta=0.02)
        self.assertEqual(kind[0], World.TERRAIN)

    def test_ray_hits_hls_side_and_top(self):
        h = self.world.hls
        z0, top = self.world.hls_z0, self.world.hls_z0 + h["height"]
        t, kind = self.world.raycast(np.array([h["x"] - 30.0, h["y"], z0 + 2.0]), np.array([[1.0, 0.0, 0.0]]), 0.1, 100.0)
        self.assertAlmostEqual(float(t[0]), 30.0 - h["radius"], delta=0.05)
        self.assertEqual(kind[0], World.HLS)
        t, kind = self.world.raycast(np.array([h["x"], h["y"], top + 5.0]), np.array([[0.0, 0.0, -1.0]]), 0.1, 100.0)
        self.assertAlmostEqual(float(t[0]), 5.0, delta=0.05)  # 上面

    def test_ray_beyond_terrain_hits_plane(self):
        o = np.array([self.world.size / 2 - 3.0, 0.0, 2.0])
        d = np.array([[1.0, 0.0, -0.01]])  # 地形の外側へ向かう浅い光線 -> z=0 の平面に当たる
        d /= np.linalg.norm(d)
        t, kind = self.world.raycast(o, d, 0.1, 300.0)
        self.assertEqual(kind[0], World.TERRAIN)
        self.assertAlmostEqual(float(o[2] + d[0, 2] * t[0]), 0.0, delta=1e-6)

    def test_lidar_scan(self):
        lidar = FlashLidar(self.cfg["sensors"]["lidars"][0], self.world)
        pts = lidar.scan(Rover(self.cfg["rover"], self.world).pose())
        self.assertGreater(len(pts), 1000)
        self.assertEqual(pts.shape[1], 4)
        self.assertTrue(np.all(pts[:, 3] <= lidar.tmax + 0.5))

    def test_lidar_range_follows_pitch(self):
        # 平坦なパッド上で俯角を深くすると、地面に当たる距離は センサ高/tan(俯角) より遠くならない
        cfg = load_config()
        c = cfg["sensors"]["lidars"][0]
        c.update(range_noise_std=0.0, dropout=0.0)
        c["mount"]["pitch_deg"] = -20
        rover = Rover(cfg["rover"], self.world)
        rover.x, rover.y, rover.yaw = -9.0, 0.0, math.pi  # HLSと反対向き
        z, pitch, roll = rover._terrain_pose(rover.x, rover.y, rover.yaw)
        rover.z, rover.pitch, rover.roll = z, pitch, roll
        pts = FlashLidar(c, self.world).scan(rover.pose())
        far = np.hypot(pts[:, 0] - c["mount"]["x"], pts[:, 1]).max()
        self.assertLess(far, 1.3 / math.tan(math.radians(5)) * 1.5)  # 視野の上端は -5° (期待 約15 m)

    def test_lidar_nearest_row_is_an_arc(self):
        # 画素方向は方位・仰角を等間隔に振る球面走査 (ピンホールではない)。
        # 平らな地面では、仰角が一定の行は弧になる: 最近の行は、中央より両端が手前 (車体の後ろ寄り) になる。
        import copy
        flat = copy.copy(self.world)
        flat.heights = np.zeros_like(self.world.heights)
        cfg = load_config()
        c = cfg["sensors"]["lidars"][0]
        c.update(range_noise_std=0.0, dropout=0.0, reference_reflectivity=1e-9)  # 検出限界は外して形だけを見る
        c["mount"]["pitch_deg"] = -20
        rover = Rover(cfg["rover"], flat)
        rover.x, rover.y, rover.yaw = -9.0, 0.0, math.pi
        rover.z, rover.pitch, rover.roll = 0.6, 0.0, 0.0
        pts = FlashLidar(c, flat).scan(rover.pose())
        hp, vp = c["h_pixels"], c["v_pixels"]
        self.assertEqual(len(pts), hp * vp)
        near = pts[:, :2].reshape(vp, hp, 2)[-1]  # 最近の行 (車体座標の x=前, y=左)
        self.assertGreater(float(near[hp // 2, 0] - near[0, 0]), 0.05)    # 中央が前、左端が手前
        self.assertGreater(float(near[hp // 2, 0] - near[-1, 0]), 0.05)   # 右端も手前
        self.assertAlmostEqual(float(near[0, 0]), float(near[-1, 0]), delta=0.01)  # 左右対称
        sensor = np.array([c["mount"]["x"], c["mount"]["y"]])
        r = np.hypot(*(near - sensor).T)                                  # センサ直下からの距離はほぼ一定 (弧)
        self.assertLess(float(np.ptp(r)), 0.25 * float(r.mean()))

    def test_rover_drives(self):
        rover = Rover(self.cfg["rover"], self.world)
        x0, y0 = rover.x, rover.y
        for _ in range(200):
            rover.step(0.02, 1.0, 0.0)
        self.assertGreater(math.hypot(rover.x - x0, rover.y - y0), 1.0)


class RayTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.world = World(cls.cfg)
        cls.rt = RayTracer(cls.cfg, cls.world)
        cls.shade = cls.world.bake_shade()

    def render(self, name, pose):
        return self.rt.render(name, pose, self.shade).astype(np.float32)

    def test_render_shape_and_hdr(self):
        img = self.render("front", Rover(self.cfg["rover"], self.world).pose())
        c = self.cfg["sensors"]["cameras"][0]
        self.assertEqual(img.shape, (c["height"], c["width"], 3))
        self.assertTrue(np.isfinite(img).all())
        self.assertLess(float(img[:5].max()), 1e-3)       # 上端は空 (黒)
        self.assertGreater(float(img[-20:].mean()), 0.0)  # 下は地面

    def test_sun_disc_visible_when_facing_sun(self):
        x, y, z, *_ = Rover(self.cfg["rover"], self.world).pose()
        img = self.render("front", (x, y, z, self.world.sun_azimuth, 0.0, 0.0))  # 太陽の方位を向く
        self.assertGreater(float(img.max()), 1e3)

    def test_shadow_of_hls(self):
        w, L = self.world, self.world.sun_dir()
        away = -L[:2] / np.linalg.norm(L[:2])  # 太陽と反対側 = 影の側
        h = w.hls
        shadowed = np.array([[h["x"] + away[0] * 60.0, h["y"] + away[1] * 60.0, 0.0]])
        sunlit = np.array([[h["x"] - away[0] * 60.0, h["y"] - away[1] * 60.0, 0.0]])
        self.assertTrue(bool(self.rt._hls_shadow(shadowed, L)[0]))
        self.assertFalse(bool(self.rt._hls_shadow(sunlit, L)[0]))


if __name__ == "__main__":
    unittest.main()
