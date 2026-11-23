"""月面ワールド: 高さマップ(起伏+クレータ), 岩, HLS, およびレイキャスト."""
from __future__ import annotations

import math

import numpy as np


def _fractal_noise(rng, n: int, res: float, wavelength: float, beta: float = 2.2) -> np.ndarray:
    """1/f^beta 型の等方ノイズ (標準偏差1に正規化). wavelength より長い成分は抑制."""
    noise = rng.standard_normal((n, n))
    spec = np.fft.rfft2(noise)
    fy = np.fft.fftfreq(n, d=res)[:, None]
    fx = np.fft.rfftfreq(n, d=res)[None, :]
    f = np.hypot(fx, fy)
    f0 = 1.0 / max(wavelength, 1e-3)
    filt = (f * f + f0 * f0) ** (-beta / 2.0)
    filt[0, 0] = 0.0
    out = np.fft.irfft2(spec * filt, s=(n, n))
    return out / (out.std() + 1e-12)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class World:
    def __init__(self, cfg: dict):
        wc, hc, rc = cfg["world"], cfg["hls"], cfg["rover"]
        self.seed = int(wc["seed"])
        self.size = float(wc["size"])
        self.res = float(wc["resolution"])
        self.n = int(round(self.size / self.res)) + 1
        self.origin = -self.size / 2.0
        self.hls = dict(hc)
        self.rover_start = dict(rc["start"])
        rng = np.random.default_rng(self.seed)

        x = self.origin + np.arange(self.n) * self.res
        X, Y = np.meshgrid(x, x)  # [iy, ix]
        self._X, self._Y = X, Y

        # --- 平坦化マスク (HLSパッド周辺 / 地形端) ---
        r_hls = np.hypot(X - hc["x"], Y - hc["y"])
        pad = float(hc["pad_radius"])
        keep_pad = _smoothstep((r_hls - pad) / 20.0)
        edge = np.minimum(np.minimum(X - self.origin, -self.origin - X),
                          np.minimum(Y - self.origin, -self.origin - Y))
        keep_edge = _smoothstep(edge / 15.0)
        mask = keep_pad * keep_edge

        # --- 起伏 ---
        u = wc["undulation"]
        h = _fractal_noise(rng, self.n, self.res, u["wavelength"], beta=3.2) * u["amplitude"]
        h += _fractal_noise(rng, self.n, self.res, 4.0, beta=3.0) * u["roughness"]
        h *= mask

        # --- クレータ (マイルド: 浅いすり鉢 + なだらかなリム) ---
        self.craters = self._place_craters(rng, wc["craters"], pad)
        for cx, cy, R, d in self.craters:
            h += self._crater_profile(X, Y, cx, cy, R, d)
        self.heights = h.astype(np.float32)  # [iy, ix]

        # --- 岩 ---
        self.rocks = self._place_rocks(rng, wc["rocks"])  # (m,4): x,y,z,r
        self.hls_z0 = float(self.height(hc["x"], hc["y"]))
        self.sun_azimuth = math.radians(wc["sun"]["azimuth_deg"])
        self.sun_elevation = math.radians(wc["sun"]["elevation_deg"])

    # ---------------- 生成 ----------------
    def _place_craters(self, rng, cc, pad):
        out = []
        hx, hy = self.hls["x"], self.hls["y"]
        sx, sy = self.rover_start["x"], self.rover_start["y"]
        rmin, rmax = float(cc["radius_min"]), float(cc["radius_max"])
        half = self.size / 2.0
        attempts = 0
        while len(out) < int(cc["count"]) and attempts < 5000:
            attempts += 1
            R = math.exp(rng.uniform(math.log(rmin), math.log(rmax)))
            m = half - 2.0 * R - 2.0
            if m <= 0:
                continue
            cx, cy = rng.uniform(-m, m, 2)
            if math.hypot(cx - hx, cy - hy) < pad + R + 5:
                continue
            if math.hypot(cx - sx, cy - sy) < R + 10:
                continue
            d = float(cc["depth_ratio"]) * R * rng.uniform(0.6, 1.0)
            out.append((float(cx), float(cy), float(R), float(d)))
        return out

    @staticmethod
    def _crater_profile(X, Y, cx, cy, R, d):
        r = np.hypot(X - cx, Y - cy) / R
        bowl = np.where(r < 1.0, -d * (1.0 - r * r), 0.0)
        rim = 0.25 * d * np.exp(-(((r - 1.0) / 0.35) ** 2))
        return bowl + rim

    def _place_rocks(self, rng, rc):
        rmin, rmax = float(rc["radius_min"]), float(rc["radius_max"])
        n = int(rc["count"])
        half = self.size / 2.0 - 3.0
        # べき乗則 (小さい岩が多い)
        r = rmin * (1.0 - rng.random(n)) ** (-1.0 / 1.8)
        r = np.minimum(r, rmax)
        xy = rng.uniform(-half, half, (n, 2))
        hx, hy = self.hls["x"], self.hls["y"]
        sx, sy = self.rover_start["x"], self.rover_start["y"]
        ok = (np.hypot(xy[:, 0] - hx, xy[:, 1] - hy) > self.hls["radius"] + 2.0 + r) & \
             (np.hypot(xy[:, 0] - sx, xy[:, 1] - sy) > 6.0 + r)
        xy, r = xy[ok], r[ok]
        z = self.height(xy[:, 0], xy[:, 1]) + 0.3 * r  # 一部埋まった球
        return np.column_stack([xy, z, r]).astype(np.float64)

    # ---------------- 問い合わせ ----------------
    def height(self, x, y):
        """双線形補間による地面高さ (ベクトル化)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        fx = np.clip((x - self.origin) / self.res, 0.0, self.n - 1.001)
        fy = np.clip((y - self.origin) / self.res, 0.0, self.n - 1.001)
        ix = fx.astype(np.int64)
        iy = fy.astype(np.int64)
        tx = fx - ix
        ty = fy - iy
        H = self.heights
        h00 = H[iy, ix]
        h10 = H[iy, ix + 1]
        h01 = H[iy + 1, ix]
        h11 = H[iy + 1, ix + 1]
        return (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty

    # ---------------- レイキャスト ----------------
    @staticmethod
    def _steps(tmin: float, tmax: float) -> list[float]:
        ts, t = [], tmin
        while t < tmax:
            ts.append(t)
            t += 0.2 + 0.012 * t
        ts.append(tmax)
        return ts

    def raycast(self, o: np.ndarray, d: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
        """原点 o (3,), 方向 d (N,3, 単位ベクトル) の最初の交点までの距離 (なければ inf)."""
        n = len(d)
        t_best = np.full(n, np.inf)
        self._ray_terrain(o, d, tmin, tmax, t_best)
        self._ray_rocks(o, d, tmin, tmax, t_best)
        self._ray_hls(o, d, tmin, tmax, t_best)
        return t_best

    def _ray_terrain(self, o, d, tmin, tmax, t_best):
        idx = np.nonzero(d[:, 2] < 0.5)[0]  # 大きく上向きの光線は地面に当たらない
        prev = tmin
        first = True
        for t in self._steps(tmin, tmax):
            if idx.size == 0:
                break
            dd = d[idx]
            px = o[0] + dd[:, 0] * t
            py = o[1] + dd[:, 1] * t
            pz = o[2] + dd[:, 2] * t
            hit = pz - self.height(px, py) <= 0.0
            if hit.any():
                hi = idx[hit]
                dh = d[hi]
                lo = np.full(len(hi), prev)
                up = np.full(len(hi), t)
                for _ in range(8):
                    mid = 0.5 * (lo + up)
                    ab = o[2] + dh[:, 2] * mid - self.height(o[0] + dh[:, 0] * mid, o[1] + dh[:, 1] * mid)
                    pos = ab > 0
                    lo = np.where(pos, mid, lo)
                    up = np.where(pos, up, mid)
                t_best[hi] = np.minimum(t_best[hi], 0.5 * (lo + up))
                idx = idx[~hit]
            prev = t
            first = False

    def _ray_rocks(self, o, d, tmin, tmax, t_best):
        if len(self.rocks) == 0:
            return
        rel = self.rocks[:, :3] - o
        dist = np.linalg.norm(rel, axis=1)
        near = dist < tmax + self.rocks[:, 3]
        if not near.any():
            return
        # 光線束の代表方向と広がりで粗く棄却
        axis = d.mean(axis=0)
        axis /= np.linalg.norm(axis) + 1e-12
        spread = float(np.arccos(np.clip((d @ axis).min(), -1, 1)))
        for k in np.nonzero(near)[0]:
            r = self.rocks[k, 3]
            dk = max(dist[k], 1e-6)
            ang = math.acos(np.clip(rel[k] @ axis / dk, -1, 1))
            if ang > spread + math.asin(min(1.0, r / dk)) + 0.02:
                continue
            oc = o - self.rocks[k, :3]
            b = d @ oc
            disc = b * b - (oc @ oc - r * r)
            m = disc > 0
            if not m.any():
                continue
            t = np.full(len(d), np.inf)
            t[m] = -b[m] - np.sqrt(disc[m])
            t[(t < tmin) | (t > tmax)] = np.inf
            np.minimum(t_best, t, out=t_best)

    def _ray_hls(self, o, d, tmin, tmax, t_best):
        h = self.hls
        c = np.array([h["x"], h["y"]])
        R = float(h["radius"])
        z0, z1 = self.hls_z0, self.hls_z0 + float(h["height"])
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        oc = o[:2] - c
        b = d[:, 0] * oc[0] + d[:, 1] * oc[1]
        cc = oc @ oc - R * R
        disc = b * b - a * cc
        m = (a > 1e-9) & (disc > 0)
        if not m.any():
            return
        t = np.full(len(d), np.inf)
        t[m] = (-b[m] - np.sqrt(disc[m])) / a[m]
        z = o[2] + d[:, 2] * t
        bad = (t < tmin) | (t > tmax) | (z < z0) | (z > z1) | ~np.isfinite(t)
        t[bad] = np.inf
        np.minimum(t_best, t, out=t_best)

    # ---------------- 配信用 ----------------
    def describe(self) -> dict:
        return {
            "size": self.size, "res": self.res, "n": self.n, "origin": self.origin,
            "sun_azimuth": self.sun_azimuth, "sun_elevation": self.sun_elevation,
            "hls_z0": self.hls_z0,
        }
