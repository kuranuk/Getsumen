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

        # --- クレータ (サイズ頻度分布に従う。重なり合って飽和した極域ハイランドの外観) ---
        self.craters = self._place_craters(rng, wc["craters"], pad)
        for cx, cy, R, d in self.craters:
            self._add_crater(h, X, Y, cx, cy, R, d)
        h *= mask  # HLSパッドと地形端は平坦化
        self.heights = h.astype(np.float32)  # [iy, ix]

        # --- 岩 ---
        self.rocks = self._place_rocks(rng, wc["rocks"])  # (m,4): x,y,z,r
        self.hls_z0 = float(self.height(hc["x"], hc["y"]))
        self.ambient = float(wc["sun"].get("ambient", 0.03))
        self.set_sun(wc["sun"]["azimuth_deg"], wc["sun"]["elevation_deg"])

    # ---------------- 生成 ----------------
    def _place_craters(self, rng, cc, pad):
        """累積サイズ頻度 N(>D) = 1.54 * sat * D^-2 [個/m^2] (傾き -2: 平衡状態) に従って配置."""
        dmin, dmax = float(cc["diameter_min"]), float(cc["diameter_max"])
        sat = float(cc["saturation"])
        area = self.size ** 2
        n = int(rng.poisson(1.54 * sat * area * (dmin ** -2 - dmax ** -2)))
        u = rng.random(n)
        D = (dmin ** -2 - u * (dmin ** -2 - dmax ** -2)) ** -0.5  # 逆関数サンプリング
        hx, hy = self.hls["x"], self.hls["y"]
        sx, sy = self.rover_start["x"], self.rover_start["y"]
        half = self.size / 2.0
        out = []
        for Di in sorted(D, reverse=True):
            R = Di / 2.0
            cx, cy = rng.uniform(-half, half, 2)
            if math.hypot(cx - hx, cy - hy) < pad + R + 3:
                continue
            if math.hypot(cx - sx, cy - sy) < R + 8:
                continue
            # 劣化度で 深さ/直径 がばらつく (新鮮ほど深い). 小さいクレータほど相対的に浅く
            ratio = float(cc["depth_ratio"]) * rng.uniform(0.4, 1.3) * min(1.0, 0.6 + 0.1 * Di)
            out.append((float(cx), float(cy), float(R), float(ratio * Di)))
        return out

    def _add_crater(self, h, X, Y, cx, cy, R, d):
        """窓領域だけに放物面の窪み + なだらかなリム/噴出物を加える."""
        res, org, n = self.res, self.origin, self.n
        w = 3.0 * R
        ix0 = max(0, int((cx - w - org) / res)); ix1 = min(n, int((cx + w - org) / res) + 2)
        iy0 = max(0, int((cy - w - org) / res)); iy1 = min(n, int((cy + w - org) / res) + 2)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        r = np.hypot(X[iy0:iy1, ix0:ix1] - cx, Y[iy0:iy1, ix0:ix1] - cy) / R
        bowl = np.where(r < 1.0, -d * (1.0 - r * r), 0.0)
        rim = 0.22 * d * np.exp(-(((r - 1.0) / 0.30) ** 2))
        ejecta = 0.06 * d * np.clip(r, 1.0, None) ** -3.0 * (r >= 1.0)
        h[iy0:iy1, ix0:ix1] += bowl + rim + ejecta

    def _place_rocks(self, rng, rc):
        """岩の大きさ分布 (Golombek): 累積面積 F(D)=k exp(-qD), q=1.79+0.152/k.
        個数密度 n(D) = -dF/dD / (pi D^2/4).  新しいクレータの縁に集中させる."""
        k = float(rc["abundance"])
        dmin, dmax = float(rc["min_diameter"]), float(rc["max_diameter"])
        q = 1.79 + 0.152 / max(k, 1e-4)
        Dg = np.linspace(dmin, dmax, 400)
        nd = k * q * np.exp(-q * Dg) / (math.pi * Dg ** 2 / 4.0)
        cdf = np.concatenate([[0], np.cumsum(0.5 * (nd[1:] + nd[:-1]) * np.diff(Dg))])
        total = cdf[-1] * self.size ** 2
        n = int(min(rc.get("max_count", 2500), rng.poisson(total)))
        D = np.interp(rng.random(n) * cdf[-1], cdf, Dg)
        r = D / 2.0

        half = self.size / 2.0 - 3.0
        xy = rng.uniform(-half, half, (n, 2))
        # 一部をクレータ縁へ
        big = [c for c in self.craters if c[2] >= 4.0]
        nrim = int(n * float(rc.get("rim_fraction", 0.0))) if big else 0
        if nrim:
            w = np.array([c[2] for c in big]); w = w / w.sum()
            ci = rng.choice(len(big), nrim, p=w)
            ang = rng.uniform(0, 2 * np.pi, nrim)
            rr = np.array([big[i][2] for i in ci]) * (1.05 + rng.normal(0, 0.18, nrim))
            xy[:nrim, 0] = np.array([big[i][0] for i in ci]) + rr * np.cos(ang)
            xy[:nrim, 1] = np.array([big[i][1] for i in ci]) + rr * np.sin(ang)
        hx, hy = self.hls["x"], self.hls["y"]
        sx, sy = self.rover_start["x"], self.rover_start["y"]
        ok = (np.abs(xy[:, 0]) < half) & (np.abs(xy[:, 1]) < half) & \
             (np.hypot(xy[:, 0] - hx, xy[:, 1] - hy) > self.hls["radius"] + 2.0 + r) & \
             (np.hypot(xy[:, 0] - sx, xy[:, 1] - sy) > 6.0 + r)
        xy, r = xy[ok], r[ok]
        z = self.height(xy[:, 0], xy[:, 1]) + 0.3 * r  # 一部埋まった球
        return np.column_stack([xy, z, r]).astype(np.float64)

    # ---------------- 太陽 / 影の焼き込み ----------------
    def set_sun(self, azimuth_deg: float, elevation_deg: float):
        self.sun_azimuth = math.radians(azimuth_deg % 360.0)
        self.sun_elevation = math.radians(max(0.0, min(10.0, elevation_deg)))

    def bake_shade(self, azimuth: float | None = None, elevation: float | None = None) -> np.ndarray:
        """地形自身が落とす影 (ホライズン判定). 戻り値 uint8 [iy,ix] (0=影, 255=日照).
        太陽の視直径 (0.53deg) 分の半影を含む. 岩/HLS/ローバーの影は描画側のシャドウマップで扱う."""
        az = self.sun_azimuth if azimuth is None else azimuth
        el = self.sun_elevation if elevation is None else elevation
        dx, dy = math.cos(az), math.sin(az)
        tan_el = math.tan(el)
        H = self.heights.astype(np.float32)
        n, res, org = self.n, self.res, self.origin
        X = (self._X.astype(np.float32)).ravel(); Y = (self._Y.astype(np.float32)).ravel()
        h0 = H.ravel()
        max_slope = np.full(h0.shape, -1e3, np.float32)
        s, ratio = 1.0, 1.03
        Hf = H.ravel()
        while s < self.size:
            fx = np.clip((X + dx * s - org) / res, 0, n - 1.001)
            fy = np.clip((Y + dy * s - org) / res, 0, n - 1.001)
            ix = fx.astype(np.int32); iy = fy.astype(np.int32)
            tx = fx - ix; ty = fy - iy
            i00 = iy * n + ix
            hh = (Hf[i00] * (1 - tx) + Hf[i00 + 1] * tx) * (1 - ty) + (Hf[i00 + n] * (1 - tx) + Hf[i00 + n + 1] * tx) * ty
            np.maximum(max_slope, (hh - h0) / s, out=max_slope)
            s = s * ratio + 0.15
        w = math.tan(math.radians(0.27))  # 太陽半径
        t = np.clip((tan_el - max_slope + w) / (2 * w), 0.0, 1.0)
        return np.rint(255 * t * t * (3 - 2 * t)).astype(np.uint8).reshape(n, n)

    def sun_dir(self) -> np.ndarray:
        e, a = self.sun_elevation, self.sun_azimuth
        return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])

    def bake_rock_shade(self, cell: float = 0.25) -> np.ndarray:
        """岩が地面に落とす影 (球 -> 地面の楕円). 戻り値 uint8 [m,m] (0=影), 格子間隔 cell [m].
        球を光線方向に射影した断面は半径 r の円で、地面では太陽方位方向に 1/sin(仰角) 倍に伸びた楕円になる."""
        m = int(round(self.size / cell)) + 1
        vis = np.ones((m, m), np.float32)
        az, el = self.sun_azimuth, self.sun_elevation
        L = self.sun_dir()
        ca, sa = math.cos(az), math.sin(az)
        sin_e = max(math.sin(el), 1e-3)
        org = self.origin
        for cx, cy, cz, r in self.rocks:
            zg = cz - 0.3 * r  # 岩の接地高さ
            sh = (cz - zg) / max(L[2], 1e-3)
            gx, gy = cx - L[0] * sh, cy - L[1] * sh  # 影の楕円の中心
            a_maj = min(r / sin_e, 60.0)
            ext = a_maj + r
            ix0 = max(0, int((gx - ext - org) / cell)); ix1 = min(m, int((gx + ext - org) / cell) + 2)
            iy0 = max(0, int((gy - ext - org) / cell)); iy1 = min(m, int((gy + ext - org) / cell) + 2)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            xs = org + np.arange(ix0, ix1) * cell - gx
            ys = org + np.arange(iy0, iy1) * cell - gy
            XX, YY = np.meshgrid(xs, ys)
            u = XX * ca + YY * sa       # 太陽方位方向
            v = -XX * sa + YY * ca      # 直交方向
            q = (u / a_maj) ** 2 + (v / r) ** 2
            val = np.clip((q - 0.8) / 0.4, 0.0, 1.0)  # 半影 (縁をなだらかに)
            win = vis[iy0:iy1, ix0:ix1]
            np.minimum(win, val.astype(np.float32), out=win)
        return np.rint(vis * 255).astype(np.uint8)

    def terrain_occluded(self, P: np.ndarray, L: np.ndarray, smax: float = 320.0) -> np.ndarray:
        """点 P (N,3) から太陽方向 L へ伸ばした光線が地形に遮られるか (地形が点より高ければ遮蔽)."""
        n = len(P)
        occ = np.zeros(n, bool)
        idx = np.arange(n)
        px, py, pz = P[:, 0].copy(), P[:, 1].copy(), P[:, 2].copy()
        zmax = float(self.heights.max()) + 0.05
        half = self.size / 2.0
        s = 0.3
        while s < smax and idx.size:
            x = px + L[0] * s
            y = py + L[1] * s
            z = pz + L[2] * s
            hit = self.height(x, y) > z + 0.02
            occ[idx[hit]] = True
            keep = ~hit & (z < zmax) & (np.abs(x) < half) & (np.abs(y) < half)
            idx, px, py, pz = idx[keep], px[keep], py[keep], pz[keep]
            s = s * 1.03 + 0.15
        return occ

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
