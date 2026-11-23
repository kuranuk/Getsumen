"""カメラ画像のレイトレーシング (numpy).

各画素から実際に光線を飛ばして 地形(高さマップ) / 岩(球) / HLS(円柱) との交点を求め、
太陽方向へ影レイを飛ばして日照を判定し、月面の反射(Lommel-Seeliger)で放射輝度[相対値]を計算する。
出力は線形(HDR)。露出/ノイズ/トーンカーブなどのセンサ処理はブラウザ側 (sensorPipeline)。

放射輝度の単位: 垂直入射の太陽光を受ける反射率1の拡散面 = 1.0 (太陽の放射照度 E=pi).
  太陽円盤 = E / 立体角(6.7e-5 sr) ~ 4.7e4
  恒星   : 放射照度 E = pi * 10^(-0.4 (m + 26.74)) を画素の立体角で割る  (m: 見かけの等級)
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import mount_position, mount_rotation, rot_zyx
from .world import World

SUN_COLOR = np.array([1.0, 0.945, 0.87], np.float32)
SUN_ANG_RADIUS = math.radians(0.265)
SUN_OMEGA = math.pi * SUN_ANG_RADIUS ** 2
SUN_RADIANCE = (math.pi / SUN_OMEGA) * SUN_COLOR
AMBIENT_COLOR = np.array([0.40, 0.42, 0.46], np.float32)


def _hex_to_linear(c) -> np.ndarray:
    if isinstance(c, str) and c.startswith("#") and len(c) == 7:
        v = np.array([int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5)], np.float32)
        return v ** 2.2
    return np.array([0.7, 0.7, 0.7], np.float32)


def _fractal_tile(rng, n: int, beta: float) -> np.ndarray:
    """周期的な 1/f^beta ノイズ (標準偏差1)."""
    sp = np.fft.rfft2(rng.standard_normal((n, n)))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.rfftfreq(n)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = 1.0
    out = np.fft.irfft2(sp * f ** (-beta / 2.0) * (f > 0), s=(n, n))
    return (out / (out.std() + 1e-12)).astype(np.float32)


class _Tile:
    """周期ノイズ + その勾配 (法線の微小起伏と反射率のムラに使う)."""

    def __init__(self, rng, n: int, meters_per_texel: float, beta: float):
        self.n = n
        self.mpt = meters_per_texel
        v = _fractal_tile(rng, n, beta)
        gy, gx = np.gradient(v)
        self.v = v
        sd = float(np.hypot(gx, gy).std()) + 1e-9
        self.gx, self.gy = (gx / sd).astype(np.float32), (gy / sd).astype(np.float32)

    def sample(self, arr, x, y):
        fx = x / self.mpt
        fy = y / self.mpt
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        tx, ty = (fx - ix).astype(np.float32), (fy - iy).astype(np.float32)
        ix0, iy0 = ix % self.n, iy % self.n
        ix1, iy1 = (ix + 1) % self.n, (iy + 1) % self.n
        return ((arr[iy0, ix0] * (1 - tx) + arr[iy0, ix1] * tx) * (1 - ty)
                + (arr[iy1, ix0] * (1 - tx) + arr[iy1, ix1] * tx) * ty)


class StarField:
    """合成の恒星カタログ (実在の星の位置ではない). 個数は N(<m) = 4800 * 2.25^(m-6) に従う."""

    def __init__(self, limit_mag: float = 7.0, seed: int = 11):
        rng = np.random.default_rng(seed)
        m_lo = -1.5
        n_tot = int(4800 * 2.25 ** (limit_mag - 6))
        # 累積 N(<m) の逆関数サンプリング
        c_lo = 4800 * 2.25 ** (m_lo - 6)
        u = rng.random(n_tot)
        mag = 6 + np.log((c_lo + u * (n_tot - c_lo)) / 4800) / math.log(2.25)
        z = rng.uniform(-1, 1, n_tot)
        a = rng.uniform(0, 2 * math.pi, n_tot)
        s = np.sqrt(1 - z * z)
        self.dirs = np.stack([s * np.cos(a), s * np.sin(a), z], 1)
        self.flux = (math.pi * 10 ** (-0.4 * (mag + 26.74))).astype(np.float32)  # 放射照度
        temp = rng.uniform(0.0, 1.0, n_tot)
        self.tint = np.stack([0.85 + 0.25 * temp, np.full(n_tot, 1.0), 1.1 - 0.25 * temp], 1).astype(np.float32)
        self.mag = mag


class CameraModel:
    def __init__(self, cfg: dict, index: int):
        self.index = index
        self.cfg = cfg
        self.name = cfg["name"]
        self.w, self.h = int(cfg["width"]), int(cfg["height"])
        self.ss = max(1, int(cfg.get("supersample", 1)))
        self.W, self.Hh = self.w * self.ss, self.h * self.ss
        self.th = math.tan(math.radians(cfg["fov_deg"]) / 2.0)
        self.tv = self.th * self.h / self.w
        u = (np.arange(self.W) + 0.5) / self.W * 2 - 1
        v = 1 - (np.arange(self.Hh) + 0.5) / self.Hh * 2
        U, V = np.meshgrid(u, v)
        d = np.stack([np.ones_like(U), -U * self.th, V * self.tv], -1).reshape(-1, 3)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.m_pos = mount_position(cfg["mount"])
        self.m_rot = mount_rotation(cfg["mount"])
        self.dirs_body = d @ self.m_rot.T
        self.pix_omega = (2 * self.th / self.W) * (2 * self.tv / self.Hh)  # 画素の立体角 [sr]
        self.stars = bool(cfg.get("stars", True))
        self.psf_sigma = float(cfg.get("star_psf_px", 0.8)) * self.ss


class RayTracer:
    def __init__(self, cfg: dict, world: World):
        self.world = world
        rc = cfg["rover"]
        self.rover = rc
        self.box_lo = np.array([-rc["length"] / 2, -rc["width"] / 2, rc["clearance"] - rc["wheel_radius"]])
        self.box_hi = np.array([rc["length"] / 2, rc["width"] / 2,
                                rc["clearance"] - rc["wheel_radius"] + rc["height"]])
        self.hls_albedo = _hex_to_linear(world.hls.get("color", "#d8dade"))
        rng = np.random.default_rng(world.seed + 101)
        self.fine = _Tile(rng, 512, 0.06, 2.0)     # 周期 ~31 m, 6 cm/texel (細かい粒状感)
        self.coarse = _Tile(rng, 512, 0.5, 2.4)    # 周期 256 m, 50 cm/texel
        self.rock_albedo = (0.20 * (0.75 + 0.5 * rng.random(len(world.rocks)))).astype(np.float32)
        self.star_cache: dict[float, StarField] = {}
        self.cams = {c["name"]: CameraModel(c, i) for i, c in enumerate(cfg["sensors"]["cameras"])
                     if c.get("render", "raytrace") == "raytrace"}

    # ------------------------------------------------------------ 交差判定
    def _march_terrain(self, o, d, tmin, tmax):
        w = self.world
        N = len(d)
        out = np.full(N, np.inf)
        idx = np.nonzero(d[:, 2] < 0.5)[0]
        dx, dy, dz = d[idx, 0].copy(), d[idx, 1].copy(), d[idx, 2].copy()
        half = w.size / 2.0
        prev = tmin
        for t in w._steps(tmin, tmax):
            if idx.size == 0:
                break
            px = o[0] + dx * t
            py = o[1] + dy * t
            pz = o[2] + dz * t
            hit = pz - w.height(px, py) <= 0.0
            if hit.any():
                hi = idx[hit]
                hx, hy, hz = dx[hit], dy[hit], dz[hit]
                lo = np.full(len(hi), prev)
                up = np.full(len(hi), t)
                for _ in range(8):
                    mid = 0.5 * (lo + up)
                    above = (o[2] + hz * mid - w.height(o[0] + hx * mid, o[1] + hy * mid)) > 0
                    lo = np.where(above, mid, lo)
                    up = np.where(above, up, mid)
                out[hi] = 0.5 * (lo + up)
            keep = ~hit & (np.abs(px) < half) & (np.abs(py) < half)  # 地形外は平面(z=0)として別処理
            if not keep.all():
                idx, dx, dy, dz = idx[keep], dx[keep], dy[keep], dz[keep]
            prev = t
        return out

    def _rocks(self, o, d, basis, cam: CameraModel, t, kind, rid):
        rocks = self.world.rocks
        rel = rocks[:, :3] - o
        pc = rel @ basis  # (前, 左, 上)
        f = pc[:, 0]
        ok = f > 0.1
        px = ((-pc[:, 1] / np.maximum(f, 1e-6)) / cam.th * 0.5 + 0.5) * cam.W
        py = (0.5 - (pc[:, 2] / np.maximum(f, 1e-6)) / cam.tv * 0.5) * cam.Hh
        pr = (rocks[:, 3] / np.maximum(f, 1e-6)) / cam.th * 0.5 * cam.W * 1.8 + 2.0
        ok &= (px > -pr) & (px < cam.W + pr) & (py > -pr) & (py < cam.Hh + pr)
        for k in np.nonzero(ok)[0]:
            x0, x1 = max(0, int(px[k] - pr[k])), min(cam.W, int(px[k] + pr[k]) + 2)
            y0, y1 = max(0, int(py[k] - pr[k])), min(cam.Hh, int(py[k] + pr[k]) + 2)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = (np.arange(y0, y1)[:, None] * cam.W + np.arange(x0, x1)[None, :]).ravel()
            r = rocks[k, 3]
            oc = o - rocks[k, :3]
            b = d[sub] @ oc
            disc = b * b - (oc @ oc - r * r)
            m = disc > 0
            if not m.any():
                continue
            tt = np.where(m, -b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
            upd = (tt > 0.05) & (tt < t[sub])
            if upd.any():
                s2 = sub[upd]
                t[s2] = tt[upd]
                kind[s2] = 2
                rid[s2] = k

    def _hls(self, o, d, t, kind):
        h = self.world.hls
        c = np.array([h["x"], h["y"]])
        R = float(h["radius"])
        z0, z1 = self.world.hls_z0, self.world.hls_z0 + float(h["height"])
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        oc = o[:2] - c
        b = d[:, 0] * oc[0] + d[:, 1] * oc[1]
        cc = oc @ oc - R * R
        disc = b * b - a * cc
        m = (a > 1e-12) & (disc > 0)
        ts = np.full(len(d), np.inf)
        ts[m] = (-b[m] - np.sqrt(disc[m])) / a[m]
        z = o[2] + d[:, 2] * ts
        ts[(ts < 0.05) | (z < z0) | (z > z1)] = np.inf
        # 上面
        tc = np.full(len(d), np.inf)
        dn = d[:, 2] < -1e-9
        tc[dn] = (z1 - o[2]) / d[dn, 2]
        px, py = o[0] + d[:, 0] * tc - c[0], o[1] + d[:, 1] * tc - c[1]
        tc[(tc < 0.05) | (px * px + py * py > R * R)] = np.inf
        tt = np.minimum(ts, tc)
        upd = tt < t
        t[upd] = tt[upd]
        kind[upd] = 3

    # ------------------------------------------------------------ 影
    def _hls_shadow(self, P, L):
        h = self.world.hls
        c = np.array([h["x"], h["y"]])
        R = float(h["radius"])
        z1 = self.world.hls_z0 + float(h["height"])
        pxy = P[:, :2] - c
        a = L[0] ** 2 + L[1] ** 2
        b = pxy[:, 0] * L[0] + pxy[:, 1] * L[1]
        cc = (pxy ** 2).sum(1) - R * R
        disc = b * b - a * cc
        s = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0.0))) / max(a, 1e-12), -1.0)
        zh = P[:, 2] + L[2] * s
        return (disc > 0) & (s > 0.02) & (zh <= z1) & (zh >= self.world.hls_z0)

    def _box_shadow(self, P, L, pose, Rm):
        pos = np.array(pose[:3])
        pb = (P - pos) @ Rm            # 車体座標へ (Rm.T の転置適用)
        lb = L @ Rm
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (self.box_lo - pb) / lb
            t2 = (self.box_hi - pb) / lb
        tn = np.nanmax(np.minimum(t1, t2), axis=1)
        tf = np.nanmin(np.maximum(t1, t2), axis=1)
        return (tf > np.maximum(tn, 0.0)) & (tf > 0.0)

    def _lookup(self, grid_u8, x, y, cell):
        n = grid_u8.shape[0]
        fx = np.clip((x - self.world.origin) / cell, 0, n - 1.001)
        fy = np.clip((y - self.world.origin) / cell, 0, n - 1.001)
        ix, iy = fx.astype(np.int64), fy.astype(np.int64)
        tx, ty = fx - ix, fy - iy
        g = grid_u8
        v = ((g[iy, ix] * (1 - tx) + g[iy, ix + 1] * tx) * (1 - ty)
             + (g[iy + 1, ix] * (1 - tx) + g[iy + 1, ix + 1] * tx) * ty)
        return v / 255.0

    # ------------------------------------------------------------ レンダリング
    def render(self, name: str, pose, shade: np.ndarray, rock_shade: np.ndarray) -> np.ndarray:
        cam = self.cams[name]
        w = self.world
        L = w.sun_dir()
        Rm = rot_zyx(pose[3], pose[4], pose[5])
        o = np.array(pose[:3]) + Rm @ cam.m_pos
        d = cam.dirs_body @ Rm.T
        basis = Rm @ cam.m_rot
        N = len(d)
        t = np.full(N, np.inf)
        kind = np.zeros(N, np.int8)
        rid = np.full(N, -1, np.int32)

        # --- 一次光線: 地形(+地形外の平面) -> 岩 -> HLS ---
        tt = self._march_terrain(o, d, 0.05, 300.0)
        hit = np.isfinite(tt)
        t[hit] = tt[hit]
        kind[hit] = 1
        far = ~hit & (d[:, 2] < -1e-5)
        t[far] = -o[2] / d[far, 2]
        kind[far] = 1
        self._rocks(o, d, basis, cam, t, kind, rid)
        self._hls(o, d, t, kind)

        rgb = np.zeros((N, 3), np.float32)
        P = o + d * np.where(np.isfinite(t), t, 0.0)[:, None]
        V = -d

        # --- 空: 太陽円盤 + 恒星 ---
        sky = kind == 0
        sun_disc = sky & ((d @ L) > math.cos(SUN_ANG_RADIUS))
        rgb[sun_disc] = SUN_RADIANCE

        # --- 地形 ---
        m1 = np.nonzero(kind == 1)[0]
        if m1.size:
            p = P[m1]
            tm = t[m1]
            e = np.clip(0.025 * tm, 0.6, 8.0)  # 遠いほど広い基線で勾配を取る (法線のLOD: ちらつき防止)
            hx = (w.height(p[:, 0] + e, p[:, 1]) - w.height(p[:, 0] - e, p[:, 1])) / (2 * e)
            hy = (w.height(p[:, 0], p[:, 1] + e) - w.height(p[:, 0], p[:, 1] - e)) / (2 * e)
            wf = np.clip(1 - tm / 25.0, 0, 1) ** 2   # 近距離だけ細かい起伏を見せる
            wc = np.clip(1 - tm / 80.0, 0, 1) ** 2
            fgx, fgy = self.fine.sample(self.fine.gx, p[:, 0], p[:, 1]), self.fine.sample(self.fine.gy, p[:, 0], p[:, 1])
            cgx, cgy = self.coarse.sample(self.coarse.gx, p[:, 0], p[:, 1]), self.coarse.sample(self.coarse.gy, p[:, 0], p[:, 1])
            nx = -hx - 0.07 * wf * fgx - 0.035 * wc * cgx
            ny = -hy - 0.07 * wf * fgy - 0.035 * wc * cgy
            n = np.stack([nx, ny, np.ones_like(nx)], 1)
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            af = self.fine.sample(self.fine.v, p[:, 0], p[:, 1])
            ac = self.coarse.sample(self.coarse.v, p[:, 0], p[:, 1])
            albedo = 0.26 * np.clip(1 + 0.18 * wf * af + 0.15 * wc * ac
                                    + 0.06 * np.sin(p[:, 0] * 0.11 + 1.3) * np.sin(p[:, 1] * 0.087 + 0.4), 0.4, 1.8)
            vis = self._lookup(shade, p[:, 0], p[:, 1], w.res) * self._lookup(rock_shade, p[:, 0], p[:, 1], 0.25)
            vis = vis * ~self._hls_shadow(p, L) * ~self._box_shadow(p, L, pose, Rm)
            rgb[m1] = self._shade(albedo, n, V[m1], L, vis)

        # --- 岩 ---
        m2 = np.nonzero(kind == 2)[0]
        if m2.size:
            k = rid[m2]
            c = w.rocks[k, :3]
            n = (P[m2] - c) / w.rocks[k, 3][:, None]
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            p = P[m2] + n * 0.02
            af = self.fine.sample(self.fine.v, p[:, 0] + p[:, 2], p[:, 1] - p[:, 2])
            albedo = self.rock_albedo[k] * np.clip(1 + 0.15 * af, 0.5, 1.6)
            vis = (~w.terrain_occluded(p, L)) & ~self._hls_shadow(p, L) & ~self._box_shadow(p, L, pose, Rm)
            rgb[m2] = self._shade(albedo, n, V[m2], L, vis.astype(np.float32))

        # --- HLS ---
        m3 = np.nonzero(kind == 3)[0]
        if m3.size:
            h = w.hls
            p = P[m3]
            n = np.zeros((m3.size, 3))
            side = p[:, 2] < w.hls_z0 + float(h["height"]) - 1e-3
            n[side, 0] = p[side, 0] - h["x"]
            n[side, 1] = p[side, 1] - h["y"]
            n[~side, 2] = 1.0
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            q = p + n * 0.05
            vis = (~w.terrain_occluded(q, L)).astype(np.float32)
            rgb[m3] = self._shade(np.tile(self.hls_albedo, (m3.size, 1)), n, V[m3], L, vis)

        # --- 恒星 (空の画素だけに点広がり関数で加算) ---
        if cam.stars and sky.any():
            self._add_stars(rgb, sky, cam, basis)

        img = rgb.reshape(cam.Hh, cam.W, 3)
        if cam.ss > 1:
            img = img.reshape(cam.h, cam.ss, cam.w, cam.ss, 3).mean(axis=(1, 3))
        return img.astype(np.float16)

    def _shade(self, albedo, n, V, L, vis):
        """Lommel-Seeliger 反射: 垂直入射で Lambert と同じになるよう 2*mu0/(mu0+mu) で正規化."""
        mu0 = np.clip(n @ L, 0.0, 1.0)
        mu = np.clip((n * V).sum(1), 0.0, 1.0)
        ls = 2.0 * mu0 / (mu0 + mu + 0.03)
        albedo = np.asarray(albedo, np.float32)
        if albedo.ndim == 1:
            albedo = albedo[:, None]
        direct = (albedo * (ls * vis)[:, None]) * SUN_COLOR
        amb = albedo * (self.world.ambient * AMBIENT_COLOR)
        return (direct + amb).astype(np.float32)

    def _add_stars(self, rgb, sky, cam: CameraModel, basis):
        sf = self.star_cache.get(cam.cfg.get("star_limit_mag", 7.0))
        if sf is None:
            sf = self.star_cache[cam.cfg.get("star_limit_mag", 7.0)] = StarField(cam.cfg.get("star_limit_mag", 7.0))
        pc = sf.dirs @ basis
        f = pc[:, 0]
        ok = f > 0.05
        u = (-pc[:, 1] / np.maximum(f, 1e-6)) / cam.th
        v = (pc[:, 2] / np.maximum(f, 1e-6)) / cam.tv
        ok &= (np.abs(u) < 1.02) & (np.abs(v) < 1.02)
        if not ok.any():
            return
        px = (u[ok] * 0.5 + 0.5) * cam.W - 0.5
        py = (0.5 - v[ok] * 0.5) * cam.Hh - 0.5
        rad = sf.flux[ok] / cam.pix_omega  # 画素に集約した場合の放射輝度 (点広がり関数で配分)
        col = sf.tint[ok]
        sig = cam.psf_sigma
        flat = rgb.reshape(-1, 3)
        ix0, iy0 = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
        offs = [(dx, dy) for dy in (-1, 0, 1, 2) for dx in (-1, 0, 1, 2)]
        wts = [np.exp(-((ix0 + dx - px) ** 2 + (iy0 + dy - py) ** 2) / (2 * sig * sig)) for dx, dy in offs]
        total = np.sum(wts, axis=0) + 1e-12  # 窓内で総フラックスが保存されるよう正規化
        for (dx, dy), wgt in zip(offs, wts):
            xx, yy = ix0 + dx, iy0 + dy
            inb = (xx >= 0) & (xx < cam.W) & (yy >= 0) & (yy < cam.Hh)
            lin = np.where(inb, yy * cam.W + xx, 0)
            good = inb & sky[lin]
            np.add.at(flat, lin[good], ((rad * wgt / total)[good])[:, None] * col[good])
