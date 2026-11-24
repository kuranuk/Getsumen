"""カメラ画像のレイトレーシング (numpy).

各画素から光線を飛ばして 地形(高さマップ) / 岩(球) / HLS(円柱) との交点を求め、
太陽方向へ影レイを飛ばして日照を判定し、月面の反射(Lommel-Seeliger)で放射輝度[相対値]を計算する。
出力は線形(HDR)。露出/ノイズ/トーンカーブなどのセンサ処理はブラウザ側 (web/cameras.js)。

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
SUN_RADIANCE = (1.0 / SUN_ANG_RADIUS ** 2) * SUN_COLOR  # E / 立体角 = pi / (pi r^2)
AMBIENT_COLOR = np.array([0.40, 0.42, 0.46], np.float32)
STAR_PSF_SIGMA = 0.8  # 恒星の点広がり関数 [px]
MAX_RANGE = 300.0     # 一次光線の地形マーチング距離 [m] (その先は z=0 の平面)


def _hex_to_linear(color: str) -> np.ndarray:
    """'#rrggbb' -> 線形RGB (ガンマ2.2)."""
    return np.array([int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)], np.float32) ** 2.2


class _NoiseTile:
    """周期ノイズ + その勾配。法線の微小起伏と反射率のムラに使う (1/f^beta 型, 標準偏差1)."""

    def __init__(self, rng, n: int, meters_per_texel: float, beta: float):
        self.n, self.mpt = n, meters_per_texel
        f = np.hypot(np.fft.rfftfreq(n)[None, :], np.fft.fftfreq(n)[:, None])
        f[0, 0] = np.inf  # 平均成分を除く
        v = np.fft.irfft2(np.fft.rfft2(rng.standard_normal((n, n))) * f ** (-beta / 2.0), s=(n, n))
        self.v = (v / v.std()).astype(np.float32)
        gy, gx = np.gradient(self.v)
        sd = float(np.hypot(gx, gy).std())
        self.gx, self.gy = (gx / sd).astype(np.float32), (gy / sd).astype(np.float32)

    def sample(self, arr, x, y):
        """双線形補間 (周期境界)."""
        fx, fy = x / self.mpt, y / self.mpt
        ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        tx, ty = (fx - ix).astype(np.float32), (fy - iy).astype(np.float32)
        x0, y0, x1, y1 = ix % self.n, iy % self.n, (ix + 1) % self.n, (iy + 1) % self.n
        return ((arr[y0, x0] * (1 - tx) + arr[y0, x1] * tx) * (1 - ty)
                + (arr[y1, x0] * (1 - tx) + arr[y1, x1] * tx) * ty)


class StarField:
    """合成の恒星カタログ (実在の星の位置ではない). 個数は N(<m) = 4800 * 2.25^(m-6), m<7 に従う."""

    def __init__(self, mag_limit: float = 7.0, seed: int = 11):
        rng = np.random.default_rng(seed)
        n_tot = int(4800 * 2.25 ** (mag_limit - 6))
        n_min = 4800 * 2.25 ** (-1.5 - 6)  # 最も明るい星を -1.5 等で打ち切る
        rank = n_min + rng.random(n_tot) * (n_tot - n_min)  # 明るい順の順位 -> 等級 (累積分布の逆関数)
        mag = 6 + np.log(rank / 4800) / math.log(2.25)
        z = rng.uniform(-1, 1, n_tot)
        a = rng.uniform(0, 2 * math.pi, n_tot)
        s = np.sqrt(1 - z * z)
        self.dirs = np.stack([s * np.cos(a), s * np.sin(a), z], 1)
        self.flux = (math.pi * 10 ** (-0.4 * (mag + 26.74))).astype(np.float32)  # 放射照度
        temp = rng.random(n_tot)
        self.tint = np.stack([0.85 + 0.25 * temp, np.ones(n_tot), 1.1 - 0.25 * temp], 1).astype(np.float32)


class CameraModel:
    """ピンホールカメラ: 画素ごとの視線方向 (車体座標) と取付姿勢."""

    def __init__(self, cfg: dict, index: int):
        self.index, self.cfg, self.name = index, cfg, cfg["name"]
        self.w, self.h = int(cfg["width"]), int(cfg["height"])
        self.th = math.tan(math.radians(cfg["fov_deg"]) / 2.0)  # 水平画角の半分の tan
        self.tv = self.th * self.h / self.w
        u = (np.arange(self.w) + 0.5) / self.w * 2 - 1   # 右が正
        v = 1 - (np.arange(self.h) + 0.5) / self.h * 2   # 上が正
        U, V = np.meshgrid(u, v)
        d = np.stack([np.ones_like(U), -U * self.th, V * self.tv], -1).reshape(-1, 3)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.m_pos = mount_position(cfg["mount"])
        self.m_rot = mount_rotation(cfg["mount"])
        self.dirs_body = d @ self.m_rot.T
        self.pix_omega = (2 * self.th / self.w) * (2 * self.tv / self.h)  # 画素の立体角 [sr]


class RayTracer:
    def __init__(self, cfg: dict, world: World):
        self.world = world
        rc = cfg["rover"]
        z0 = rc["clearance"] - rc["wheel_radius"]  # 箱の下面 (車体座標)
        self.box_lo = np.array([-rc["length"] / 2, -rc["width"] / 2, z0])
        self.box_hi = np.array([rc["length"] / 2, rc["width"] / 2, z0 + rc["height"]])
        self.hls_albedo = _hex_to_linear(world.hls["color"])
        rng = np.random.default_rng(world.seed + 101)
        self.fine = _NoiseTile(rng, 512, 0.06, 2.0)   # 6 cm/texel: 細かい粒状感
        self.coarse = _NoiseTile(rng, 512, 0.5, 2.4)  # 50 cm/texel
        self.rock_albedo = (0.20 * (0.75 + 0.5 * rng.random(len(world.rocks)))).astype(np.float32)
        self.stars = StarField()
        self.cams = {c["name"]: CameraModel(c, i) for i, c in enumerate(cfg["sensors"]["cameras"])}

    # ------------------------------------------------------------ レンダリング
    def render(self, name: str, pose, shade: np.ndarray, rock_shade: np.ndarray) -> np.ndarray:
        """pose=(x,y,z,yaw,pitch,roll). shade/rock_shade: World.bake_shade / bake_rock_shade の結果.
        戻り値: (h, w, 3) float16 の線形RGB (HDR)."""
        cam, w = self.cams[name], self.world
        L = w.sun_dir()
        Rm = rot_zyx(*pose[3:])
        o = np.array(pose[:3]) + Rm @ cam.m_pos
        d = cam.dirs_body @ Rm.T
        basis = Rm @ cam.m_rot  # 列 = カメラの (前, 左, 上) の世界座標

        # --- 一次光線: 地形・平面・HLS は World、岩はカメラ用の画素窓方式 ---
        t, kind, rid = w.raycast(o, d, 0.05, MAX_RANGE, rocks=False)
        self._rocks(o, d, basis, cam, t, kind, rid)
        P = o + d * np.where(np.isfinite(t), t, 0.0)[:, None]

        rgb = np.zeros((len(d), 3), np.float32)
        sky = kind == 0
        rgb[sky & (d @ L > math.cos(SUN_ANG_RADIUS))] = SUN_RADIANCE  # 太陽円盤
        if cam.cfg["stars"] and sky.any():
            self._add_stars(rgb, sky, cam, basis)

        # --- 面ごとに 法線・反射率・日照 を求めて着色 ---
        for k in (1, 2, 3):
            m = np.nonzero(kind == k)[0]
            if m.size:
                rgb[m] = self._surface(k, P[m], t[m], rid[m], -d[m], L, shade, rock_shade, pose, Rm)
        return rgb.reshape(cam.h, cam.w, 3).astype(np.float16)

    def _surface(self, kind, P, t, rid, V, L, shade, rock_shade, pose, Rm):
        """kind 1=地形 2=岩 3=HLS の交点 P での放射輝度."""
        w = self.world
        x, y = P[:, 0], P[:, 1]
        if kind == 1:
            n, albedo = self._terrain_normal_albedo(P, t)
            vis = self._lookup(shade, x, y, w.res) * self._lookup(rock_shade, x, y, 0.25)  # 地形/岩の影
            vis = vis * ~self._hls_shadow(P, L) * ~self._box_shadow(P, L, pose, Rm)
        elif kind == 2:
            n = w.surface_normals(P, np.full(len(P), 2), rid)
            P = P + n * 0.02
            albedo = self.rock_albedo[rid] * np.clip(1 + 0.15 * self.fine.sample(self.fine.v, x + P[:, 2], y - P[:, 2]), 0.5, 1.6)
            vis = ~w.terrain_occluded(P, L) & ~self._hls_shadow(P, L) & ~self._box_shadow(P, L, pose, Rm)
        else:
            n = w.surface_normals(P, np.full(len(P), 3), rid)
            albedo = np.tile(self.hls_albedo, (len(P), 1))
            vis = ~w.terrain_occluded(P + n * 0.05, L)
        return self._shade(albedo, n, V, L, vis.astype(np.float32), lambert=(kind == 3))

    def _terrain_normal_albedo(self, P, t):
        """地形の法線 (高さマップの勾配 + 微小起伏) と反射率."""
        x, y = P[:, 0], P[:, 1]
        e = np.clip(0.025 * t, 0.6, 8.0)  # 遠いほど広い基線で勾配を取る (法線のLOD: ちらつき防止)
        hx = (self.world.height(x + e, y) - self.world.height(x - e, y)) / (2 * e)
        hy = (self.world.height(x, y + e) - self.world.height(x, y - e)) / (2 * e)
        wf = np.clip(1 - t / 25.0, 0, 1) ** 2   # 近距離だけ細かい起伏を見せる
        wc = np.clip(1 - t / 80.0, 0, 1) ** 2
        fine, coarse = self.fine, self.coarse
        nx = -hx - 0.07 * wf * fine.sample(fine.gx, x, y) - 0.035 * wc * coarse.sample(coarse.gx, x, y)
        ny = -hy - 0.07 * wf * fine.sample(fine.gy, x, y) - 0.035 * wc * coarse.sample(coarse.gy, x, y)
        n = np.stack([nx, ny, np.ones_like(nx)], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        albedo = 0.26 * np.clip(1 + 0.18 * wf * fine.sample(fine.v, x, y) + 0.15 * wc * coarse.sample(coarse.v, x, y)
                                + 0.06 * np.sin(x * 0.11 + 1.3) * np.sin(y * 0.087 + 0.4), 0.4, 1.8)
        return n, albedo

    def _shade(self, albedo, n, V, L, vis, lambert=False):
        """Lommel-Seeliger 反射 (レゴリス): 垂直入射で Lambert と同じになるよう 2*mu0/(mu0+mu) で正規化.
        lambert=True は人工物(塗装面)用。太陽と反対側は直接光ゼロ (反射光は ambient のみ)."""
        mu0 = np.clip(n @ L, 0.0, 1.0)
        mu = np.clip((n * V).sum(1), 0.0, 1.0)
        ls = mu0 if lambert else 2.0 * mu0 / (mu0 + mu + 0.03)
        albedo = albedo[:, None] if albedo.ndim == 1 else albedo
        return (albedo * (ls * vis)[:, None] * SUN_COLOR + albedo * self.world.ambient * AMBIENT_COLOR).astype(np.float32)

    # ------------------------------------------------------------ 岩 (一次光線)
    def _rocks(self, o, d, basis, cam: CameraModel, t, kind, rid):
        """各岩が画面に占める矩形の画素だけを球と交差判定する (全画素x全岩より大幅に速い)."""
        rocks = self.world.rocks
        pc = (rocks[:, :3] - o) @ basis  # (前, 左, 上)
        f = np.maximum(pc[:, 0], 1e-6)
        px = ((-pc[:, 1] / f) / cam.th * 0.5 + 0.5) * cam.w
        py = (0.5 - (pc[:, 2] / f) / cam.tv * 0.5) * cam.h
        pr = (rocks[:, 3] / f) / cam.th * 0.5 * cam.w * 1.8 + 2.0  # 画面上の半径 [px] (余裕つき)
        visible = (pc[:, 0] > 0.1) & (px > -pr) & (px < cam.w + pr) & (py > -pr) & (py < cam.h + pr)
        for k in np.nonzero(visible)[0]:
            x0, x1 = max(0, int(px[k] - pr[k])), min(cam.w, int(px[k] + pr[k]) + 2)
            y0, y1 = max(0, int(py[k] - pr[k])), min(cam.h, int(py[k] + pr[k]) + 2)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = (np.arange(y0, y1)[:, None] * cam.w + np.arange(x0, x1)[None, :]).ravel()
            r = rocks[k, 3]
            oc = o - rocks[k, :3]
            b = d[sub] @ oc
            disc = b * b - (oc @ oc - r * r)
            tk = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
            closer = (tk > 0.05) & (tk < t[sub])
            t[sub[closer]], kind[sub[closer]], rid[sub[closer]] = tk[closer], 2, k

    # ------------------------------------------------------------ 影
    def _hls_shadow(self, P, L):
        """点 P から太陽方向へ伸ばした光線が HLS(円柱) に遮られるか."""
        h = self.world.hls
        R = float(h["radius"])
        z0, z1 = self.world.hls_z0, self.world.hls_z0 + float(h["height"])
        pxy = P[:, :2] - [h["x"], h["y"]]
        a = L[0] ** 2 + L[1] ** 2
        b = pxy[:, 0] * L[0] + pxy[:, 1] * L[1]
        disc = b * b - a * ((pxy ** 2).sum(1) - R * R)
        s = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0.0))) / max(a, 1e-12), -1.0)
        zh = P[:, 2] + L[2] * s
        return (disc > 0) & (s > 0.02) & (zh >= z0) & (zh <= z1)

    def _box_shadow(self, P, L, pose, Rm):
        """点 P から太陽方向へ伸ばした光線がローバー(箱)に遮られるか (車体座標系でのスラブ法)."""
        pb = (P - np.array(pose[:3])) @ Rm  # 世界 -> 車体
        lb = L @ Rm
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (self.box_lo - pb) / lb, (self.box_hi - pb) / lb
        t_in = np.nanmax(np.minimum(t1, t2), axis=1)
        t_out = np.nanmin(np.maximum(t1, t2), axis=1)
        return (t_out > np.maximum(t_in, 0.0)) & (t_out > 0.0)

    def _lookup(self, grid_u8, x, y, cell):
        """uint8 の影マップを双線形補間して 0..1 で返す."""
        n = grid_u8.shape[0]
        fx = np.clip((x - self.world.origin) / cell, 0, n - 1.001)
        fy = np.clip((y - self.world.origin) / cell, 0, n - 1.001)
        ix, iy = fx.astype(np.int64), fy.astype(np.int64)
        tx, ty = fx - ix, fy - iy
        g = grid_u8
        v = ((g[iy, ix] * (1 - tx) + g[iy, ix + 1] * tx) * (1 - ty)
             + (g[iy + 1, ix] * (1 - tx) + g[iy + 1, ix + 1] * tx) * ty)
        return v / 255.0

    # ------------------------------------------------------------ 恒星
    def _add_stars(self, rgb, sky, cam: CameraModel, basis):
        """空の画素に、恒星を点広がり関数(ガウス)で加算する。総フラックスは保存。"""
        sf = self.stars
        pc = sf.dirs @ basis
        f = np.maximum(pc[:, 0], 1e-6)
        u, v = (-pc[:, 1] / f) / cam.th, (pc[:, 2] / f) / cam.tv
        ok = (pc[:, 0] > 0.05) & (np.abs(u) < 1.02) & (np.abs(v) < 1.02)
        if not ok.any():
            return
        px = (u[ok] * 0.5 + 0.5) * cam.w - 0.5
        py = (0.5 - v[ok] * 0.5) * cam.h - 0.5
        radiance = sf.flux[ok] / cam.pix_omega
        ix0, iy0 = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
        offsets = [(dx, dy) for dy in (-1, 0, 1, 2) for dx in (-1, 0, 1, 2)]  # 周囲 4x4 画素に配る
        sig2 = 2 * STAR_PSF_SIGMA ** 2
        weights = [np.exp(-((ix0 + dx - px) ** 2 + (iy0 + dy - py) ** 2) / sig2) for dx, dy in offsets]
        total = np.sum(weights, axis=0) + 1e-12
        flat = rgb.reshape(-1, 3)
        for (dx, dy), wgt in zip(offsets, weights):
            xx, yy = ix0 + dx, iy0 + dy
            inside = (xx >= 0) & (xx < cam.w) & (yy >= 0) & (yy < cam.h)
            lin = np.where(inside, yy * cam.w + xx, 0)
            use = inside & sky[lin]
            np.add.at(flat, lin[use], ((radiance * wgt / total)[use])[:, None] * sf.tint[ok][use])
