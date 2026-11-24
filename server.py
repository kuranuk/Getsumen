"""月面ロボットシミュレータ (有人与圧ローバー x HLS アプローチ検討用)。このファイル1つに Python 側のすべてがある。

    python server.py [-c 設定.yaml] [--port 8080] [--seed 7]     ->  ブラウザで http://localhost:8080/

上から順に:
  1. 設定        config/default.yaml (既定) + 指定ファイル (差分) の読み込み
  2. 座標系      車体座標 x=前 y=左 z=上 / 世界座標 x=東 y=北 z=上
  3. 月面        高さマップ (起伏+クレータ), 岩, 太陽の影, レイキャスト
  4. ローバー    ユニサイクル運動 + 地形追従
  5. ライダー    フラッシュライダー (格子状レイ -> 点群)
  6. カメラ画像  レイトレーシング
  7. ワーカー    重い計算 (レイトレ/ライダー/影の再計算) を別プロセスで実行
  8. サーバ      aiohttp + WebSocket (状態 JSON, 点群・カメラ画像バイナリ), 静的ファイル (web/)
  9. 起動        コマンドライン引数
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import multiprocessing as mp
import os
import struct
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from aiohttp import WSMsgType, web


# ============================================================================
# 1. 設定
# ============================================================================

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def deep_merge(base: dict, over: dict) -> dict:
    """base に over を再帰的に上書きした新しい dict (リストは置き換え)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path=None) -> dict:
    cfg = _read(DEFAULT_CONFIG)
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        cfg = deep_merge(cfg, _read(path))

    # カメラ/ライダーの各項目に、共通の既定値 (*_defaults) を適用する
    sensors = cfg["sensors"]
    names = set()
    for kind, defaults in (("cameras", "camera_defaults"), ("lidars", "lidar_defaults")):
        sensors[kind] = [deep_merge(sensors[defaults], c) for c in sensors[kind]]
        for c in sensors[kind]:
            c.setdefault("label", c["name"])
            if c["name"] in names:
                raise ValueError(f"センサ名が重複しています: {c['name']}")
            names.add(c["name"])
    return cfg


# ============================================================================
# 2. 座標系
#
# 世界座標: x=東, y=北, z=上 (右手系)
# 車体座標: x=前, y=左, z=上
# 姿勢は R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。pitch は「機首下げ」が正 (y軸まわりの右手系回転)。
# センサ取り付け角の pitch は設定ファイル上「上向きが正」(内部で符号反転)。
# ============================================================================

def rot_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def mount_rotation(mount: dict) -> np.ndarray:
    """設定の mount(yaw_deg, pitch_deg[上向き正]) -> 車体座標系での回転行列."""
    yaw = math.radians(mount.get("yaw_deg", 0.0))
    pitch_up = math.radians(mount.get("pitch_deg", 0.0))
    return rot_zyx(yaw, -pitch_up, 0.0)


def mount_position(mount: dict) -> np.ndarray:
    return np.array([mount.get("x", 0.0), mount.get("y", 0.0), mount.get("z", 0.0)], float)


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ============================================================================
# 3. 月面: 高さマップ・岩・太陽の影・レイキャスト
# ============================================================================

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
        """レイマーチングの刻み (遠いほど粗く)."""
        ts, t = [], tmin
        while t < tmax:
            ts.append(t)
            t += 0.2 + 0.012 * t
        ts.append(tmax)
        return ts

    def raycast(self, o: np.ndarray, d: np.ndarray, tmin: float, tmax: float, rocks: bool = True):
        """原点 o (3,) から方向 d (N,3 単位ベクトル) の各光線の最初の交点.
        戻り値 (t, kind, rid): 距離 (当たらなければ inf) / 種別 (0=なし 1=地形 2=岩 3=HLS) / 岩の番号.
        地形は tmax までマーチングし、地形の外側は z=0 の平面 (距離無制限) として扱う。
        rocks=False なら岩は含めない (カメラ用には画素窓を使う高速版 RayTracer._rocks がある)."""
        t = self._ray_terrain(o, d, tmin, tmax)
        kind = np.where(np.isfinite(t), 1, 0).astype(np.int8)
        rid = np.full(len(d), -1, np.int32)

        far = (kind == 0) & (d[:, 2] < -1e-5)  # 地形の外側の平面
        t[far] = -o[2] / d[far, 2]
        kind[far] = 1

        if rocks:
            self._ray_rocks(o, d, tmin, tmax, t, kind, rid)
        self._ray_hls(o, d, tmin, tmax, t, kind)
        return t, kind, rid

    def _ray_terrain(self, o, d, tmin, tmax) -> np.ndarray:
        """高さマップのレイマーチング (交差を挟んだ区間を二分法で詰める). 当たらなければ inf."""
        t_hit = np.full(len(d), np.inf)
        idx = np.nonzero(d[:, 2] < 0.5)[0]  # 大きく上向きの光線は地面に当たらない
        dx, dy, dz = d[idx, 0].copy(), d[idx, 1].copy(), d[idx, 2].copy()
        half = self.size / 2.0
        prev = tmin
        for t in self._steps(tmin, tmax):
            if idx.size == 0:
                break
            px, py, pz = o[0] + dx * t, o[1] + dy * t, o[2] + dz * t
            hit = pz - self.height(px, py) <= 0.0
            if hit.any():
                lo = np.full(hit.sum(), prev)
                up = np.full(hit.sum(), t)
                hx, hy, hz = dx[hit], dy[hit], dz[hit]
                for _ in range(8):
                    mid = 0.5 * (lo + up)
                    above = o[2] + hz * mid - self.height(o[0] + hx * mid, o[1] + hy * mid) > 0
                    lo = np.where(above, mid, lo)
                    up = np.where(above, up, mid)
                t_hit[idx[hit]] = 0.5 * (lo + up)
            keep = ~hit & (np.abs(px) < half) & (np.abs(py) < half)  # 当たった光線と地形外へ出た光線は打ち切る
            idx, dx, dy, dz = idx[keep], dx[keep], dy[keep], dz[keep]
            prev = t
        return t_hit

    def _ray_rocks(self, o, d, tmin, tmax, t, kind, rid):
        """岩 (球) との交差. 光線束の円錐に入らない岩は先に除外する."""
        if len(self.rocks) == 0:
            return
        rel = self.rocks[:, :3] - o
        dist = np.linalg.norm(rel, axis=1)
        axis = d.mean(axis=0)
        axis /= np.linalg.norm(axis) + 1e-12
        spread = float(np.arccos(np.clip((d @ axis).min(), -1, 1)))
        for k in np.nonzero(dist < tmax + self.rocks[:, 3])[0]:
            r = self.rocks[k, 3]
            dk = max(dist[k], 1e-6)
            ang = math.acos(np.clip(rel[k] @ axis / dk, -1, 1))
            if ang > spread + math.asin(min(1.0, r / dk)) + 0.02:
                continue
            oc = o - self.rocks[k, :3]
            b = d @ oc
            disc = b * b - (oc @ oc - r * r)
            tk = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
            tk[(tk < tmin) | (tk > tmax)] = np.inf
            closer = tk < t
            t[closer], kind[closer], rid[closer] = tk[closer], 2, k

    def _ray_hls(self, o, d, tmin, tmax, t, kind):
        """HLS (鉛直な円柱: 側面 + 上面) との交差."""
        h = self.hls
        c = np.array([h["x"], h["y"]])
        R = float(h["radius"])
        z0, z1 = self.hls_z0, self.hls_z0 + float(h["height"])
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        oc = o[:2] - c
        b = d[:, 0] * oc[0] + d[:, 1] * oc[1]
        disc = b * b - a * (oc @ oc - R * R)
        ok = (a > 1e-12) & (disc > 0)
        with np.errstate(invalid="ignore"):
            side = np.full(len(d), np.inf)
            side[ok] = (-b[ok] - np.sqrt(disc[ok])) / a[ok]
            z = o[2] + d[:, 2] * side
            side[(side < tmin) | (side > tmax) | (z < z0) | (z > z1)] = np.inf

            top = np.full(len(d), np.inf)
            down = d[:, 2] < -1e-9
            top[down] = (z1 - o[2]) / d[down, 2]
            inside = (o[0] + d[:, 0] * top - c[0]) ** 2 + (o[1] + d[:, 1] * top - c[1]) ** 2 <= R * R
            top[(top < tmin) | (top > tmax) | ~inside] = np.inf

        th = np.minimum(side, top)
        closer = th < t
        t[closer], kind[closer] = th[closer], 3

    def surface_normals(self, P, kind, rid):
        """交点 P (N,3) での面の法線 (地形=勾配, 岩=球, HLS=円柱側面/上面)."""
        n = np.zeros((len(P), 3))
        m = kind == 1
        if m.any():
            e = 0.6
            x, y = P[m, 0], P[m, 1]
            hx = (self.height(x + e, y) - self.height(x - e, y)) / (2 * e)
            hy = (self.height(x, y + e) - self.height(x, y - e)) / (2 * e)
            n[m] = np.stack([-hx, -hy, np.ones_like(hx)], 1)
        m = kind == 2
        if m.any():
            n[m] = (P[m] - self.rocks[rid[m], :3]) / self.rocks[rid[m], 3][:, None]
        m = kind == 3
        if m.any():
            h = self.hls
            v = P[m] - [h["x"], h["y"], 0.0]
            v[:, 2] = 0.0
            on_top = P[m, 2] >= self.hls_z0 + float(h["height"]) - 1e-3
            v[on_top] = [0.0, 0.0, 1.0]
            n[m] = v
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

    # ---------------- 配信用 ----------------
    def describe(self) -> dict:
        return {
            "size": self.size, "res": self.res, "n": self.n, "origin": self.origin,
            "sun_azimuth": self.sun_azimuth, "sun_elevation": self.sun_elevation,
            "hls_z0": self.hls_z0,
        }


# ============================================================================
# 4. ローバー
# ============================================================================

class Rover:
    def __init__(self, cfg: dict, world: World):
        self.c = cfg
        self.world = world
        self.L = float(cfg["length"])
        self.W = float(cfg["width"])
        self.wheel_r = float(cfg["wheel_radius"])
        self.vmax = float(cfg["max_speed"])
        self.wmax = math.radians(cfg["max_yaw_rate_deg"])
        self.acc = float(cfg["accel"])
        self.wacc = math.radians(cfg["yaw_accel_deg"])
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        s = self.c["start"]
        with self._lock:
            self.x, self.y = float(s["x"]), float(s["y"])
            self.yaw = math.radians(s.get("yaw_deg", 0.0))
            self.v = self.w = 0.0
            self.odo = 0.0
            z, pitch, roll = self._terrain_pose(self.x, self.y, self.yaw)
            self.z, self.pitch, self.roll = z, pitch, roll

    # ---- 地形追従 ----
    def _terrain_pose(self, x, y, yaw):
        a, b = self.L * 0.4, self.W / 2.0
        cy, sy = math.cos(yaw), math.sin(yaw)
        pts = np.array([[a, 0], [-a, 0], [0, b], [0, -b]], float)
        wx = x + cy * pts[:, 0] - sy * pts[:, 1]
        wy = y + sy * pts[:, 0] + cy * pts[:, 1]
        hf, hr, hl, hrt = self.world.height(wx, wy)
        slope_f = (hf - hr) / (2 * a)
        slope_l = (hl - hrt) / (2 * b)
        z = (hf + hr + hl + hrt) / 4.0 + self.wheel_r
        return float(z), float(-math.atan(slope_f)), float(math.atan(slope_l))

    # ---- 運動 ----
    def step(self, dt: float, cmd_v: float, cmd_w: float):
        with self._lock:
            tv = max(-1.0, min(1.0, cmd_v)) * self.vmax
            tw = max(-1.0, min(1.0, cmd_w)) * self.wmax
            self.v += max(-self.acc * dt, min(self.acc * dt, tv - self.v))
            self.w += max(-self.wacc * dt, min(self.wacc * dt, tw - self.w))

            nyaw = self.yaw + self.w * dt
            mid = 0.5 * (self.yaw + nyaw)
            nx = self.x + self.v * math.cos(mid) * dt
            ny = self.y + self.v * math.sin(mid) * dt

            # HLS との衝突 (円近似) -> 進入を止める
            h = self.world.hls
            keep = h["radius"] + self.L / 2.0
            if math.hypot(nx - h["x"], ny - h["y"]) < keep and \
               math.hypot(nx - h["x"], ny - h["y"]) < math.hypot(self.x - h["x"], self.y - h["y"]):
                nx, ny, self.v = self.x, self.y, 0.0
            half = self.world.size / 2.0 - 2.0
            nx, ny = max(-half, min(half, nx)), max(-half, min(half, ny))

            self.odo += math.hypot(nx - self.x, ny - self.y)
            self.x, self.y, self.yaw = nx, ny, wrap_pi(nyaw)

            z, pitch, roll = self._terrain_pose(self.x, self.y, self.yaw)
            k = min(1.0, dt * 8.0)  # サスペンションの簡易ローパス
            self.z += (z - self.z) * k
            self.pitch += (pitch - self.pitch) * k
            self.roll += (roll - self.roll) * k

    def pose(self) -> tuple[float, float, float, float, float, float]:
        with self._lock:
            return (self.x, self.y, self.z, self.yaw, self.pitch, self.roll)

    def state(self, sim_t: float) -> dict:
        with self._lock:
            return {
                "t": "state", "sim_t": round(sim_t, 3),
                "x": self.x, "y": self.y, "z": self.z,
                "yaw": self.yaw, "pitch": self.pitch, "roll": self.roll,
                "v": self.v, "w": self.w, "odo": self.odo,
            }


# ============================================================================
# 5. ライダー
# ============================================================================

class FlashLidar:
    def __init__(self, cfg: dict, world: World, seed: int = 0):
        self.world = world
        self.name = cfg["name"]
        self.rate = float(cfg["rate_hz"])
        self.tmin = float(cfg["min_range"])
        self.tmax = float(cfg["max_range"])
        self.noise = float(cfg["range_noise_std"])
        self.dropout = float(cfg["dropout"])
        # 検出限界 (max_range の基準反射率と、面の種別 [なし, 地形, 岩, HLS] ごとの反射率)
        self.rho_ref = float(cfg["reference_reflectivity"])
        self.rho = np.array([0.0, cfg["ground_reflectivity"], cfg["rock_reflectivity"], cfg["hls_reflectivity"]])
        self.rng = np.random.default_rng(seed)

        # 画素ごとの方向 (センサ座標: x=前, y=左, z=上) を車体座標へ
        az = np.radians(np.linspace(cfg["h_fov_deg"] / 2, -cfg["h_fov_deg"] / 2, int(cfg["h_pixels"])))
        el = np.radians(np.linspace(cfg["v_fov_deg"] / 2, -cfg["v_fov_deg"] / 2, int(cfg["v_pixels"])))
        AZ, EL = np.meshgrid(az, el)
        d = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
        self.m_pos = mount_position(cfg["mount"])
        self.dirs_body = d @ mount_rotation(cfg["mount"]).T

    def scan(self, pose) -> np.ndarray:
        """pose=(x,y,z,yaw,pitch,roll). 戻り値: (n,4) float32 [x,y,z,range] (車体座標系)."""
        R = rot_zyx(*pose[3:])
        origin = np.array(pose[:3]) + R @ self.m_pos
        d_world = self.dirs_body @ R.T
        t, kind, rid = self.world.raycast(origin, d_world, self.tmin, self.tmax)

        # 検出限界: 受信強度 ∝ 反射率 x cos(入射角) / 距離^2。地面すれすれや暗い面ほど短い距離で検出できなくなる
        hit = np.isfinite(t)
        normals = self.world.surface_normals(origin + d_world * np.where(hit, t, 0.0)[:, None], kind, rid)
        cos_inc = np.abs((normals * d_world).sum(1))
        r_eff = self.tmax * np.sqrt(self.rho[kind] * cos_inc / self.rho_ref)
        valid = hit & (t <= np.minimum(r_eff, self.tmax))
        valid &= self.rng.random(len(t)) >= self.dropout

        t = t[valid] + self.rng.normal(0.0, self.noise, valid.sum())
        pts = self.m_pos + self.dirs_body[valid] * t[:, None]
        return np.column_stack([pts, t]).astype(np.float32)


# ============================================================================
# 6. カメラ画像のレイトレーシング
#
# 各画素から光線を飛ばして 地形(高さマップ) / 岩(球) / HLS(円柱) との交点を求め、
# 太陽方向へ影レイを飛ばして日照を判定し、月面の反射(Lommel-Seeliger)で放射輝度[相対値]を計算する。
# 出力は線形(HDR)。露出/ノイズ/トーンカーブなどのセンサ処理はブラウザ側 (web/app.js の Cameras)。
#
# 放射輝度の単位: 垂直入射の太陽光を受ける反射率1の拡散面 = 1.0 (太陽の放射照度 E=pi).
#   太陽円盤 = E / 立体角(6.7e-5 sr) ~ 4.7e4
#   恒星   : 放射照度 E = pi * 10^(-0.4 (m + 26.74)) を画素の立体角で割る  (m: 見かけの等級)
# ============================================================================

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


# ============================================================================
# 7. ワーカー: 重い計算を別プロセスで
#
# メインプロセス (asyncio) の GIL を塞がないよう ProcessPoolExecutor で並列化する。
# 各ワーカーは同じ設定(seed)から同じ World を決定論的に再生成するので、地形データの転送は不要。
# ============================================================================

_S: dict = {}  # ワーカープロセスごとの World / RayTracer / FlashLidar


def worker_init(cfg: dict) -> None:
    world = World(cfg)
    _S["world"] = world
    _S["rt"] = RayTracer(cfg, world)
    _S["lidars"] = [FlashLidar(c, world, seed=100 + i) for i, c in enumerate(cfg["sensors"]["lidars"])]


def worker_render_camera(name: str, pose, sun_az: float, sun_el: float, shade, rock_shade):
    world = _S["world"]
    world.sun_azimuth, world.sun_elevation = sun_az, sun_el  # 太陽はメイン側の値に合わせる (ラジアン)
    return _S["rt"].render(name, pose, shade, rock_shade)


def worker_scan_lidar(idx: int, pose):
    return _S["lidars"][idx].scan(pose)


def worker_bake(az_deg: float, el_deg: float):
    """太陽の方位/仰角 [deg] を設定して影を再計算. 戻り値: (方位rad, 仰角rad, 地形の影, 岩の影)."""
    world = _S["world"]
    world.set_sun(az_deg, el_deg)
    return world.sun_azimuth, world.sun_elevation, world.bake_shade(), world.bake_rock_shade()


# ============================================================================
# 8. サーバ
# ============================================================================

MSG_LIDAR = 1
MSG_CAMERA = 2


class SimServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.world = World(cfg)
        self.rover = Rover(cfg["rover"], self.world)
        self.cameras = RayTracer(cfg, self.world).cams  # 名前 -> CameraModel (画像はワーカーで生成)
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
                                        initializer=worker_init, initargs=(cfg,))

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
                az_r, el_r, shade, rock_shade = await loop.run_in_executor(self.pool, worker_bake, az, el)
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
        pts = await asyncio.get_running_loop().run_in_executor(self.pool, worker_scan_lidar, idx, self.rover.pose())
        await self._broadcast(struct.pack("<BBxxIf", MSG_LIDAR, idx, len(pts), self.sim_t) + pts.tobytes())

    async def camera_job(self, name: str):
        """レイトレーシングで画像を生成して配信 (HDR float16, 線形). 露出/ノイズ等はブラウザ側で加える."""
        cam, pose = self.cameras[name], self.rover.pose()
        img = await asyncio.get_running_loop().run_in_executor(
            self.pool, worker_render_camera, name, pose,
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
        return web.FileResponse(ROOT / "web" / "index.html")

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
    app.router.add_static("/", ROOT / "web", show_index=False)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


# ============================================================================
# 9. 起動
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="月面ロボットシミュレータ (Web UI)")
    ap.add_argument("--config", "-c", help="設定YAML (config/default.yaml への差分を書く)")
    ap.add_argument("--host", help="待受ホスト (設定を上書き)")
    ap.add_argument("--port", type=int, help="待受ポート (設定を上書き)")
    ap.add_argument("--seed", type=int, help="地形の乱数シード (設定を上書き)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.host:
        cfg["server"]["host"] = args.host
    if args.port:
        cfg["server"]["port"] = args.port
    if args.seed is not None:
        cfg["world"]["seed"] = args.seed

    t0 = time.time()
    app = create_app(cfg)
    host, port = cfg["server"]["host"], cfg["server"]["port"]
    print(f"[server] 地形生成 {time.time() - t0:.2f}s / seed={cfg['world']['seed']}")
    print(f"[server] ブラウザで開く: http://localhost:{port}/")
    web.run_app(app, host=host, port=port, print=None)



if __name__ == "__main__":
    main()
