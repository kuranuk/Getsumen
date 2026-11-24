// カメラ画像: サーバがレイトレーシングしたHDR画像(線形, float16)を受け取り、
// 実機センサ風の処理 (自動露出 → ノイズ・周辺減光 → トーンカーブ → sRGB, 太陽グレア, 地面距離の目盛) を加えて表示する。
import * as THREE from './vendor/three.module.min.js';
import { deg, rad } from './scene.js';

// three.js カメラ (-Z が視線, +Y が上) -> 車体座標 (+X が前, +Z が上)
const CAM_BASIS = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(
  0, 0, -1, 0,
  -1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 0, 1));

const HEADER_BYTES = 36; // type, カメラ番号, 幅, 高さ, 時刻, 撮影時のローバー姿勢 x6 (server.py と一致させる)

// float16 -> float32 変換表
const HALF = new Float32Array(65536);
for (let i = 0; i < 65536; i++) {
  const e = (i >> 10) & 31, m = i & 1023, sign = i & 0x8000 ? -1 : 1;
  HALF[i] = e === 0 ? sign * m * 2 ** -24 : e === 31 ? (m ? NaN : sign * Infinity) : sign * (1 + m / 1024) * 2 ** (e - 15);
}

const aces = (x) => Math.min(1, Math.max(0, (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)));
const srgb = (x) => (x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055);
const gauss = () => Math.random() + Math.random() + Math.random() - 1.5; // 標準偏差 0.5

export class Cameras {
  /** @param container カメラ画像を並べる要素 */
  constructor(world, view, container) {
    this.view = view;
    this.wheelRadius = world.rover.wheel_radius;
    this.distanceOverlay = true; // 画面のチェックボックスで切り替える
    container.style.setProperty('--cols', world.ui.camera_grid_columns);
    this.cams = world.cameras.map((cfg) => this.createCamera(cfg, container));
  }

  createCamera(cfg, container) {
    // 投影の計算用 (取付位置・向きにある three.js のカメラ) と、取付位置を示す小さなマーカー
    const group = this.view.mountGroup(cfg.mount);
    const aspect = cfg.width / cfg.height;
    const cam = new THREE.PerspectiveCamera(deg(2 * Math.atan(Math.tan(rad(cfg.fov_deg) / 2) / aspect)), aspect);
    cam.quaternion.copy(CAM_BASIS);
    group.add(cam, new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 0.18), new THREE.MeshBasicMaterial({ color: 0x111111 })));

    // 画面の部品: 見出し + canvas。配置は cfg.grid = {row, col, span}
    const el = document.createElement('div');
    el.className = 'cam';
    el.innerHTML = `<header><span>${cfg.label}</span><span class="fps"></span></header>`;
    const canvas = document.createElement('canvas');
    canvas.width = cfg.width;
    canvas.height = cfg.height;
    canvas.style.filter = cfg.blur_px > 0 ? `blur(${cfg.blur_px}px)` : 'none';
    el.appendChild(canvas);
    if (cfg.grid) {
      el.style.gridRow = String(cfg.grid.row);
      el.style.gridColumn = `${cfg.grid.col} / span ${cfg.grid.span || 1}`;
    }
    container.appendChild(el);

    return {
      cfg, cam, ctx: canvas.getContext('2d'), fpsEl: el.querySelector('.fps'),
      hdr: new Float32Array(cfg.width * cfg.height * 4),         // 受信した線形RGB (上の行から)
      img: new ImageData(cfg.width, cfg.height),
      exposure: 1.0, lum: [], frames: 0, t0: performance.now(),
    };
  }

  /** サーバから届いた画像 (バイナリメッセージ) を処理して表示する */
  onFrame(buf, dv) {
    const c = this.cams[dv.getUint8(1)];
    const w = dv.getUint16(2, true), h = dv.getUint16(4, true);
    if (!c || w !== c.cfg.width || h !== c.cfg.height) return;
    const pose = [0, 1, 2, 3, 4, 5].map((i) => dv.getFloat32(12 + i * 4, true)); // x, y, z, yaw, pitch, roll
    const half = new Uint16Array(buf, HEADER_BYTES, w * h * 3);
    for (let i = 0; i < w * h; i++) {
      c.hdr[i * 4] = HALF[half[i * 3]]; c.hdr[i * 4 + 1] = HALF[half[i * 3 + 1]]; c.hdr[i * 4 + 2] = HALF[half[i * 3 + 2]];
    }

    // グレアと距離目盛の投影は、撮影した時点のローバー姿勢で計算する (走行中でも画像と一致させる)
    const rover = this.view.rover, keepPos = rover.position.clone(), keepRot = rover.rotation.clone();
    this.view.setRoverPose(...pose);
    c.cam.matrixWorldInverse.copy(c.cam.matrixWorld).invert();
    this.draw(c);
    rover.position.copy(keepPos);
    rover.rotation.copy(keepRot);
    rover.updateMatrixWorld(true);

    c.frames++;
    const now = performance.now();
    if (now - c.t0 > 2000) {
      c.fpsEl.textContent = `${((c.frames * 1000) / (now - c.t0)).toFixed(1)} Hz`;
      c.frames = 0;
      c.t0 = now;
    }
  }

  draw(c) {
    this.updateExposure(c);
    this.drawPixels(c);
    this.drawGlare(c);
    if (c.cfg.distance_overlay && this.distanceOverlay) this.drawDistanceRings(c);
  }

  /** 自動露出: 空(真っ黒)を除いた輝度の対数平均を中間調に合わせ、日向の明部(上位3%)が飽和しすぎないよう頭打ちにする */
  updateExposure(c) {
    const cfg = c.cfg, f = c.hdr;
    if (cfg.exposure !== 'auto') { c.exposure = Number(cfg.exposure) * 2 ** cfg.ev_bias; return; }
    let sumLog = 0;
    c.lum.length = 0;
    for (let i = 0; i < cfg.width * cfg.height; i += 7) {
      const L = 0.2126 * f[i * 4] + 0.7152 * f[i * 4 + 1] + 0.0722 * f[i * 4 + 2];
      if (L > 2e-4) { c.lum.push(Math.min(L, 3)); sumLog += Math.log(Math.min(L, 3)); }
    }
    let target = 1;
    if (c.lum.length > 20) {
      c.lum.sort((a, b) => a - b);
      const p97 = c.lum[Math.floor(c.lum.length * 0.97)];
      target = Math.min(cfg.auto_exposure_max, 0.16 / Math.exp(sumLog / c.lum.length), 1.6 / p97);
    }
    c.exposure += (target * 2 ** cfg.ev_bias - c.exposure) * 0.35; // 急に変わらないようなます
  }

  /** 露出 → 周辺減光 → ノイズ (読み出し + ショット) → トーンカーブ → sRGB */
  drawPixels(c) {
    const { width: w, height: h, read_noise: readNoise, shot_noise: shotNoise, vignette } = c.cfg;
    const f = c.hdr, out = c.img.data, e = c.exposure, readVar = readNoise ** 2;
    for (let y = 0; y < h; y++) {
      const ny = ((y + 0.5) / h) * 2 - 1;
      for (let x = 0; x < w; x++) {
        const nx = ((x + 0.5) / w) * 2 - 1, r2 = (nx * nx + ny * ny) / 2;
        const vig = 1 - vignette * r2 * r2 * 1.6 - vignette * 0.2 * r2;
        const i = (y * w + x) * 4;
        for (let k = 0; k < 3; k++) {
          let v = f[i + k] * e * vig;
          v += 2 * gauss() * Math.sqrt(readVar + shotNoise * Math.max(v, 0) * 0.04);
          out[i + k] = 255 * srgb(aces(v)) + 0.5;
        }
        out[i + 3] = 255;
      }
    }
    c.ctx.putImageData(c.img, 0, 0);
  }

  /** 画面に投影した点 (カメラ座標 p) のピクセル位置。カメラの後ろなら null */
  toPixel(c, p) {
    if (p.z > -0.05) return null;
    const t = Math.tan(rad(c.cam.fov) / 2);
    return [(p.x / -p.z / (t * c.cam.aspect) * 0.5 + 0.5) * c.cfg.width, (1 - (p.y / -p.z / t * 0.5 + 0.5)) * c.cfg.height];
  }

  /** 太陽が実際に写っているときだけ光芒を足す (画素の輝度で判定: 地形やHLSに隠れれば出ない) */
  drawGlare(c) {
    const { width: w, height: h, glare } = c.cfg, ctx = c.ctx;
    if (glare <= 0) return;
    const p = this.toPixel(c, this.view.sunDir.clone().transformDirection(c.cam.matrixWorldInverse));
    if (!p || p[0] < -w * 0.3 || p[0] > w * 1.3 || p[1] < -h * 0.3 || p[1] > h * 1.3) return;
    let visible = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const x = Math.round(p[0]) + dx, y = Math.round(p[1]) + dy;
      if (x >= 0 && y >= 0 && x < w && y < h && c.hdr[(y * w + x) * 4] > 30) visible++; // 太陽円盤は極端に明るい
    }
    const a = Math.min(1, visible / 6) * glare;
    if (a < 0.02) return;
    const g = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], w * 0.55);
    g.addColorStop(0, `rgba(255,246,230,${0.95 * a})`);
    g.addColorStop(0.05, `rgba(255,240,215,${0.6 * a})`);
    g.addColorStop(0.25, `rgba(255,225,190,${0.16 * a})`);
    g.addColorStop(1, 'rgba(255,220,180,0)');
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    ctx.globalCompositeOperation = 'source-over';
  }

  /** 地面距離の目盛: カメラ位置を中心とする半径 d の円を、平坦な接地面 (車輪の接地高さ) に仮定して投影する。
   *  地形の凹凸の分だけ実距離とずれる推定値 (ライダーの点群と比べると違いが分かる)。 */
  drawDistanceRings(c) {
    const { width: w, height: h, mount: m, fov_deg: fov, overlay_distances: distances } = c.cfg, ctx = c.ctx;
    const groundZ = -this.wheelRadius;
    const p = new THREE.Vector3();
    const project = (x, y) => {
      p.set(x, y, groundZ).applyMatrix4(this.view.rover.matrixWorld).applyMatrix4(c.cam.matrixWorldInverse);
      return this.toPixel(c, p);
    };
    const yaw = rad(m.yaw_deg || 0), spread = rad(fov) / 2 + 0.2;
    ctx.save();
    ctx.font = '11px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    let lastLabelY = 1e9; // 近い順に描き、ラベルが密集する遠方は間引く
    for (const d of [...distances].sort((a, b) => a - b)) {
      ctx.strokeStyle = 'rgba(255,230,120,0.7)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      let penDown = false, label = null;
      for (let a = -spread; a <= spread; a += rad(1)) {
        const q = project((m.x || 0) + d * Math.cos(yaw + a), (m.y || 0) + d * Math.sin(yaw + a));
        if (!q || q[0] < -20 || q[0] > w + 20 || q[1] < -20 || q[1] > h + 20) { penDown = false; continue; }
        penDown ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]);
        penDown = true;
        const inside = q[1] > 10 && q[1] < h - 6;
        if (inside && (!label || Math.abs(q[0] - w / 2) < Math.abs(label[0] - w / 2))) label = q; // 画像の中央に近い点にラベル
      }
      ctx.stroke();
      if (label && Math.abs(label[1] - lastLabelY) >= 13) {
        lastLabelY = label[1];
        ctx.strokeStyle = 'rgba(0,0,0,0.85)';
        ctx.lineWidth = 3;
        ctx.strokeText(`${d} m`, label[0] + 3, label[1] - 6);
        ctx.fillStyle = '#ffe678';
        ctx.fillText(`${d} m`, label[0] + 3, label[1] - 6);
      }
    }
    ctx.restore();
  }
}
