// ライダーの点群: 車体の上から見た図 (左上のcanvas) と、右の3Dビュー上の点群。
// サーバから届くバイナリ: 12バイトのヘッダ + (x, y, z, 距離) の float32 x 点数。座標は車体座標。
import * as THREE from './vendor/three.module.min.js';
import { rad } from './scene.js';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const HEADER_BYTES = 12;
const STOPS = [[0.1, 0.2, 0.9], [0.0, 0.8, 0.9], [0.2, 0.9, 0.3], [1.0, 0.9, 0.1], [1.0, 0.25, 0.1]]; // 低 -> 高

/** 高さ z [m] -> 色 (out[o..o+2] に 0..1 の RGB を書く) */
function heightColor(z, out, o) {
  const f = clamp((z + 1.5) / 4.5, 0, 1) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(f)), k = f - i;
  for (let c = 0; c < 3; c++) out[o + c] = STOPS[i][c] * (1 - k) + STOPS[i + 1][c] * k;
}

export class Lidars {
  constructor(world, view, canvas) {
    this.rover = world.rover;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.viewRange = world.ui.lidar_view_range; // 表示半径 [m]
    this.lidars = world.lidars.map((cfg) => this.create3d(cfg, view));

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.viewRange = clamp(this.viewRange * Math.exp(e.deltaY * 0.001), 5, 120);
    }, { passive: false });
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement); // canvas 自身ではなく親を監視 (canvas のサイズ変更で再通知されないように)
  }

  /** 3Dビュー用: 取付マーカーと、車体座標の点群 */
  create3d(cfg, view) {
    const group = view.mountGroup(cfg.mount);
    group.add(new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.5, 0.25), new THREE.MeshBasicMaterial({ color: 0xffd23a })));
    const maxPoints = cfg.h_pixels * cfg.v_pixels;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxPoints * 3), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(maxPoints * 3), 3));
    geo.setDrawRange(0, 0);
    const points = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.12, vertexColors: true }));
    points.frustumCulled = false;
    view.rover.add(points);
    return { cfg, points, frame: null };
  }

  setPointsVisible(visible) { this.lidars.forEach((l) => { l.points.visible = visible; }); }

  resize() {
    const dpr = Math.min(devicePixelRatio, 2);
    this.canvas.width = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    this.canvas.height = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
  }

  /** バイナリメッセージ (type=1) を受け取る */
  onFrame(buf, dv) {
    const l = this.lidars[dv.getUint8(1)];
    if (!l) return;
    const n = dv.getUint32(4, true);
    const a = new Float32Array(buf, HEADER_BYTES, n * 4);
    l.frame = { a, n };
    const pos = l.points.geometry.attributes.position, col = l.points.geometry.attributes.color;
    for (let i = 0; i < n; i++) {
      pos.setXYZ(i, a[i * 4], a[i * 4 + 1], a[i * 4 + 2]);
      heightColor(a[i * 4 + 2], col.array, i * 3);
    }
    pos.needsUpdate = col.needsUpdate = true;
    l.points.geometry.setDrawRange(0, n);
  }

  // ---------------------------------------------------------------- 車体の上から見た図 (前が上)
  draw() {
    const { ctx, canvas, viewRange } = this;
    const W = canvas.width, H = canvas.height, dpr = W / Math.max(1, canvas.clientWidth);
    const cx = W / 2, cy = H * 0.62; // 自車を少し下に置き、前方を広く表示
    const scale = Math.min(W / 2, Math.min(cy, H - cy) + H * 0.2) / viewRange;
    const sx = (y) => cx - y * scale, sy = (x) => cy - x * scale; // 左(+y)=画面左, 前(+x)=画面上
    ctx.fillStyle = '#05070a';
    ctx.fillRect(0, 0, W, H);

    this.drawRings(cx, cy, scale, W, H, dpr);
    this.lidars.forEach((l) => this.drawFootprint(l.cfg, sx, sy, dpr));
    this.lidars.forEach((l) => l.frame && this.drawPoints(l.frame, sx, sy, scale, W, H, dpr));
    this.drawRover(cx, cy, scale, dpr);
  }

  drawRings(cx, cy, scale, W, H, dpr) {
    const ctx = this.ctx, step = this.viewRange > 60 ? 20 : this.viewRange > 25 ? 10 : 5;
    ctx.lineWidth = dpr;
    ctx.font = `${11 * dpr}px system-ui`;
    for (let r = step; r <= this.viewRange * 1.8; r += step) {
      ctx.strokeStyle = '#1b232c';
      ctx.beginPath();
      ctx.arc(cx, cy, r * scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = '#4b5968';
      ctx.fillText(`${r}m`, cx + 3 * dpr, cy - r * scale - 2 * dpr);
    }
    ctx.strokeStyle = '#1b232c';
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
    ctx.moveTo(0, cy); ctx.lineTo(W, cy);
    ctx.stroke();
  }

  /** ライダーが地面に当たる範囲 (平坦な地面を仮定した概算): 水平距離 = センサ高 / tan(俯角)。視野の上端が水平以上なら最大距離まで */
  drawFootprint(cfg, sx, sy, dpr) {
    const ctx = this.ctx, m = cfg.mount, yaw = rad(m.yaw_deg || 0), half = rad(cfg.h_fov_deg) / 2;
    const ox = m.x || 0, oy = m.y || 0, height = (m.z || 0) + this.rover.wheel_radius;
    const elTop = (m.pitch_deg || 0) + cfg.v_fov_deg / 2, elBottom = (m.pitch_deg || 0) - cfg.v_fov_deg / 2;
    const far = elTop < -0.01 ? Math.min(cfg.max_range, height / Math.tan(rad(-elTop))) : cfg.max_range;
    const near = elBottom < -0.01 ? Math.min(far, Math.max(cfg.min_range, height / Math.tan(rad(-elBottom)))) : cfg.min_range;
    const at = (r, a) => [sx(oy + r * Math.sin(a)), sy(ox + r * Math.cos(a))];
    const STEPS = 24;
    ctx.strokeStyle = '#ffd23a77';
    ctx.fillStyle = '#ffd23a10';
    ctx.lineWidth = dpr;
    ctx.beginPath();
    for (let k = 0; k <= STEPS; k++) { const q = at(far, yaw + half - (2 * half * k) / STEPS); k ? ctx.lineTo(...q) : ctx.moveTo(...q); }
    for (let k = STEPS; k >= 0; k--) ctx.lineTo(...at(near, yaw + half - (2 * half * k) / STEPS));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  drawPoints(frame, sx, sy, scale, W, H, dpr) {
    const ctx = this.ctx, a = frame.a, size = Math.max(1.5 * dpr, 0.12 * scale), rgb = [0, 0, 0];
    for (let i = 0; i < frame.n; i++) {
      const px = sx(a[i * 4 + 1]), py = sy(a[i * 4]);
      if (px < -4 || px > W + 4 || py < -4 || py > H + 4) continue;
      heightColor(a[i * 4 + 2], rgb, 0);
      ctx.fillStyle = `rgb(${(rgb[0] * 255) | 0},${(rgb[1] * 255) | 0},${(rgb[2] * 255) | 0})`;
      ctx.fillRect(px - size / 2, py - size / 2, size, size);
    }
  }

  drawRover(cx, cy, scale, dpr) {
    const ctx = this.ctx, L = this.rover.length * scale, W = this.rover.width * scale;
    ctx.fillStyle = '#e9eaec';
    ctx.strokeStyle = '#4cc2ff';
    ctx.lineWidth = 1.5 * dpr;
    ctx.fillRect(cx - W / 2, cy - L / 2, W, L);
    ctx.strokeRect(cx - W / 2, cy - L / 2, W, L);
    ctx.fillStyle = '#4cc2ff'; // 前を示す三角
    ctx.beginPath();
    ctx.moveTo(cx, cy - L / 2 - 6 * dpr);
    ctx.lineTo(cx - 5 * dpr, cy - L / 2 + 2 * dpr);
    ctx.lineTo(cx + 5 * dpr, cy - L / 2 + 2 * dpr);
    ctx.fill();
  }
}
