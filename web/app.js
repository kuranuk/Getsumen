// 月面ロボットシミュレータ (ブラウザ側)。このファイル1つで、次の部品を組み立てる:
//   MoonView 右の3Dビュー / Cameras 左下のカメラ画像 / Lidars 左上の点群 / setupInput キーボード / main 全体と通信
import * as THREE from './vendor/three.module.min.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1); // 世界座標・車体座標とも z-up

const $ = (s) => document.querySelector(s);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const rad = (d) => (d * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;

// ======================================================================
// 右側の3Dビュー (three.js): 月面・岩・HLS・ローバー・太陽。世界座標・車体座標とも z-up (x=前/東, y=左/北)
// ======================================================================

const SUN_DIST = 1500;
const SHADOW_MAP = 2048;

// 月面の反射: Lommel-Seeliger 則 (L ∝ μ0 / (μ0 + μ))。Lambert (L ∝ μ0) だと、太陽が地平線近くの平地が真っ暗になる。
// 2倍して、垂直入射で Lambert と同じ明るさに正規化する。
const LS_CHUNK = THREE.ShaderChunk.lights_lambert_pars_fragment.replace(
  'vec3 irradiance = dotNL * directLight.color;',
  'float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );\n\tvec3 irradiance = ( 2.0 * dotNL / ( dotNL + dotNV + 0.03 ) ) * directLight.color;');

class MoonView {
  constructor(world, heights, container) {
    this.world = world;
    this.terrain = world.terrain;
    this.sunDir = new THREE.Vector3(0, 0, 1);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(this.renderer.domElement);

    this.addLights();
    this.addStars();
    this.addTerrain(heights);
    this.addRocks();
    this.addHls();
    this.addRover();
    this.setSun(world.sun.az, world.sun.el);
  }

  // ---------------------------------------------------------------- 光源
  addLights() {
    // 太陽 (南極: 仰角 0〜数度)。強度 π: 垂直入射の白面が反射率どおりの明るさになる
    this.sun = new THREE.DirectionalLight(0xfff1de, Math.PI);
    const s = this.sun;
    s.castShadow = true;
    s.shadow.mapSize.set(SHADOW_MAP, SHADOW_MAP);
    s.shadow.bias = -0.0003;
    s.shadow.normalBias = 0.04;
    s.shadow.camera.near = 1;
    s.shadow.camera.far = SUN_DIST * 2.2; // 低い太陽では影が非常に長く伸びる
    this.scene.add(s, s.target);

    // 太陽円盤 (視直径 0.53°)。地形/HLS に隠れれば見えない
    this.sunDisc = new THREE.Mesh(new THREE.SphereGeometry(SUN_DIST * Math.tan(rad(0.265)), 16, 8),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(400, 370, 320) }));
    this.scene.add(this.sunDisc);

    // 周囲からの反射光 (地形の相互反射/地球照) はごく弱い
    this.scene.add(new THREE.HemisphereLight(0x9aa3b5, 0x39383a, Math.PI * this.world.sun.ambient));
  }

  setSun(azRad, elRad) {
    this.sunDir.set(Math.cos(elRad) * Math.cos(azRad), Math.cos(elRad) * Math.sin(azRad), Math.sin(elRad));
    this.sunDisc.position.copy(this.sunDir).multiplyScalar(SUN_DIST * 0.95);
  }

  addStars() { // 装飾 (物理的な明るさではない)
    const n = 700, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1, a = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u);
      pos.set([s * Math.cos(a) * 1500, s * Math.sin(a) * 1500, Math.abs(u) * 1500 * 0.9 + 20], i * 3);
      col.fill(0.02 + Math.random() * 0.12, i * 3, i * 3 + 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    this.scene.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 1.3, sizeAttenuation: false, vertexColors: true })));
  }

  // ---------------------------------------------------------------- 月面
  /** 月面用の材質: Lommel-Seeliger 反射 + 3Dビューだけの補助光 (影の中の地形が見えるように。HLS/ローバーには付けない) */
  regolithMaterial(params) {
    const m = new THREE.MeshLambertMaterial(params);
    m.onBeforeCompile = (sh) => { sh.fragmentShader = sh.fragmentShader.replace('#include <lights_lambert_pars_fragment>', LS_CHUNK); };
    m.customProgramCacheKey = () => 'regolith-ls';
    m.emissive = new THREE.Color(0xffffff);
    m.emissiveMap = params.map || null;
    m.emissiveIntensity = 0.14 * this.world.ui.view_fill;
    return m;
  }

  addTerrain(heights) {
    const T = this.terrain;
    const geo = new THREE.PlaneGeometry(T.size, T.size, T.n - 1, T.n - 1);
    const pos = geo.attributes.position;
    this.vertexCell = new Int32Array(pos.count);  // 頂点 -> 高さマップ/影マップの添字
    this.vertexBase = new Float32Array(pos.count); // 影を掛ける前の明るさ (緩やかなムラ)
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i);
      const ix = Math.round((x + T.size / 2) / T.res), iy = Math.round((y + T.size / 2) / T.res);
      this.vertexCell[i] = iy * T.n + ix;
      pos.setZ(i, heights[this.vertexCell[i]]);
      this.vertexBase[i] = 0.86 + 0.14 * Math.sin(x * 0.11 + 1.3) * Math.sin(y * 0.087 + 0.4) + 0.05 * Math.sin(x * 0.43 + y * 0.31);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3));
    geo.computeVertexNormals();
    this.terrainGeo = geo;

    const terrain = new THREE.Mesh(geo, this.regolithMaterial({ map: regolithTexture(T.size / 10), vertexColors: true }));
    terrain.receiveShadow = true;
    terrain.castShadow = false; // 地形自身の影はサーバで焼き込む (低い太陽ではシャドウマップだと崩れるため)
    this.scene.add(terrain);

    // 地形の外側 (地平線まで続く平らな月面)
    const far = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000), this.regolithMaterial({ map: regolithTexture(600), color: 0x8a8a8a }));
    far.position.z = -0.12;
    far.receiveShadow = true;
    this.scene.add(far);
  }

  /** 地形自身の影 (0=影, 255=日照) をサーバから取得して頂点色に反映する */
  async loadShade() {
    const shade = new Uint8Array(await (await fetch(`/api/shade?${Date.now()}`)).arrayBuffer());
    const col = this.terrainGeo.attributes.color;
    for (let i = 0; i < this.vertexCell.length; i++) {
      const v = this.vertexBase[i] * (0.015 + 0.985 * (shade[this.vertexCell[i]] / 255));
      col.setXYZ(i, v, v, v * 0.98);
    }
    col.needsUpdate = true;
  }

  addRocks() { // サーバと同じ球 (見た目だけ少し歪ませる)
    const rocks = this.world.rocks;
    const g = new THREE.IcosahedronGeometry(1, 1);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
      const k = 1 + 0.10 * Math.sin(x * 5.1 + y * 3.7) * Math.cos(z * 4.3 + x * 2.1);
      p.setXYZ(i, x * k, y * k, z * k);
    }
    g.computeVertexNormals();
    const mesh = new THREE.InstancedMesh(g, this.regolithMaterial({ color: 0x8d8984 }), rocks.length);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler();
    rocks.forEach(([x, y, z, r], i) => {
      e.set(Math.random() * 6, Math.random() * 6, Math.random() * 6);
      mesh.setMatrixAt(i, m.compose(new THREE.Vector3(x, y, z), q.setFromEuler(e), new THREE.Vector3(r, r, r)));
    });
    mesh.castShadow = mesh.receiveShadow = true;
    this.scene.add(mesh);
  }

  // ---------------------------------------------------------------- HLS (円柱) / ローバー (箱)
  /** 影の中でも形が分かるよう、輪郭線を付ける */
  withOutline(mesh) {
    mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 30), new THREE.LineBasicMaterial({ color: 0x8fd0ff, toneMapped: false })));
    mesh.castShadow = mesh.receiveShadow = true;
    return mesh;
  }

  addHls() {
    const h = this.world.hls;
    const mesh = this.withOutline(new THREE.Mesh(new THREE.CylinderGeometry(h.radius, h.radius, h.height, 48),
      new THREE.MeshLambertMaterial({ color: h.color })));
    mesh.rotation.x = Math.PI / 2; // 円柱の軸 y -> z
    mesh.position.set(h.x, h.y, this.terrain.hls_z0 + h.height / 2);
    this.scene.add(mesh);
  }

  addRover() {
    const r = this.world.rover;
    this.rover = new THREE.Group();
    this.rover.rotation.order = 'ZYX'; // R = Rz(yaw) Ry(pitch) Rx(roll)
    this.scene.add(this.rover);
    // 箱の下面は地面から clearance。車体座標の原点は 地面 + wheel_radius
    const box = this.withOutline(new THREE.Mesh(new THREE.BoxGeometry(r.length, r.width, r.height),
      new THREE.MeshLambertMaterial({ color: r.color })));
    box.position.z = r.clearance - r.wheel_radius + r.height / 2;
    this.rover.add(box);
  }

  setRoverPose(x, y, z, yaw, pitch, roll) {
    this.rover.position.set(x, y, z);
    this.rover.rotation.set(roll, pitch, yaw);
    this.rover.updateMatrixWorld(true);
  }

  /** センサの取付位置・向き (車体座標) に置いた Group を返す。yaw は左が正、pitch は上が正 */
  mountGroup(m) {
    const g = new THREE.Group();
    g.position.set(m.x || 0, m.y || 0, m.z || 0);
    g.rotation.order = 'ZYX';
    g.rotation.set(0, -rad(m.pitch_deg || 0), rad(m.yaw_deg || 0));
    this.rover.add(g);
    return g;
  }

  // ---------------------------------------------------------------- 描画
  setExposure(v) { this.renderer.toneMappingExposure = v; }

  resize(w, h) { this.renderer.setSize(w, h, false); }

  render(camera) {
    this.updateShadowCamera();
    this.renderer.render(this.scene, camera);
  }

  /** ローバーとHLSの両方が影の範囲に入るよう、その中点を中心に影カメラを張る */
  updateShadowCamera() {
    const hls = this.world.hls, p = this.rover.position;
    const half = clamp(Math.hypot(p.x - hls.x, p.y - hls.y) / 2 + 35, 50, 120);
    const snap = (half * 2) / SHADOW_MAP; // 影のちらつき防止: 影マップの1画素単位で動かす
    const cx = Math.round(((p.x + hls.x) / 2) / snap) * snap, cy = Math.round(((p.y + hls.y) / 2) / snap) * snap;
    this.sun.target.position.set(cx, cy, 0);
    this.sun.position.copy(this.sunDir).multiplyScalar(SUN_DIST).add(this.sun.target.position);
    const c = this.sun.shadow.camera;
    c.left = -half; c.right = half; c.top = half; c.bottom = -half;
    c.updateProjectionMatrix();
    this.sun.target.updateMatrixWorld();
  }
}

/** レゴリスの質感テクスチャ (斑点を重ねて描く。継ぎ目なく繰り返せる) */
function regolithTexture(repeat) {
  const cv = document.createElement('canvas');
  cv.width = cv.height = 512;
  const g = cv.getContext('2d');
  g.fillStyle = '#8c8a86';
  g.fillRect(0, 0, 512, 512);
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const blots = (n, rmin, rmax, alpha) => {
    for (let i = 0; i < n; i++) {
      const v = 90 + rnd() * 90, x = rnd() * 512, y = rnd() * 512, r = rmin + rnd() * (rmax - rmin);
      g.fillStyle = `rgba(${v},${v - 2},${v - 6},${alpha})`;
      for (const dx of [-512, 0, 512]) for (const dy of [-512, 0, 512]) { g.beginPath(); g.arc(x + dx, y + dy, r, 0, 7); g.fill(); }
    }
  };
  blots(300, 20, 60, 0.08);
  blots(2500, 4, 14, 0.12);
  blots(9000, 0.6, 2.5, 0.35);
  const t = new THREE.CanvasTexture(cv);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeat, repeat);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

// ======================================================================
// カメラ画像: サーバがレイトレーシングしたHDR画像(線形, float16)に、実機センサ風の処理を加えて表示する
// ======================================================================

// three.js カメラ (-Z が視線, +Y が上) -> 車体座標 (+X が前, +Z が上)
const CAM_BASIS = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(
  0, 0, -1, 0,
  -1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 0, 1));

const CAMERA_HEADER_BYTES = 36; // type, カメラ番号, 幅, 高さ, 時刻, 撮影時のローバー姿勢 x6 (server.py と一致させる)

// float16 -> float32 変換表
const HALF = new Float32Array(65536);
for (let i = 0; i < 65536; i++) {
  const e = (i >> 10) & 31, m = i & 1023, sign = i & 0x8000 ? -1 : 1;
  HALF[i] = e === 0 ? sign * m * 2 ** -24 : e === 31 ? (m ? NaN : sign * Infinity) : sign * (1 + m / 1024) * 2 ** (e - 15);
}

const aces = (x) => Math.min(1, Math.max(0, (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)));
const srgb = (x) => (x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055);
const gauss = () => Math.random() + Math.random() + Math.random() - 1.5; // 標準偏差 0.5

class Cameras {
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
    const half = new Uint16Array(buf, CAMERA_HEADER_BYTES, w * h * 3);
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

// ======================================================================
// ライダーの点群: 車体の上から見た図 (左上のcanvas) と、右の3Dビュー上の点群。座標は車体座標
// ======================================================================

// サーバから届くバイナリ: 12バイトのヘッダ + (x, y, z, 距離) の float32 x 点数
const LIDAR_HEADER_BYTES = 12;
const STOPS = [[0.1, 0.2, 0.9], [0.0, 0.8, 0.9], [0.2, 0.9, 0.3], [1.0, 0.9, 0.1], [1.0, 0.25, 0.1]]; // 低 -> 高

/** 高さ z [m] -> 色 (out[o..o+2] に 0..1 の RGB を書く) */
function heightColor(z, out, o) {
  const f = clamp((z + 1.5) / 4.5, 0, 1) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(f)), k = f - i;
  for (let c = 0; c < 3; c++) out[o + c] = STOPS[i][c] * (1 - k) + STOPS[i + 1][c] * k;
}

class Lidars {
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
    const a = new Float32Array(buf, LIDAR_HEADER_BYTES, n * 4);
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

// ======================================================================
// キーボード操作 (W/S・A/D・矢印, Shift=低速, Space=停止, R=リセット, V=視点切替)
// ======================================================================

// - e.code で判定するので、キー配列や IME に依存しない。
// - キャプチャ段階で処理するので、スライダやボタンにフォーカスがあっても矢印キーや Space を奪われない。
const KEYS = { KeyW: 'fwd', ArrowUp: 'fwd', KeyS: 'back', ArrowDown: 'back', KeyA: 'left', ArrowLeft: 'left', KeyD: 'right', ArrowRight: 'right' };
const SLOW = 0.3;       // Shift 押下中の指令の倍率
const RESEND_MS = 50;   // 押している間の指令の再送間隔 (サーバは途切れると車両を止める)

/** @param send  サーバへメッセージを送る関数 */
function setupInput({ send, onToggleView }) {
  const pressed = new Set();
  let slow = false;

  function sendDrive() {
    const k = slow ? SLOW : 1;
    send({ t: 'drive', v: ((pressed.has('fwd') ? 1 : 0) - (pressed.has('back') ? 1 : 0)) * k,
                      w: ((pressed.has('left') ? 1 : 0) - (pressed.has('right') ? 1 : 0)) * k });
  }
  function press(action, on) {
    const changed = pressed.has(action) !== on;
    on ? pressed.add(action) : pressed.delete(action);
    if (changed) sendDrive(); // 押した/離した瞬間に送る
  }
  function stop() { pressed.clear(); sendDrive(); }

  addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    slow = e.shiftKey;
    const action = KEYS[e.code];
    if (action || e.code === 'Space') {
      e.preventDefault();
      e.stopPropagation();
      action ? press(action, true) : stop();
    } else if (!e.repeat && e.code === 'KeyR') send({ t: 'reset' });
    else if (!e.repeat && e.code === 'KeyV') onToggleView();
  }, true);

  addEventListener('keyup', (e) => {
    slow = e.shiftKey;
    if (KEYS[e.code]) { e.preventDefault(); press(KEYS[e.code], false); }
    if (e.code === 'Space') e.preventDefault(); // フォーカス中のボタンを click させない
  }, true);

  addEventListener('blur', stop);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  setInterval(() => { if (pressed.size) sendDrive(); }, RESEND_MS);

  // ボタン/スライダを操作したらフォーカスを外し、以後のキー入力が確実に届くようにする
  document.addEventListener('click', (e) => { if (e.target.matches('button')) e.target.blur(); });
  document.addEventListener('change', (e) => { if (e.target.matches('input')) e.target.blur(); });
  window.focus();
}

// ======================================================================
// 全体の組み立て: サーバとの通信, 画面の部品, 視点, メインループ
// ======================================================================

const MSG_LIDAR = 1, MSG_CAMERA = 2; // server.py のバイナリメッセージの種別

async function main() {
  const world = await (await fetch('/api/world')).json();
  const heights = new Float32Array(await (await fetch('/api/heightmap')).arrayBuffer());

  const view = new MoonView(world, heights, $('#view3d'));
  const cameras = new Cameras(world, view, $('#cams'));
  const lidars = new Lidars(world, view, $('#lidar-canvas'));
  await view.loadShade();

  const rover = { x: world.rover.start.x, y: world.rover.start.y, z: 0, yaw: rad(world.rover.start.yaw_deg), pitch: 0, roll: 0, v: 0, w: 0 };
  let roverUpdatedAt = performance.now();

  // ------------------------------------------------------------------ サーバとの通信
  let ws;
  const send = (msg) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg)); };

  function connect() {
    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => setConnection(true);
    ws.onclose = () => { setConnection(false); setTimeout(connect, 1000); };
    ws.onmessage = (ev) => (typeof ev.data === 'string' ? onText(JSON.parse(ev.data)) : onBinary(ev.data));
  }
  function setConnection(ok) {
    $('#conn').textContent = ok ? '接続中' : '切断 (再接続中…)';
    $('#conn').className = `conn ${ok ? 'ok' : 'off'}`;
  }
  function onText(m) {
    if (m.t === 'state') {
      Object.assign(rover, m);
      roverUpdatedAt = performance.now();
      updateHud(m);
    } else if (m.t === 'sun') {
      showSun(m.az_deg, m.el_deg);
      view.loadShade();
    }
  }
  function onBinary(buf) {
    const dv = new DataView(buf);
    const type = dv.getUint8(0);
    if (type === MSG_LIDAR) lidars.onFrame(buf, dv);
    else if (type === MSG_CAMERA) cameras.onFrame(buf, dv);
  }

  // ------------------------------------------------------------------ 画面の部品
  function updateHud(m) {
    $('#h-v').textContent = m.v.toFixed(2);
    $('#h-w').textContent = deg(m.w).toFixed(1);
    $('#h-pitch').textContent = deg(m.pitch).toFixed(1);
    $('#h-roll').textContent = deg(m.roll).toFixed(1);
    $('#h-odo').textContent = m.odo.toFixed(0);
  }

  // 太陽の方位/仰角: スライダ操作は即座に3Dビューへ反映し、サーバへは少し待って送る (影の再計算が重いため)
  const sunAz = $('#sun-az'), sunEl = $('#sun-el');
  function showSun(azDeg, elDeg) {
    sunAz.value = azDeg; sunEl.value = elDeg.toFixed(1);
    $('#sun-az-v').textContent = `${((azDeg % 360) + 360) % 360 | 0}°`;
    $('#sun-el-v').textContent = `${elDeg.toFixed(1)}°`;
    view.setSun(rad(azDeg), rad(elDeg));
  }
  let sunTimer = 0;
  const onSunSlider = () => {
    showSun(+sunAz.value, +sunEl.value);
    clearTimeout(sunTimer);
    sunTimer = setTimeout(() => send({ t: 'sun', az_deg: +sunAz.value, el_deg: +sunEl.value }), 250);
  };
  sunAz.oninput = sunEl.oninput = onSunSlider;
  showSun(deg(world.sun.az), deg(world.sun.el));

  // 3Dビューの明るさ (カメラ画像には影響しない)。スライダは 2 のべき乗
  let exposure = world.ui.exposure;
  try { exposure = parseFloat(localStorage.getItem('viewExposure')) || exposure; } catch (e) { /* 保存は任意 */ }
  const expSlider = $('#view-exp');
  const showExposure = () => { view.setExposure(exposure); $('#view-exp-v').textContent = `${exposure.toFixed(1)}×`; };
  expSlider.value = Math.log2(exposure);
  expSlider.oninput = () => {
    exposure = 2 ** +expSlider.value;
    showExposure();
    try { localStorage.setItem('viewExposure', String(exposure)); } catch (e) { /* 保存は任意 */ }
  };
  showExposure();

  $('#c-points').onchange = (e) => lidars.setPointsVisible(e.target.checked);
  $('#c-dist').onchange = (e) => { cameras.distanceOverlay = e.target.checked; };

  // ------------------------------------------------------------------ 視点 (追従 / 俯瞰。ドラッグで回転、ホイールでズーム)
  const views = [
    { az: Math.PI, el: 0.32, dist: 20, follow: true },   // ローバーの後ろから追従
    { az: -2.2, el: 0.5, dist: 100, follow: false },     // HLS を中心に俯瞰
  ];
  let current = 0;
  const camera = new THREE.PerspectiveCamera(55, 1, 0.2, 5000);
  camera.up.set(0, 0, 1);

  function updateCamera() {
    const v = views[current], hls = world.hls;
    const target = v.follow ? view.rover.position.clone().add(new THREE.Vector3(0, 0, 1.5)) : new THREE.Vector3(hls.x, hls.y, 6);
    const az = v.az + (v.follow ? rover.yaw : 0); // 追従のときは車体の向きに合わせて回る
    camera.position.set(target.x + v.dist * Math.cos(v.el) * Math.cos(az),
                        target.y + v.dist * Math.cos(v.el) * Math.sin(az),
                        Math.max(0.8, target.z + v.dist * Math.sin(v.el)));
    camera.lookAt(target);
  }

  const dom = view.renderer.domElement;
  let drag = null;
  dom.addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY]; dom.setPointerCapture(e.pointerId); });
  dom.addEventListener('pointerup', () => { drag = null; });
  dom.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const v = views[current];
    v.az -= (e.clientX - drag[0]) * 0.006;
    v.el = clamp(v.el + (e.clientY - drag[1]) * 0.006, 0.02, 1.5);
    drag = [e.clientX, e.clientY];
  });
  dom.addEventListener('wheel', (e) => {
    e.preventDefault();
    views[current].dist = clamp(views[current].dist * Math.exp(e.deltaY * 0.001), 4, 400);
  }, { passive: false });

  new ResizeObserver(() => {
    const el = $('#view3d');
    view.resize(el.clientWidth, el.clientHeight);
    camera.aspect = el.clientWidth / Math.max(1, el.clientHeight);
    camera.updateProjectionMatrix();
  }).observe($('#view3d'));

  setupInput({ send, onToggleView: () => { current = (current + 1) % views.length; } });

  // ------------------------------------------------------------------ メインループ
  function frame(now) {
    // 状態は 30Hz で届くので、その間はローバーの速度で外挿して滑らかに見せる
    const dt = Math.min((now - roverUpdatedAt) / 1000, 0.15);
    const yaw = rover.yaw + rover.w * dt;
    view.setRoverPose(rover.x + rover.v * Math.cos(yaw) * dt, rover.y + rover.v * Math.sin(yaw) * dt, rover.z, yaw, rover.pitch, rover.roll);
    updateCamera();
    view.render(camera);
    lidars.draw();
    requestAnimationFrame(frame);
  }
  connect();
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  document.body.insertAdjacentHTML('beforeend', `<pre style="color:#f88;position:fixed;top:0;left:0;z-index:9">${e.stack || e}</pre>`);
});
