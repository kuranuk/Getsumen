// 右側の3Dビュー (three.js): 月面・岩・HLS・ローバー・太陽。
// 世界座標・車体座標とも z-up (x=前/東, y=左/北)。
import * as THREE from './vendor/three.module.min.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

export const rad = (d) => (d * Math.PI) / 180;
export const deg = (r) => (r * 180) / Math.PI;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const SUN_DIST = 1500;
const SHADOW_MAP = 2048;

// 月面の反射: Lommel-Seeliger 則 (L ∝ μ0 / (μ0 + μ))。Lambert (L ∝ μ0) だと、太陽が地平線近くの平地が真っ暗になる。
// 2倍して、垂直入射で Lambert と同じ明るさに正規化する。
const LS_CHUNK = THREE.ShaderChunk.lights_lambert_pars_fragment.replace(
  'vec3 irradiance = dotNL * directLight.color;',
  'float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );\n\tvec3 irradiance = ( 2.0 * dotNL / ( dotNL + dotNV + 0.03 ) ) * directLight.color;');

export class MoonView {
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
