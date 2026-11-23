import * as THREE from './vendor/three.module.min.js';

// 世界座標・車体座標とも z-up (x=前/東, y=左/北) に統一する
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const $ = (s) => document.querySelector(s);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const deg = (r) => (r * 180) / Math.PI;
const rad = (d) => (d * Math.PI) / 180;

// 高さ -> 色 (点群の色付け)
function heightColor(z, out, o) {
  const t = clamp((z + 1.5) / 4.5, 0, 1);
  const stops = [[0.1, 0.2, 0.9], [0.0, 0.8, 0.9], [0.2, 0.9, 0.3], [1.0, 0.9, 0.1], [1.0, 0.25, 0.1]];
  const f = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), k = f - i;
  for (let c = 0; c < 3; c++) out[o + c] = stops[i][c] * (1 - k) + stops[i + 1][c] * k;
}

async function main() {
  const world = await (await fetch('/api/world')).json();
  const H = new Float32Array(await (await fetch('/api/heightmap')).arrayBuffer());
  const T = world.terrain, hls = world.hls, rcfg = world.rover;

  // ------------------------------------------------------------------ シーン
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.shadowMap.autoUpdate = false; // 1フレームに1回だけ影を更新 (カメラ画像でも共用)
  $('#view3d').appendChild(renderer.domElement);

  // ---- 光源 (太陽 + ごく弱い環境光) ----
  const sunDir = new THREE.Vector3(
    Math.cos(T.sun_elevation) * Math.cos(T.sun_azimuth),
    Math.cos(T.sun_elevation) * Math.sin(T.sun_azimuth),
    Math.sin(T.sun_elevation));
  const sun = new THREE.DirectionalLight(0xfff4e6, 3.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.05;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 600;
  scene.add(sun, sun.target);
  scene.add(new THREE.HemisphereLight(0x8a93a8, 0x2a2a2a, 0.35));

  // ---- 星 ----
  {
    const n = 1100, p = new Float32Array(n * 3), c = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1, a = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u);
      p.set([s * Math.cos(a) * 1500, s * Math.sin(a) * 1500, Math.abs(u) * 1500 * 0.9 + 20], i * 3);
      const b = 0.4 + Math.random() * 0.6; c.set([b, b, b], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    g.setAttribute('color', new THREE.BufferAttribute(c, 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 1.3, sizeAttenuation: false, vertexColors: true, fog: false })));
  }

  // ---- レゴリス テクスチャ ----
  function regolithTexture(repeat) {
    const cv = document.createElement('canvas'); cv.width = cv.height = 512;
    const g = cv.getContext('2d');
    g.fillStyle = '#8c8a86'; g.fillRect(0, 0, 512, 512);
    let seed = 12345; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    const blot = (n, rmin, rmax, a) => {
      for (let i = 0; i < n; i++) {
        const v = 90 + rnd() * 90, x = rnd() * 512, y = rnd() * 512, r = rmin + rnd() * (rmax - rmin);
        g.fillStyle = `rgba(${v},${v - 2},${v - 6},${a})`;
        for (const dx of [-512, 0, 512]) for (const dy of [-512, 0, 512]) { g.beginPath(); g.arc(x + dx, y + dy, r, 0, 7); g.fill(); }
      }
    };
    blot(300, 20, 60, 0.08); blot(2500, 4, 14, 0.12); blot(9000, 0.6, 2.5, 0.35);
    const t = new THREE.CanvasTexture(cv);
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat, repeat);
    t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
    return t;
  }

  // ---- 地形 ----
  {
    const geo = new THREE.PlaneGeometry(T.size, T.size, T.n - 1, T.n - 1);
    const pos = geo.attributes.position;
    const col = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i);
      const ix = Math.round((x + T.size / 2) / T.res), iy = Math.round((y + T.size / 2) / T.res);
      pos.setZ(i, H[iy * T.n + ix]);
      const v = 0.86 + 0.14 * Math.sin(x * 0.11 + 1.3) * Math.sin(y * 0.087 + 0.4) + 0.05 * Math.sin(x * 0.43 + y * 0.31);
      col.set([v, v, v * 0.98], i * 3);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.computeVertexNormals();
    const mat = new THREE.MeshLambertMaterial({ map: regolithTexture(T.size / 10), vertexColors: true });
    const terrain = new THREE.Mesh(geo, mat);
    terrain.receiveShadow = true;
    scene.add(terrain);

    // 地形の外側 (地平線まで連続した月面)
    const far = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000),
      new THREE.MeshLambertMaterial({ map: regolithTexture(600), color: 0xdddddd }));
    far.position.z = -0.12; far.receiveShadow = true;
    scene.add(far);
  }

  // ---- 岩 (サーバと同じ球で当たり判定。見た目だけ少し歪ませる) ----
  {
    const g = new THREE.IcosahedronGeometry(1, 1);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
      const k = 1 + 0.10 * Math.sin(x * 5.1 + y * 3.7) * Math.cos(z * 4.3 + x * 2.1);
      p.setXYZ(i, x * k, y * k, z * k);
    }
    g.computeVertexNormals();
    const mesh = new THREE.InstancedMesh(g, new THREE.MeshLambertMaterial({ color: 0x7d7a76 }), world.rocks.length);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler();
    world.rocks.forEach(([x, y, z, r], i) => {
      e.set(Math.random() * 6, Math.random() * 6, Math.random() * 6);
      m.compose(new THREE.Vector3(x, y, z), q.setFromEuler(e), new THREE.Vector3(r, r, r));
      mesh.setMatrixAt(i, m);
    });
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
  }

  // ---- HLS (円筒近似 + ノーズ + ハッチ) ----
  {
    const g = new THREE.Group();
    g.position.set(hls.x, hls.y, T.hls_z0);
    const metal = new THREE.MeshLambertMaterial({ color: 0xd8dade });
    const body = new THREE.Mesh(new THREE.CylinderGeometry(hls.radius, hls.radius, hls.height, 48), metal);
    body.rotation.x = Math.PI / 2; body.position.z = hls.height / 2;
    const nose = new THREE.Mesh(new THREE.CylinderGeometry(hls.radius * 0.25, hls.radius, hls.radius * 2.2, 48), metal);
    nose.rotation.x = Math.PI / 2; nose.position.z = hls.height + hls.radius * 1.1;
    const skirt = new THREE.Mesh(new THREE.CylinderGeometry(hls.radius * 1.02, hls.radius * 1.02, 0.6, 48),
      new THREE.MeshLambertMaterial({ color: 0x3a3a3c }));
    skirt.rotation.x = Math.PI / 2; skirt.position.z = 0.3;
    g.add(body, nose, skirt);

    // 乗降ハッチ
    const a = rad(hls.hatch_azimuth_deg);
    const out = new THREE.Vector3(Math.cos(a), Math.sin(a), 0);
    const tan = new THREE.Vector3(-Math.sin(a), Math.cos(a), 0);
    const basis = new THREE.Matrix4().makeBasis(tan, new THREE.Vector3(0, 0, 1), out);
    const hatch = new THREE.Group();
    hatch.quaternion.setFromRotationMatrix(basis);
    hatch.position.copy(out).multiplyScalar(hls.radius + 0.03).setZ(hls.hatch_height);
    const frame = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 3.6), new THREE.MeshBasicMaterial({ color: 0xff8c1a }));
    const door = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 3.2), new THREE.MeshBasicMaterial({ color: 0x15171a }));
    door.position.z = 0.01;
    hatch.add(frame, door);
    g.add(hatch);
    g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    scene.add(g);

    // パッドの焼け跡 (見た目のみ)
    const pad = new THREE.Mesh(new THREE.CircleGeometry(hls.pad_radius * 0.55, 64),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.18, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2 }));
    pad.position.set(hls.x, hls.y, T.hls_z0 + 0.02);
    scene.add(pad);
  }

  // ------------------------------------------------------------------ ローバー
  const rover = new THREE.Group();
  rover.rotation.order = 'ZYX'; // R = Rz(yaw) Ry(pitch) Rx(roll)
  scene.add(rover);
  const wheels = [];
  {
    const L = rcfg.length, W = rcfg.width, wr = rcfg.wheel_radius;
    const white = new THREE.MeshLambertMaterial({ color: 0xe9eaec });
    const grey = new THREE.MeshLambertMaterial({ color: 0x55595f });
    const dark = new THREE.MeshLambertMaterial({ color: 0x14202b });
    const orange = new THREE.MeshLambertMaterial({ color: 0xe8741c });
    const add = (geo, mat, x, y, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); rover.add(m); return m; };
    add(new THREE.BoxGeometry(L, W - 1.0, 1.0), grey, 0, 0, 0.45);                   // シャシ
    add(new THREE.BoxGeometry(L * 0.76, W - 1.0, 1.5), white, -L * 0.02, 0, 1.7);    // 与圧キャビン
    add(new THREE.BoxGeometry(0.1, W - 1.5, 0.8), dark, L * 0.76 / 2 - L * 0.02 + 0.03, 0, 2.0); // 前面窓
    add(new THREE.BoxGeometry(L * 0.76, W - 0.95, 0.12), orange, -L * 0.02, 0, 1.2);  // ストライプ
    add(new THREE.CylinderGeometry(0.04, 0.04, 1.4, 8), grey, -L * 0.3, 0.6, 3.1).rotation.x = Math.PI / 2; // アンテナ
    add(new THREE.CylinderGeometry(0.6, 0.6, 0.12, 24), white, -L * 0.25, -0.5, 2.5).rotation.x = Math.PI / 2; // 皿
    const wgeo = new THREE.CylinderGeometry(wr, wr, 0.7, 20); // 軸 = y
    const tread = new THREE.CylinderGeometry(wr * 0.45, wr * 0.45, 0.72, 12);
    for (const sx of [0.36, 0, -0.36]) for (const sy of [1, -1]) {
      const w = add(wgeo, new THREE.MeshLambertMaterial({ color: 0x2c2f33 }), sx * L, sy * (W / 2 - 0.35), 0);
      const hub = new THREE.Mesh(tread, new THREE.MeshLambertMaterial({ color: 0xb8bcc2 }));
      w.add(hub); wheels.push(w);
    }
    rover.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  }

  // ---- センサ取り付け (設定どおりの位置・向き) ----
  // カメラ座標(-Z視線,+Y上) -> 車体座標(+X前,+Z上): 列=(右=-y, 上=+z, 後=-x)
  const camBasis = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(
    0, 0, -1, 0,
    -1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1));
  function mountGroup(m) {
    const g = new THREE.Group();
    g.position.set(m.x || 0, m.y || 0, m.z || 0);
    g.rotation.order = 'ZYX';
    g.rotation.set(0, -rad(m.pitch_deg || 0), rad(m.yaw_deg || 0));
    rover.add(g);
    return g;
  }

  const cams = world.cameras.map((c) => {
    const mg = mountGroup(c.mount);
    const aspect = c.width / c.height;
    const vfov = deg(2 * Math.atan(Math.tan(rad(c.fov_deg) / 2) / aspect));
    const cam = new THREE.PerspectiveCamera(vfov, aspect, 0.1, 4000);
    cam.quaternion.copy(camBasis);
    mg.add(cam);
    const marker = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 0.18), new THREE.MeshBasicMaterial({ color: 0x111111 }));
    mg.add(marker);

    // 画面側の要素
    const el = document.createElement('div'); el.className = 'cam';
    el.innerHTML = `<header><span>${c.label}</span><span class="fps"></span></header>`;
    const canvas = document.createElement('canvas'); canvas.width = c.width; canvas.height = c.height;
    el.appendChild(canvas); $('#cams').appendChild(el);
    const rt = new THREE.WebGLRenderTarget(c.width, c.height, { samples: 4 });
    rt.texture.colorSpace = THREE.SRGBColorSpace;
    return {
      cfg: c, cam, canvas, ctx: canvas.getContext('2d'), rt,
      buf: new Uint8Array(c.width * c.height * 4), img: new ImageData(c.width, c.height),
      period: 1000 / (c.rate_hz || 10), last: -1e9, fpsEl: el.querySelector('.fps'), n: 0, t0: performance.now(),
    };
  });

  // ---- ライダー: 3D 点群表示 + 取付マーカ ----
  const lidars = world.lidars.map((c, idx) => {
    const mg = mountGroup(c.mount);
    mg.add(new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.5, 0.25), new THREE.MeshBasicMaterial({ color: 0xffd23a })));
    const maxN = c.h_pixels * c.v_pixels;
    const geo = new THREE.BufferGeometry();
    const posA = new THREE.BufferAttribute(new Float32Array(maxN * 3), 3);
    const colA = new THREE.BufferAttribute(new Float32Array(maxN * 3), 3);
    geo.setAttribute('position', posA); geo.setAttribute('color', colA);
    geo.setDrawRange(0, 0);
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.12, vertexColors: true, sizeAttenuation: true }));
    pts.frustumCulled = false;
    rover.add(pts); // 点群は車体座標系
    return { cfg: c, idx, pts, posA, colA, frame: null };
  });
  $('#lidar-title').textContent = world.lidars.length ? `LiDAR 点群 (車両上方から) — ${world.lidars.map((l) => l.label).join(' / ')}` : 'LiDAR なし';
  $('#c-points').addEventListener('change', (e) => lidars.forEach((l) => (l.pts.visible = e.target.checked)));

  // ------------------------------------------------------------------ 状態
  const st = { x: rcfg.start.x, y: rcfg.start.y, z: 0, yaw: rad(rcfg.start.yaw_deg), pitch: 0, roll: 0, v: 0, w: 0, odo: 0, auto: false, got: 0 };
  let lastStateTs = performance.now();

  function applyPose(now) {
    // 状態更新間隔の間は速度で外挿して滑らかに見せる
    const dt = Math.min((now - lastStateTs) / 1000, 0.15);
    const yaw = st.yaw + st.w * dt;
    rover.position.set(st.x + st.v * Math.cos(yaw) * dt, st.y + st.v * Math.sin(yaw) * dt, st.z);
    rover.rotation.set(st.roll, st.pitch, yaw);
    for (const w of wheels) w.rotation.y = (w.rotation.y + (st.v * (1 / 60)) / rcfg.wheel_radius) % (Math.PI * 2);
    rover.updateMatrixWorld(true);
  }

  // ------------------------------------------------------------------ WebSocket
  let ws, sendTimer = 0;
  function connect() {
    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { $('#conn').textContent = '接続中'; $('#conn').className = 'conn ok'; };
    ws.onclose = () => { $('#conn').textContent = '切断 (再接続中…)'; $('#conn').className = 'conn off'; setTimeout(connect, 1000); };
    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        const m = JSON.parse(ev.data);
        if (m.t === 'state') {
          Object.assign(st, m); lastStateTs = performance.now(); st.got++;
          $('#b-auto').classList.toggle('on', !!m.auto);
          updateHud(m);
        }
      } else onBinary(ev.data);
    };
  }
  const send = (o) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); };

  function onBinary(buf) {
    const dv = new DataView(buf);
    if (dv.getUint8(0) !== 1) return;
    const idx = dv.getUint8(1), n = dv.getUint32(4, true);
    const l = lidars[idx]; if (!l) return;
    const a = new Float32Array(buf, 12, n * 4);
    l.frame = { a, n, t: dv.getFloat32(8, true), rx: performance.now() };
    const p = l.posA.array, c = l.colA.array;
    for (let i = 0; i < n; i++) {
      p[i * 3] = a[i * 4]; p[i * 3 + 1] = a[i * 4 + 1]; p[i * 3 + 2] = a[i * 4 + 2];
      heightColor(a[i * 4 + 2], c, i * 3);
    }
    l.posA.needsUpdate = l.colA.needsUpdate = true;
    l.pts.geometry.setDrawRange(0, n);
  }

  function updateHud(m) {
    $('#h-v').textContent = m.v.toFixed(2);
    $('#h-w').textContent = deg(m.w).toFixed(1);
    $('#h-gap').textContent = Math.max(0, m.hls_gap).toFixed(1);
    $('#h-brg').textContent = deg(m.hls_bearing).toFixed(0);
    $('#h-pitch').textContent = deg(m.pitch).toFixed(1);
    $('#h-roll').textContent = deg(m.roll).toFixed(1);
    $('#h-odo').textContent = m.odo.toFixed(0);
  }

  // ------------------------------------------------------------------ 入力
  const keys = new Set();
  addEventListener('keydown', (e) => {
    if (e.repeat) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    keys.add(k);
    if (k === 'r') send({ t: 'reset' });
    if (k === 'v') cycleView();
    if (k === ' ') e.preventDefault();
  });
  addEventListener('keyup', (e) => keys.delete(e.key.length === 1 ? e.key.toLowerCase() : e.key));
  addEventListener('blur', () => keys.clear());
  setInterval(() => {
    const f = (keys.has('w') || keys.has('ArrowUp') ? 1 : 0) - (keys.has('s') || keys.has('ArrowDown') ? 1 : 0);
    const t = (keys.has('a') || keys.has('ArrowLeft') ? 1 : 0) - (keys.has('d') || keys.has('ArrowRight') ? 1 : 0);
    const k = keys.has('Shift') ? 0.3 : 1.0;
    const manual = f !== 0 || t !== 0 || keys.has(' ');
    if (manual || sendTimer++ % 10 === 0) send({ t: 'drive', v: f * k, w: t * k });
  }, 50);
  $('#b-auto').onclick = () => send({ t: 'auto', on: !st.auto });
  $('#b-reset').onclick = () => send({ t: 'reset' });
  $('#b-view').onclick = () => cycleView();

  // ---- 視点 (追従 / 俯瞰) ----
  const views = [
    { name: '追従', az: Math.PI, el: 0.32, dist: 20, follow: true },
    { name: '俯瞰', az: -2.2, el: 0.5, dist: 100, follow: false },
  ];
  let vi = 0;
  function cycleView() { vi = (vi + 1) % views.length; $('#b-view').textContent = `視点: ${views[vi].name}`; }
  const dom = renderer.domElement;
  let drag = null;
  dom.addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY]; dom.setPointerCapture(e.pointerId); });
  dom.addEventListener('pointerup', () => (drag = null));
  dom.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const v = views[vi];
    v.az -= (e.clientX - drag[0]) * 0.006;
    v.el = clamp(v.el + (e.clientY - drag[1]) * 0.006, 0.02, 1.5);
    drag = [e.clientX, e.clientY];
  });
  dom.addEventListener('wheel', (e) => { e.preventDefault(); const v = views[vi]; v.dist = clamp(v.dist * Math.exp(e.deltaY * 0.001), 4, 400); }, { passive: false });

  const mainCam = new THREE.PerspectiveCamera(55, 1, 0.2, 5000);
  mainCam.up.set(0, 0, 1);
  function updateMainCamera() {
    const v = views[vi];
    const target = v.follow ? rover.position.clone().add(new THREE.Vector3(0, 0, 1.5)) : new THREE.Vector3(hls.x, hls.y, 6);
    const az = v.az + (v.follow ? st.yaw : 0);
    mainCam.position.set(target.x + v.dist * Math.cos(v.el) * Math.cos(az), target.y + v.dist * Math.cos(v.el) * Math.sin(az), target.z + v.dist * Math.sin(v.el));
    const ground = H.length ? 0 : 0; // (地面貫通は許容)
    mainCam.position.z = Math.max(mainCam.position.z, 0.8 + ground);
    mainCam.lookAt(target);
  }

  function updateSunShadow() {
    // ローバーとHLSの両方が影の範囲に入るよう、中点を中心に影カメラを張る
    const mx = (rover.position.x + hls.x) / 2, my = (rover.position.y + hls.y) / 2;
    const half = clamp(Math.hypot(rover.position.x - hls.x, rover.position.y - hls.y) / 2 + 35, 50, 120);
    const snap = (half * 2) / 2048;
    const cx = Math.round(mx / snap) * snap, cy = Math.round(my / snap) * snap;
    sun.target.position.set(cx, cy, 0);
    sun.position.copy(sunDir).multiplyScalar(250).add(sun.target.position);
    const s = sun.shadow.camera; s.left = -half; s.right = half; s.top = half; s.bottom = -half;
    s.updateProjectionMatrix();
    sun.target.updateMatrixWorld();
  }

  // ------------------------------------------------------------------ リサイズ
  function resize() {
    const el = $('#view3d'), w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h, false);
    mainCam.aspect = w / Math.max(1, h); mainCam.updateProjectionMatrix();
    const lc = $('#lidar-canvas'), dpr = Math.min(devicePixelRatio, 2);
    lc.width = Math.max(1, Math.floor(lc.clientWidth * dpr)); lc.height = Math.max(1, Math.floor(lc.clientHeight * dpr));
  }
  new ResizeObserver(resize).observe($('#view3d'));
  new ResizeObserver(resize).observe($('#lidar-panel'));

  // ------------------------------------------------------------------ ライダー俯瞰図 (車体前方が上)
  let viewRange = world.ui.lidar_view_range || 40;
  $('#lidar-canvas').addEventListener('wheel', (e) => { e.preventDefault(); viewRange = clamp(viewRange * Math.exp(e.deltaY * 0.001), 5, 120); }, { passive: false });
  const lctx = $('#lidar-canvas').getContext('2d');

  function drawLidar() {
    const cv = $('#lidar-canvas'), W = cv.width, Hh = cv.height, dpr = W / Math.max(1, cv.clientWidth);
    const cx = W / 2, cy = Hh * 0.62; // 自車を少し下に置き前方を広く表示
    const scale = Math.min(W / 2, Math.min(cy, Hh - cy) * 1.0 + Hh * 0.2) / viewRange;
    const sx = (y) => cx - y * scale, sy = (x) => cy - x * scale; // 左(+y)=画面左, 前(+x)=画面上
    lctx.fillStyle = '#05070a'; lctx.fillRect(0, 0, W, Hh);

    // 距離リング
    const step = viewRange > 60 ? 20 : viewRange > 25 ? 10 : 5;
    lctx.lineWidth = 1 * dpr; lctx.font = `${11 * dpr}px system-ui`; lctx.textAlign = 'left';
    for (let r = step; r <= viewRange * 1.8; r += step) {
      lctx.strokeStyle = '#1b232c'; lctx.beginPath(); lctx.arc(cx, cy, r * scale, 0, Math.PI * 2); lctx.stroke();
      lctx.fillStyle = '#4b5968'; lctx.fillText(`${r}m`, cx + 3 * dpr, cy - r * scale - 2 * dpr);
    }
    lctx.strokeStyle = '#1b232c'; lctx.beginPath(); lctx.moveTo(cx, 0); lctx.lineTo(cx, Hh); lctx.moveTo(0, cy); lctx.lineTo(W, cy); lctx.stroke();

    // 視野 (ライダー FOV)
    for (const l of lidars) {
      const c = l.cfg, m = c.mount, y0 = rad(m.yaw_deg || 0), hf = rad(c.h_fov_deg) / 2;
      const ox = m.x || 0, oy = m.y || 0;
      lctx.strokeStyle = '#ffd23a55'; lctx.fillStyle = '#ffd23a0d'; lctx.beginPath(); lctx.moveTo(sx(oy), sy(ox));
      for (const a of [y0 + hf, y0 - hf]) lctx.lineTo(sx(oy + c.max_range * Math.sin(a)), sy(ox + c.max_range * Math.cos(a)));
      lctx.closePath(); lctx.fill(); lctx.stroke();
    }

    // HLS 位置マーカ (範囲外なら表示端に方位だけ示す)
    const b = st.hls_bearing || 0;
    const dist = (st.hls_gap ?? 0) + hls.radius + rcfg.length / 2;
    const rMax = Math.min(cx, W - cx, cy, Hh - cy) * 0.92;
    const rr = Math.min(dist * scale, rMax), inside = dist * scale <= rMax;
    const hx = cx + (-Math.sin(b)) * rr, hy = cy - Math.cos(b) * rr;
    lctx.strokeStyle = '#ff8c1a'; lctx.fillStyle = '#ff8c1a'; lctx.lineWidth = 2 * dpr;
    if (inside) { // 円筒の断面を円で表示
      lctx.beginPath(); lctx.arc(hx, hy, Math.max(3 * dpr, hls.radius * scale), 0, 7); lctx.globalAlpha = 0.25; lctx.fill(); lctx.globalAlpha = 1; lctx.stroke();
    } else {
      lctx.beginPath(); lctx.moveTo(cx + (-Math.sin(b)) * (rr - 24 * dpr), cy - Math.cos(b) * (rr - 24 * dpr)); lctx.lineTo(hx, hy); lctx.stroke();
      lctx.beginPath(); lctx.arc(hx, hy, 4 * dpr, 0, 7); lctx.fill();
    }
    lctx.fillStyle = '#ff8c1a'; lctx.font = `${11 * dpr}px system-ui`;
    lctx.fillText(`HLS ${dist.toFixed(0)}m`, hx + 8 * dpr, hy + 4 * dpr);

    // 点群
    let info = [];
    for (const l of lidars) {
      const f = l.frame; if (!f) continue;
      const a = f.a, sz = Math.max(1.5 * dpr, 0.12 * scale);
      const tmp = [0, 0, 0];
      for (let i = 0; i < f.n; i++) {
        const x = a[i * 4], y = a[i * 4 + 1], z = a[i * 4 + 2];
        const px = sx(y), py = sy(x);
        if (px < -4 || px > W + 4 || py < -4 || py > Hh + 4) continue;
        heightColor(z, tmp, 0);
        lctx.fillStyle = `rgb(${tmp[0] * 255 | 0},${tmp[1] * 255 | 0},${tmp[2] * 255 | 0})`;
        lctx.fillRect(px - sz / 2, py - sz / 2, sz, sz);
      }
      info.push(`${l.cfg.name}: ${f.n} pts`);
    }
    $('#lidar-info').textContent = `${info.join(' | ')}  表示半径 ${viewRange.toFixed(0)} m`;

    // 自車
    const L = rcfg.length * scale, Wd = rcfg.width * scale;
    lctx.fillStyle = '#e9eaec'; lctx.strokeStyle = '#4cc2ff'; lctx.lineWidth = 1.5 * dpr;
    lctx.fillRect(cx - Wd / 2, cy - L / 2, Wd, L); lctx.strokeRect(cx - Wd / 2, cy - L / 2, Wd, L);
    lctx.fillStyle = '#4cc2ff'; lctx.beginPath(); lctx.moveTo(cx, cy - L / 2 - 6 * dpr); lctx.lineTo(cx - 5 * dpr, cy - L / 2 + 2 * dpr); lctx.lineTo(cx + 5 * dpr, cy - L / 2 + 2 * dpr); lctx.fill();

  }

  // ------------------------------------------------------------------ カメラ画像 (RT -> 2D canvas)
  function renderCameras(now) {
    for (const l of lidars) l.pts.visible = false; // 点群をカメラ画像に写さない
    const prevAuto = renderer.shadowMap.autoUpdate;
    for (const c of cams) {
      if (now - c.last < c.period) continue;
      c.last = now;
      renderer.setRenderTarget(c.rt);
      renderer.render(scene, c.cam);
      renderer.readRenderTargetPixels(c.rt, 0, 0, c.cfg.width, c.cfg.height, c.buf);
      const w = c.cfg.width, h = c.cfg.height, d = c.img.data, s = c.buf;
      for (let y = 0; y < h; y++) d.set(s.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4); // 上下反転
      c.ctx.putImageData(c.img, 0, 0);
      c.n++;
      if (now - c.t0 > 1000) { c.fpsEl.textContent = `${(c.n * 1000 / (now - c.t0)).toFixed(0)} Hz`; c.n = 0; c.t0 = now; }
    }
    renderer.setRenderTarget(null);
    renderer.shadowMap.autoUpdate = prevAuto;
    const on = $('#c-points').checked;
    for (const l of lidars) l.pts.visible = on;
  }

  // ------------------------------------------------------------------ メインループ
  resize();
  connect();
  function frame(now) {
    applyPose(now);
    updateSunShadow();
    updateMainCamera();
    renderer.shadowMap.needsUpdate = true;
    renderer.render(scene, mainCam);
    renderCameras(now);
    drawLidar();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // デバッグ/自動テスト用
  window.__sim = { scene, st, cams, lidars, renderer };
}

main().catch((e) => { console.error(e); document.body.insertAdjacentHTML('beforeend', `<pre style="color:#f88;position:fixed;top:0;left:0;z-index:9">${e.stack || e}</pre>`); });
