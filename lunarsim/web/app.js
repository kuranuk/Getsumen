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
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  let viewExp = world.ui.exposure ?? 2.5;
  try { const v = parseFloat(localStorage.getItem('viewExposure')); if (v > 0) viewExp = v; } catch (e) { /* 保存は任意 */ }
  renderer.toneMappingExposure = viewExp;
  $('#view3d').appendChild(renderer.domElement);

  // ---- 太陽 (南極: 仰角 0〜数度の低い太陽。方位/仰角は UI から変更可) ----
  const sunDir = new THREE.Vector3(0, 0, 1);
  const SUN_DIST = 1500;
  const sun = new THREE.DirectionalLight(0xfff1de, Math.PI); // 垂直入射の白面が反射率どおりに見える強度
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0003;
  sun.shadow.normalBias = 0.04;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = SUN_DIST * 2.2; // 低い太陽では影の方向に非常に長く伸びる
  scene.add(sun, sun.target);
  // 太陽円盤 (視直径 0.53°, HDR 輝度)。地形/HLSに隠れれば見えない。
  const sunDisc = new THREE.Mesh(new THREE.SphereGeometry(SUN_DIST * Math.tan(rad(0.265)), 16, 8),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(400, 370, 320), fog: false }));
  scene.add(sunDisc);
  // 周囲からの反射光 (地形の相互反射/地球照) はごく弱い
  const hemi = new THREE.HemisphereLight(0x9aa3b5, 0x39383a, 0);
  const AMBIENT = Math.PI * (world.sun.ambient ?? 0.03);
  // 右の3Dビューだけに足す補助光。地形/岩の自己発光として足すので、HLSやローバーの影側は真っ黒のまま (カメラ画像にも影響しない)
  const FILL_RAD = 0.14 * (world.ui.view_fill ?? 0.12);
  const fillMats = [];
  const outlines = []; // 3Dビューだけに表示する輪郭線 (影の中でも形が分かるように。カメラ画像には写さない)
  const addOutline = (mesh, geo, angle = 30) => {
    const ol = new THREE.LineSegments(new THREE.EdgesGeometry(geo, angle), new THREE.LineBasicMaterial({ color: 0x8fd0ff, toneMapped: false }));
    mesh.add(ol); outlines.push(ol);
  };
  hemi.intensity = AMBIENT;
  scene.add(hemi);
  function setSun(azRad, elRad) {
    sunDir.set(Math.cos(elRad) * Math.cos(azRad), Math.cos(elRad) * Math.sin(azRad), Math.sin(elRad));
    sunDisc.position.copy(sunDir).multiplyScalar(SUN_DIST * 0.95);
    $('#sun-az-v').textContent = `${((deg(azRad) % 360 + 360) % 360).toFixed(0)}°`;
    $('#sun-el-v').textContent = `${deg(elRad).toFixed(1)}°`;
  }
  setSun(world.sun.az, world.sun.el);

  // ---- 月面の反射: Lommel-Seeliger 則 (レゴリスは太陽が低くても平地が暗くならず、逆光側で明るい) ----
  //   Lambert: L ∝ μ0        L-S: L ∝ μ0 / (μ0 + μ)   (μ0=入射, μ=出射の余弦; 2倍で垂直入射を Lambert と同じに正規化)
  const LS_CHUNK = THREE.ShaderChunk.lights_lambert_pars_fragment.replace(
    'vec3 irradiance = dotNL * directLight.color;',
    'float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );\n\tvec3 irradiance = ( 2.0 * dotNL / ( dotNL + dotNV + 0.03 ) ) * directLight.color;');
  function regolithMaterial(params) {
    const m = new THREE.MeshLambertMaterial(params);
    m.onBeforeCompile = (sh) => { sh.fragmentShader = sh.fragmentShader.replace('#include <lights_lambert_pars_fragment>', LS_CHUNK); };
    m.customProgramCacheKey = () => 'regolith-ls';
    m.emissive = new THREE.Color(0xffffff); m.emissiveMap = params.map || null; m.emissiveIntensity = 0; fillMats.push(m);
    return m;
  }

  // ---- 星 ----
  {
    const n = 700, p = new Float32Array(n * 3), c = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1, a = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u);
      p.set([s * Math.cos(a) * 1500, s * Math.sin(a) * 1500, Math.abs(u) * 1500 * 0.9 + 20], i * 3);
      const b = 0.02 + Math.random() * 0.12; c.set([b, b, b], i * 3); // HDR: 日向に合わせた露出ではほぼ写らない
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
  let shade = null, terrainGeo = null, vIndex = null, vBase = null;
  {
    const geo = new THREE.PlaneGeometry(T.size, T.size, T.n - 1, T.n - 1);
    const pos = geo.attributes.position;
    const col = new Float32Array(pos.count * 3);
    vIndex = new Int32Array(pos.count); vBase = new Float32Array(pos.count);
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i);
      const ix = Math.round((x + T.size / 2) / T.res), iy = Math.round((y + T.size / 2) / T.res);
      vIndex[i] = iy * T.n + ix;
      pos.setZ(i, H[vIndex[i]]);
      vBase[i] = 0.86 + 0.14 * Math.sin(x * 0.11 + 1.3) * Math.sin(y * 0.087 + 0.4) + 0.05 * Math.sin(x * 0.43 + y * 0.31);
      col.fill(vBase[i], i * 3, i * 3 + 3);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.computeVertexNormals();
    terrainGeo = geo;
    const terrain = new THREE.Mesh(geo, regolithMaterial({ map: regolithTexture(T.size / 10), vertexColors: true }));
    terrain.receiveShadow = true;
    terrain.castShadow = false; // 地形自身の影はサーバで焼き込み (低い太陽ではシャドウマップだと崩れるため)
    scene.add(terrain);

    // 地形の外側 (地平線まで連続した月面)
    const far = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000),
      regolithMaterial({ map: regolithTexture(600), color: 0x8a8a8a }));
    far.position.z = -0.12; far.receiveShadow = true;
    scene.add(far);
  }
  // 地形が落とす影 (0=影,255=日照) を頂点色へ反映
  function applyShade() {
    if (!shade) return;
    const col = terrainGeo.attributes.color, a = col.array;
    for (let i = 0; i < vIndex.length; i++) {
      const v = vBase[i] * (0.015 + 0.985 * (shade[vIndex[i]] / 255));
      a[i * 3] = a[i * 3 + 1] = v; a[i * 3 + 2] = v * 0.98;
    }
    col.needsUpdate = true;
  }
  async function loadShade() {
    shade = new Uint8Array(await (await fetch(`/api/shade?${Date.now()}`)).arrayBuffer());
    applyShade();
  }
  await loadShade();

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
    const mesh = new THREE.InstancedMesh(g, regolithMaterial({ color: 0x8d8984 }), world.rocks.length);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler();
    world.rocks.forEach(([x, y, z, r], i) => {
      e.set(Math.random() * 6, Math.random() * 6, Math.random() * 6);
      m.compose(new THREE.Vector3(x, y, z), q.setFromEuler(e), new THREE.Vector3(r, r, r));
      mesh.setMatrixAt(i, m);
    });
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
  }

  // ---- HLS (model: cylinder=単純な円柱 / detailed=ノーズ+ハッチ) ----
  if (hls.model !== 'detailed') {
    const g = new THREE.Mesh(new THREE.CylinderGeometry(hls.radius, hls.radius, hls.height, 48),
      new THREE.MeshLambertMaterial({ color: hls.color || '#d8dade' }));
    g.rotation.x = Math.PI / 2; g.position.set(hls.x, hls.y, T.hls_z0 + hls.height / 2);
    g.castShadow = g.receiveShadow = true;
    addOutline(g, g.geometry);
    scene.add(g);
  } else {
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
  if (rcfg.model !== 'detailed') {
    // 単純な箱: 下面は地面から clearance、車体座標の原点は 地面+wheel_radius
    const box = new THREE.Mesh(new THREE.BoxGeometry(rcfg.length, rcfg.width, rcfg.height),
      new THREE.MeshLambertMaterial({ color: rcfg.color || '#e9eaec' }));
    box.position.set(0, 0, rcfg.clearance - rcfg.wheel_radius + rcfg.height / 2);
    box.castShadow = box.receiveShadow = true;
    addOutline(box, box.geometry);
    rover.add(box);
  } else {
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
    el.appendChild(canvas);
    // 配置: cfg.grid = {row, col, span} (列数は ui.camera_grid_columns)。未指定なら自動配置
    if (c.grid) { el.style.gridRow = String(c.grid.row); el.style.gridColumn = `${c.grid.col} / span ${c.grid.span || 1}`; }
    $('#cams').style.setProperty('--cols', world.ui.camera_grid_columns || 4);
    $('#cams').appendChild(el);
    // HDR(線形・浮動小数点)で描画し、露出→ノイズ→トーンマップ→sRGB をセンサモデルとして自前で適用する
    const raytraced = (c.render || 'raytrace') !== 'raster'; // true: サーバのレイトレーシング画像を受信 / false: ブラウザで描画
    el.querySelector('header span').insertAdjacentHTML('beforeend', ` <small style="opacity:.7">${raytraced ? 'RT' : 'raster'}</small>`);
    const rt = raytraced ? null : new THREE.WebGLRenderTarget(c.width, c.height, { samples: 4, type: THREE.FloatType });
    canvas.style.filter = c.blur_px > 0 ? `blur(${c.blur_px}px)` : 'none';
    return {
      cfg: c, cam, canvas, ctx: canvas.getContext('2d'), rt, exp: 1.0, topDown: raytraced,
      buf: new Float32Array(c.width * c.height * 4), img: new ImageData(c.width, c.height),
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
  // ---- 3Dビューの明るさ ----
  {
    const sl = $('#view-exp'), val = $('#view-exp-v');
    sl.value = Math.log2(viewExp); val.textContent = `${viewExp.toFixed(1)}×`;
    sl.oninput = () => {
      viewExp = 2 ** (+sl.value); renderer.toneMappingExposure = viewExp; val.textContent = `${viewExp.toFixed(1)}×`;
      try { localStorage.setItem('viewExposure', String(viewExp)); } catch (e) { /* 保存は任意 */ }
    };
    sl.onchange = () => sl.blur();
  }

  // ---- 太陽の方位/仰角 UI ----
  {
    const az = $('#sun-az'), el = $('#sun-el');
    az.value = ((deg(world.sun.az) % 360) + 360) % 360; el.value = deg(world.sun.el).toFixed(1);
    let timer = 0;
    const push = () => {
      setSun(rad(+az.value), rad(+el.value));
      clearTimeout(timer);
      timer = setTimeout(() => send({ t: 'sun', az_deg: +az.value, el_deg: +el.value }), 250);
    };
    az.oninput = push; el.oninput = push;
  }
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
        if (m.t === 'sun') { setSun(m.az, m.el); $('#sun-az').value = m.az_deg; $('#sun-el').value = m.el_deg.toFixed(1); loadShade(); }
        else if (m.t === 'state') {
          Object.assign(st, m); lastStateTs = performance.now(); st.got++;
          $('#b-auto').classList.toggle('on', !!m.auto);
          updateHud(m);
        }
      } else onBinary(ev.data);
    };
  }
  const send = (o) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); };

  // float16 -> float32 変換表
  const HALF = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) {
    const e = (i >> 10) & 31, m = i & 1023, sgn = i & 0x8000 ? -1 : 1;
    HALF[i] = e === 0 ? sgn * m * 2 ** -24 : e === 31 ? (m ? NaN : sgn * Infinity) : sgn * (1 + m / 1024) * 2 ** (e - 15);
  }
  const _eul = new THREE.Euler(0, 0, 0, 'ZYX');
  function onCameraFrame(buf, dv) {
    const c = cams[dv.getUint8(1)]; if (!c) return;
    const w = dv.getUint16(2, true), h = dv.getUint16(4, true);
    if (w !== c.cfg.width || h !== c.cfg.height) return;
    const pose = [0, 1, 2, 3, 4, 5].map((i) => dv.getFloat32(12 + i * 4, true)); // x,y,z,yaw,pitch,roll (撮影時点)
    const hv = new Uint16Array(buf, 36, w * h * 3), f = c.buf;
    for (let i = 0, n = w * h; i < n; i++) {
      f[i * 4] = HALF[hv[i * 3]]; f[i * 4 + 1] = HALF[hv[i * 3 + 1]]; f[i * 4 + 2] = HALF[hv[i * 3 + 2]];
    }
    // 撮影時点のローバー姿勢に合わせて、グレア/距離目盛の投影を計算する (走行中でも画像と目盛が一致する)
    const keep = [rover.position.clone(), rover.rotation.clone()];
    rover.position.set(pose[0], pose[1], pose[2]); rover.rotation.set(pose[5], pose[4], pose[3], 'ZYX');
    rover.updateMatrixWorld(true);
    c.cam.matrixWorldInverse.copy(c.cam.matrixWorld).invert();
    sensorPipeline(c);
    rover.position.copy(keep[0]); rover.rotation.copy(keep[1]); rover.updateMatrixWorld(true);
    const now = performance.now();
    c.n++;
    if (now - c.t0 > 2000) { c.fpsEl.textContent = `${(c.n * 1000 / (now - c.t0)).toFixed(1)} Hz`; c.n = 0; c.t0 = now; }
  }

  function onBinary(buf) {
    const dv = new DataView(buf);
    if (dv.getUint8(0) === 2) return onCameraFrame(buf, dv);
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
  // キーボードは e.code で判定 (配列/IME に依存しない)。キャプチャ段階で処理し、
  // スライダ/ボタンにフォーカスがあっても矢印キー・Space を奪われない。
  const KEYMAP = { KeyW: 'fwd', ArrowUp: 'fwd', KeyS: 'back', ArrowDown: 'back', KeyA: 'left', ArrowLeft: 'left', KeyD: 'right', ArrowRight: 'right' };
  const act = new Set();       // 押されている操作: fwd/back/left/right (キーボードと画面パッドで共用)
  let slow = false;
  function sendDrive() {
    const f = (act.has('fwd') ? 1 : 0) - (act.has('back') ? 1 : 0);
    const t = (act.has('left') ? 1 : 0) - (act.has('right') ? 1 : 0);
    const k = slow || $('#pad-slow').checked ? 0.3 : 1.0;
    send({ t: 'drive', v: f * k, w: t * k });
    return f !== 0 || t !== 0;
  }
  function setAct(a, on) {
    const had = act.has(a);
    on ? act.add(a) : act.delete(a);
    if (had !== on) sendDrive(); // 押した/離した瞬間に即送信 (50ms周期を待たない)
    document.querySelectorAll('#pad [data-act]').forEach((b) => b.classList.toggle('on', act.has(b.dataset.act)));
  }
  function stopAll() { act.clear(); setAct('', false); send({ t: 'drive', v: 0, w: 0 }); }
  addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest && e.target.closest('input[type=text], textarea, select')) return;
    slow = e.shiftKey;
    const a = KEYMAP[e.code];
    if (a) { e.preventDefault(); e.stopPropagation(); setAct(a, true); return; }
    if (e.code === 'Space') { e.preventDefault(); e.stopPropagation(); send({ t: 'auto', on: false }); stopAll(); return; }
    if (e.repeat) return;
    if (e.code === 'KeyR') send({ t: 'reset' });
    else if (e.code === 'KeyV') cycleView();
  }, true);
  addEventListener('keyup', (e) => {
    slow = e.shiftKey;
    const a = KEYMAP[e.code];
    if (a) { e.preventDefault(); setAct(a, false); }
    if (e.code === 'Space') e.preventDefault(); // ボタンにフォーカスがあっても click させない
  }, true);
  addEventListener('blur', stopAll);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopAll(); });
  // ボタン/スライダ操作後はフォーカスを外し、以後のキー入力が確実に操縦に届くようにする
  document.addEventListener('click', (e) => { if (e.target.matches && e.target.matches('button')) e.target.blur(); });
  document.addEventListener('change', (e) => { if (e.target.matches && e.target.matches('input')) e.target.blur(); });
  // 画面パッド (マウス/タッチで押している間だけ走行)
  document.querySelectorAll('#pad [data-act]').forEach((b) => {
    const a = b.dataset.act;
    b.addEventListener('pointerdown', (e) => { e.preventDefault(); b.setPointerCapture(e.pointerId); setAct(a, true); });
    for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) b.addEventListener(ev, () => setAct(a, false));
  });
  $('#pad-slow').addEventListener('change', (e) => e.target.blur());
  window.focus();
  setInterval(() => { // 押している間は 50ms ごとに指令を再送 (途絶えると車両は止まる)
    const moving = act.size > 0;
    if (moving || sendTimer++ % 10 === 0) sendDrive();
  }, 50);
  $('#b-auto').onclick = () => send({ t: 'auto', on: !st.auto });

  // ---- 視点 (追従 / 俯瞰) ----
  const views = [
    { name: '追従', az: Math.PI, el: 0.32, dist: 20, follow: true },
    { name: '俯瞰', az: -2.2, el: 0.5, dist: 100, follow: false },
  ];
  let vi = 0;
  function cycleView() { vi = (vi + 1) % views.length; } // V キーで 追従/俯瞰 を切替 (ボタンは非表示)
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
    sun.position.copy(sunDir).multiplyScalar(SUN_DIST).add(sun.target.position);
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

    // 視野 (ライダー FOV): 俯角を考慮した「地面に当たる範囲」(平坦な地面を仮定した概算) を描く
    // 地面到達の水平距離 = センサ高 / tan(俯角)。視野の上端が水平以上なら最大距離まで届く。
    const footprints = [];
    for (const l of lidars) {
      const c = l.cfg, m = c.mount, y0 = rad(m.yaw_deg || 0), hf = rad(c.h_fov_deg) / 2;
      const ox = m.x || 0, oy = m.y || 0, hgt = (m.z || 0) + rcfg.wheel_radius;
      const elTop = (m.pitch_deg || 0) + c.v_fov_deg / 2, elBot = (m.pitch_deg || 0) - c.v_fov_deg / 2;
      const far = elTop < -0.01 ? Math.min(c.max_range, hgt / Math.tan(rad(-elTop))) : c.max_range;
      const near = elBot < -0.01 ? Math.min(far, Math.max(c.min_range, hgt / Math.tan(rad(-elBot)))) : c.min_range;
      footprints.push({ near, far, name: c.name });
      const P = (r, a) => [sx(oy + r * Math.sin(a)), sy(ox + r * Math.cos(a))];
      lctx.strokeStyle = '#ffd23a77'; lctx.fillStyle = '#ffd23a10'; lctx.lineWidth = 1 * dpr;
      lctx.beginPath();
      let q = P(far, y0 + hf); lctx.moveTo(q[0], q[1]);
      for (let k = 1; k <= 24; k++) { q = P(far, y0 + hf - (2 * hf * k) / 24); lctx.lineTo(q[0], q[1]); }
      for (let k = 24; k >= 0; k--) { q = P(near, y0 + hf - (2 * hf * k) / 24); lctx.lineTo(q[0], q[1]); }
      lctx.closePath(); lctx.fill(); lctx.stroke();
    }

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
      const fp = footprints.find((x) => x.name === l.cfg.name);
      info.push(`${l.cfg.name}: ${f.n} pts${fp ? `  地面 ${fp.near.toFixed(1)}〜${fp.far.toFixed(1)} m` : ''}`);
    }

    // 自車
    const L = rcfg.length * scale, Wd = rcfg.width * scale;
    lctx.fillStyle = '#e9eaec'; lctx.strokeStyle = '#4cc2ff'; lctx.lineWidth = 1.5 * dpr;
    lctx.fillRect(cx - Wd / 2, cy - L / 2, Wd, L); lctx.strokeRect(cx - Wd / 2, cy - L / 2, Wd, L);
    lctx.fillStyle = '#4cc2ff'; lctx.beginPath(); lctx.moveTo(cx, cy - L / 2 - 6 * dpr); lctx.lineTo(cx - 5 * dpr, cy - L / 2 + 2 * dpr); lctx.lineTo(cx + 5 * dpr, cy - L / 2 + 2 * dpr); lctx.fill();

  }

  // ------------------------------------------------------------------ カメラ画像 (HDR RT -> センサモデル -> 2D canvas)
  const aces = (x) => { const v = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14); return v < 0 ? 0 : v > 1 ? 1 : v; };
  const srgb = (x) => (x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055);
  const gauss = () => Math.random() + Math.random() + Math.random() - 1.5; // 分散 1/4
  const _v = new THREE.Vector3();

  function sensorPipeline(c) {
    const cfg = c.cfg, w = cfg.width, h = cfg.height, f = c.buf, out = c.img.data;

    // --- 自動露出 (空=真っ黒の画素は除外、太陽など極端な輝度は頭打ち) ---
    if (cfg.exposure === 'auto' || cfg.exposure == null) {
      let sum = 0, cnt = 0;
      const ls = c.lum || (c.lum = []); ls.length = 0;
      for (let i = 0; i < w * h; i += 7) {
        const L = 0.2126 * f[i * 4] + 0.7152 * f[i * 4 + 1] + 0.0722 * f[i * 4 + 2];
        if (L > 2e-4) { const Lc = Math.min(L, 3); sum += Math.log(Lc); ls.push(Lc); cnt++; }
      }
      let target = 1;
      if (cnt > 20) {
        ls.sort((a, b) => a - b);
        const p97 = ls[Math.floor(cnt * 0.97)];
        // 中間調を狙いつつ、日向の明部(上位3%)が飽和しすぎないよう露出を頭打ち (ハイライト保護)
        target = Math.min(cfg.auto_exposure_max, 0.16 / Math.exp(sum / cnt), 1.6 / p97);
      }
      c.exp += (target * Math.pow(2, cfg.ev_bias || 0) - c.exp) * 0.35;
    } else c.exp = Number(cfg.exposure) * Math.pow(2, cfg.ev_bias || 0);
    const e = c.exp, rn2 = cfg.read_noise ** 2, sn = cfg.shot_noise, vg = cfg.vignette;

    for (let y = 0; y < h; y++) {
      const src = (c.topDown ? y : h - 1 - y) * w * 4, dst = y * w * 4, ny = (y + 0.5) / h * 2 - 1;
      for (let x = 0; x < w; x++) {
        const nx = (x + 0.5) / w * 2 - 1, r2 = (nx * nx + ny * ny) / 2;
        const vig = 1 - vg * r2 * r2 * 1.6 - vg * 0.2 * r2;
        for (let k = 0; k < 3; k++) {
          let v = f[src + x * 4 + k] * e * vig;
          v += 2 * gauss() * Math.sqrt(rn2 + sn * Math.max(v, 0) * 0.04); // 読み出し + ショットノイズ
          out[dst + x * 4 + k] = 255 * srgb(aces(v)) + 0.5;
        }
        out[dst + x * 4 + 3] = 255;
      }
    }
    c.ctx.putImageData(c.img, 0, 0);

    // --- 太陽が直接写っていればグレア (画素の輝度で実際に見えているか判定: 地形/HLSに隠れれば出ない) ---
    if (cfg.glare > 0) {
      _v.copy(sunDir).transformDirection(c.cam.matrixWorldInverse);
      if (_v.z < -0.02) {
        const t = Math.tan(rad(c.cam.fov) / 2);
        const px = (_v.x / -_v.z / (t * c.cam.aspect) * 0.5 + 0.5) * w, py = (1 - (_v.y / -_v.z / t * 0.5 + 0.5)) * h;
        if (px > -w * 0.3 && px < w * 1.3 && py > -h * 0.3 && py < h * 1.3) {
          let vis = 0;
          for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
            const xx = Math.round(px) + dx, yy = Math.round(py) + dy;
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
            if (f[((c.topDown ? yy : h - 1 - yy) * w + xx) * 4] > 30) vis++;
          }
          const a = Math.min(1, vis / 6) * cfg.glare;
          if (a > 0.02) {
            const R = w * 0.55, g = c.ctx.createRadialGradient(px, py, 0, px, py, R);
            g.addColorStop(0, `rgba(255,246,230,${0.95 * a})`); g.addColorStop(0.05, `rgba(255,240,215,${0.6 * a})`);
            g.addColorStop(0.25, `rgba(255,225,190,${0.16 * a})`); g.addColorStop(1, 'rgba(255,220,180,0)');
            c.ctx.globalCompositeOperation = 'lighter'; c.ctx.fillStyle = g; c.ctx.fillRect(0, 0, w, h);
            c.ctx.globalCompositeOperation = 'source-over';
          }
        }
      }
    }
    drawDistanceOverlay(c);
  }

  // ---- 地面距離の目盛: カメラ位置から平坦な接地面(車輪接地高さ)を仮定した推定値 ----
  // 車体座標系で「カメラ直下を中心とする半径 d の円」を接地面上に描き、実際のカメラモデルで投影する。
  // 地形の凹凸や車体の傾きによる実距離との差は、この目盛とレンジ(ライダー)との比較で分かる。
  const _p = new THREE.Vector3();
  function drawDistanceOverlay(c) {
    const cfg = c.cfg, ctx = c.ctx, w = cfg.width, h = cfg.height;
    if (!cfg.distance_overlay || !$('#c-dist').checked) return;
    const m = cfg.mount, zg = -rcfg.wheel_radius; // 接地面の高さ (車体座標)
    const t = Math.tan(rad(c.cam.fov) / 2), asp = c.cam.aspect;
    const project = (bx, by) => {
      _p.set(bx, by, zg).applyMatrix4(rover.matrixWorld).applyMatrix4(c.cam.matrixWorldInverse);
      if (_p.z > -0.3) return null;
      return [(_p.x / -_p.z / (t * asp) * 0.5 + 0.5) * w, (1 - (_p.y / -_p.z / t * 0.5 + 0.5)) * h];
    };
    ctx.save();
    ctx.lineWidth = 1; ctx.font = '11px system-ui, sans-serif'; ctx.textBaseline = 'middle';
    const y0 = rad(m.yaw_deg || 0), half = rad(cfg.fov_deg) / 2 + 0.2;
    let lastY = 1e9; // 近い順に描き、ラベルが密集する遠方は間引く
    for (const d of [...cfg.overlay_distances].sort((a, b) => a - b)) {
      ctx.strokeStyle = 'rgba(255,230,120,0.7)';
      ctx.beginPath();
      let pen = false, best = null;
      for (let a = -half; a <= half; a += rad(1.0)) {
        const pt = project(m.x + d * Math.cos(y0 + a), m.y + d * Math.sin(y0 + a));
        if (!pt || pt[0] < -20 || pt[0] > w + 20 || pt[1] < -20 || pt[1] > h + 20) { pen = false; continue; }
        pen ? ctx.lineTo(pt[0], pt[1]) : ctx.moveTo(pt[0], pt[1]);
        pen = true;
        // ラベルは画像中央付近 (x が中心に最も近い点) に置く。ただし画面内に収まるものだけ
        const score = Math.abs(pt[0] - w * 0.5);
        if (pt[1] > 10 && pt[1] < h - 6 && (!best || score < best[2])) best = [pt[0], pt[1], score];
      }
      ctx.stroke();
      if (best && Math.abs(best[1] - lastY) >= 13) {
        lastY = best[1];
        const label = `${d} m`;
        ctx.strokeStyle = 'rgba(0,0,0,0.85)'; ctx.lineWidth = 3; ctx.strokeText(label, best[0] + 3, best[1] - 6);
        ctx.fillStyle = '#ffe678'; ctx.fillText(label, best[0] + 3, best[1] - 6); ctx.lineWidth = 1;
      }
    }
    ctx.restore();
  }

  function renderCameras(now) {
    for (const l of lidars) l.pts.visible = false; // 点群をカメラ画像に写さない
    for (const c of cams) {
      if (!c.rt || now - c.last < c.period) continue;
      c.last = now;
      renderer.setRenderTarget(c.rt);
      renderer.render(scene, c.cam);
      renderer.readRenderTargetPixels(c.rt, 0, 0, c.cfg.width, c.cfg.height, c.buf);
      sensorPipeline(c);
      c.n++;
      if (now - c.t0 > 1000) { c.fpsEl.textContent = `${(c.n * 1000 / (now - c.t0)).toFixed(0)} Hz`; c.n = 0; c.t0 = now; }
    }
    renderer.setRenderTarget(null);
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
    for (const m of fillMats) m.emissiveIntensity = FILL_RAD;
    for (const o of outlines) o.visible = true;
    renderer.render(scene, mainCam);
    for (const m of fillMats) m.emissiveIntensity = 0;
    for (const o of outlines) o.visible = false;
    renderCameras(now);
    drawLidar();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // デバッグ/自動テスト用
  window.__sim = { scene, st, cams, lidars, renderer };
}

main().catch((e) => { console.error(e); document.body.insertAdjacentHTML('beforeend', `<pre style="color:#f88;position:fixed;top:0;left:0;z-index:9">${e.stack || e}</pre>`); });
