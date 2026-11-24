// 月面ロボットシミュレータ: 全体の組み立て (サーバとの通信, 画面部品, 視点, メインループ)。
//   scene.js   右: 月面の3Dビュー      cameras.js 左下: カメラ画像 (レイトレ結果 + センサ処理)
//   lidar.js   左上: 点群の俯瞰図       input.js   キーボード操作
import { MoonView, deg, rad } from './scene.js';
import { Cameras } from './cameras.js';
import { Lidars } from './lidar.js';
import { setupInput } from './input.js';
import * as THREE from './vendor/three.module.min.js';

const $ = (s) => document.querySelector(s);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
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
