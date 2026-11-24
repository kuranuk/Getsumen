// キーボード操作: W/S・↑/↓ = 前進/後退, A/D・←/→ = 左/右旋回, Shift = 低速, Space = 停止, R = リセット, V = 視点切替。
//
// - e.code で判定するので、キー配列や IME に依存しない。
// - キャプチャ段階で処理するので、スライダやボタンにフォーカスがあっても矢印キーや Space を奪われない。
const KEYS = { KeyW: 'fwd', ArrowUp: 'fwd', KeyS: 'back', ArrowDown: 'back', KeyA: 'left', ArrowLeft: 'left', KeyD: 'right', ArrowRight: 'right' };
const SLOW = 0.3;       // Shift 押下中の指令の倍率
const RESEND_MS = 50;   // 押している間の指令の再送間隔 (サーバは途切れると車両を止める)

/** @param send  サーバへメッセージを送る関数 */
export function setupInput({ send, onToggleView }) {
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
