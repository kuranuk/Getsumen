# Getsumen — 月面ロボットシミュレータ

有人与圧ローバーが HLS（有人着陸機）に接近するオペレーションを検討するための、Python ベースの月面シミュレーション環境です。
起動するとブラウザから見られます。

```
左: LiDAR点群(車両上方から) + 前/後/左/右カメラ画像     右: 月面の3D(CG)ビュー
```

## 起動

```bash
pip install -r requirements.txt
python -m lunarsim                        # 既定: config/default.yaml
python -m lunarsim -c my.yaml --port 9000 --seed 3
```

ブラウザで http://localhost:8080/ を開きます。（three.js は `lunarsim/web/vendor/` に同梱しているのでオフラインでも動きます）

## 操作

| 操作 | 内容 |
|---|---|
| W / S（↑ / ↓） | 前進 / 後退 |
| A / D（← / →） | 左 / 右旋回 |
| Shift / Space | 低速 / 停止 |
| 自動アプローチ | HLS へ向けて自動走行し、設定した距離で停止 |
| V / ドラッグ / ホイール | 視点切替（追従・俯瞰） / 回転 / ズーム |
| R | 初期位置へリセット |
| 左上の LiDAR 図でホイール | 表示半径の変更 |

HUD には速度・HLS までの距離（車体前端〜HLS 表面）・HLS 方位・車体傾斜を表示します。

## 設定ファイル（`config/default.yaml`）

カメラとフラッシュライダーはすべて YAML で設定できます（変更後は再起動）。

```yaml
sensors:
  cameras:                  # 数・名称は自由。画面に自動で並びます
    - name: front
      mount: {x: 2.9, y: 0.0, z: 1.3, yaw_deg: 0, pitch_deg: -3}   # 車体座標 x前 y左 z上 / pitch は上向き正
      fov_deg: 90           # 水平画角
      width: 320
      height: 240
      rate_hz: 10
  lidars:
    - name: front_flash
      mount: {x: 3.0, y: 0.0, z: 0.7, yaw_deg: 0, pitch_deg: -6}
      h_fov_deg: 40         # 視野
      v_fov_deg: 30
      h_pixels: 128         # 画素数 (フラッシュライダーのアレイ)
      v_pixels: 96
      min_range: 0.5
      max_range: 60.0
      rate_hz: 5
      range_noise_std: 0.02 # 距離ノイズ σ [m]
      dropout: 0.01         # 欠測率
```

その他、`world`（地形シード・クレータ数/大きさ/深さ・岩の数/大きさ・太陽方位/仰角）、`hls`（位置・半径・高さ・ハッチ方向）、
`rover`（寸法・初期位置・最高速度・加速度・自動アプローチ停止距離）も調整できます。

## 構成

```
lunarsim/
  world.py   高さマップ(起伏+クレータ) / 岩 / HLS と レイキャスト
  lidar.py   フラッシュライダー (格子状レイ → 距離 → 点群)
  rover.py   ユニサイクル運動 + 4点接地による姿勢追従、HLS 衝突停止、自動アプローチ
  server.py  aiohttp + WebSocket (状態 JSON / 点群バイナリを配信)
  web/       three.js フロントエンド (CG・カメラ画像・点群ビュー)
config/default.yaml
tests/       python -m unittest discover -s tests
```

- **ライダー点群** は Python 側で計算します（高さマップのレイマーチング + 岩=球 + HLS=円筒の解析交差）。
  3D ビューの CG とは同一の地形データを使うので、点群と画像は整合します。
- **カメラ画像** はブラウザ側で同じシーンを車体取付位置から描画します（点群表示はカメラ画像には写りません）。
  ブラウザが GPU を使えない環境ではカメラ更新が遅くなります（`rate_hz` や解像度を下げてください）。

## 簡略化している点（今後の拡張候補）

- HLS は円筒＋ノーズの簡易形状（脚・エレベータ等は未モデル化）。ハッチは描画のみ。
- ローバーはキネマティクスのみ（スリップ・サスペンション動力学・轍は無し）。岩との衝突判定も無し（乗り越える扱い）。
- 月の反射特性（強いオポジション効果等）、地球照、ダスト巻き上げは未実装。
