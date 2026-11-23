"""起動: python -m lunarsim [--config config/default.yaml]"""
from __future__ import annotations

import argparse
import time

from aiohttp import web

from .config import load_config
from .server import create_app


def main():
    ap = argparse.ArgumentParser(description="月面ロボットシミュレータ (Web UI)")
    ap.add_argument("--config", "-c", default="config/default.yaml", help="設定YAML")
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
    print(f"[lunarsim] 地形生成 {time.time() - t0:.2f}s / seed={cfg['world']['seed']}")
    print(f"[lunarsim] ブラウザで開く: http://localhost:{port}/")
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
