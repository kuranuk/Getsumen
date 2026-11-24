"""設定の読み込み。既定値はすべて config/default.yaml にあり、指定された設定ファイルはその差分として上書きされる."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def deep_merge(base: dict, over: dict) -> dict:
    """base に over を再帰的に上書きした新しい dict (リストは置き換え)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path=None) -> dict:
    cfg = _read(DEFAULT_CONFIG)
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        cfg = deep_merge(cfg, _read(path))

    # カメラ/ライダーの各項目に、共通の既定値 (*_defaults) を適用する
    sensors = cfg["sensors"]
    names = set()
    for kind, defaults in (("cameras", "camera_defaults"), ("lidars", "lidar_defaults")):
        sensors[kind] = [deep_merge(sensors[defaults], c) for c in sensors[kind]]
        for c in sensors[kind]:
            c.setdefault("label", c["name"])
            if c["name"] in names:
                raise ValueError(f"センサ名が重複しています: {c['name']}")
            names.add(c["name"])
    return cfg
