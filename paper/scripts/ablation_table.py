"""从 300-epoch 消融运行汇总 Table III 的均值±标准差（可重复运行）。"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "full": lambda e: (e.get("loss", "metric") == "metric" and e.get("angle_mode", "circular") == "circular"
                       and e.get("base_skip", True) is not False and not e.get("pca_path")),
    "mse": lambda e: e.get("loss") == "mse" and not e.get("pca_path"),
    "nobase": lambda e: e.get("base_skip") is False and not e.get("pca_path"),
    "dense": lambda e: e.get("angle_mode") == "dense" and not e.get("pca_path"),
}

rows = {}
for p in glob.glob(str(WT / "experiments/axis_codec/indoor_cr4/*/metrics.json")):
    d = json.loads(Path(p).read_text())
    cfg = d.get("config") or {}
    sel = (d.get("selected") or {}).get("best_val") or {}
    if not sel or d.get("epochs_completed") != 300:
        continue
    extra = cfg.get("model.extra") or {}
    if extra.get("channels", 8) != 8 or extra.get("depth", 8) != 8:
        continue
    for name, pred in VARIANTS.items():
        if pred(extra):
            rows.setdefault(name, []).append(sel["nmse"])
            break

for name in ("full", "mse", "nobase", "dense"):
    v = np.array(rows.get(name, []))
    if len(v):
        print(f"{name:7s} n={len(v)} mean={v.mean():.3f} std={v.std(ddof=1) if len(v) > 1 else float('nan'):.3f} "
              f"values={np.round(v, 3).tolist()}")
    else:
        print(f"{name:7s} no runs")
