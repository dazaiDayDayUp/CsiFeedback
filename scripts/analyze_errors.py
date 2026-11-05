#!/usr/bin/env python
"""误差分解：按角度 bin 与时延 tap 统计误差/信号能量占比，并输出头部/尾部精度。

用于回答「模型的残差落在哪里」：CSI 的 NMSE 由时延尾部主导还是头部主导，
是判断结构先验是否有效的关键诊断。

用法：
    python scripts/analyze_errors.py --config <yaml> --checkpoint <pth> \
        --split test --n 2000 --out breakdown.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.data.cost2100 import Cost2100Dataset
from csifeedback.metrics.csi_metrics import compute_nmse, compute_rho
from csifeedback.models import get_model_from_config
from csifeedback.utils.checkpoint import load_checkpoint, load_model_state_dict
from csifeedback.utils.config import load_config


def main():
    ap = argparse.ArgumentParser(description="CSI 反馈模型的误差分解")
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--checkpoint", "-ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=2000, help="最多评估的样本数")
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--head-taps", type=int, default=4, help="头部包含的时延 tap 数")
    ap.add_argument("--data-dir", default=None, help="覆盖配置中的 data.data_dir")
    ap.add_argument("--out", "-o", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config.data.data_dir = args.data_dir
    model = get_model_from_config(config.model)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    load_model_state_dict(model, ckpt.model_state_dict)
    model.eval()

    need_raw = args.split == "test"
    ds = Cost2100Dataset(config.data.data_dir, config.data.scenario, args.split, load_raw=need_raw)
    if len(ds) > args.n:
        idx = np.random.RandomState(0).choice(len(ds), args.n, replace=False)
        xs = ds.tensors[0][idx]
        raws = ds.tensors[1][idx] if need_raw else None
    else:
        xs = ds.tensors[0]
        raws = ds.tensors[1] if need_raw else None

    preds = []
    with torch.no_grad():
        for i in range(0, len(xs), args.batch_size):
            preds.append(model(xs[i : i + args.batch_size]))
    preds = torch.cat(preds, dim=0)

    gt = (xs - 0.5).numpy()
    pr = (preds - 0.5).numpy()
    err_map = ((pr - gt) ** 2).sum(axis=1).sum(axis=0)   # (angle, delay)
    sig_map = (gt**2).sum(axis=1).sum(axis=0)
    ratio = ((pr - gt) ** 2).sum(axis=(1, 2, 3)) / np.clip((gt**2).sum(axis=(1, 2, 3)), 1e-12, None)

    t0 = args.head_taps
    res = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n": int(len(xs)),
        "nmse_db": float(compute_nmse(preds, xs).item()),
        "median_ratio_db": float(10 * np.log10(np.median(ratio))),
        "p90_ratio_db": float(10 * np.log10(np.percentile(ratio, 90))),
        "worst5pct_share": float(np.sort(ratio)[-len(ratio) // 20 :].sum() / ratio.sum()),
        "delay_err_frac": [float(v) for v in err_map.sum(axis=0) / err_map.sum()],
        "delay_sig_frac": [float(v) for v in sig_map.sum(axis=0) / sig_map.sum()],
        "angle_err_frac": [float(v) for v in err_map.sum(axis=1) / err_map.sum()],
        "angle_sig_frac": [float(v) for v in sig_map.sum(axis=1) / sig_map.sum()],
        "head_err_frac": float(err_map[:, :t0].sum() / err_map.sum()),
        "head_sig_frac": float(sig_map[:, :t0].sum() / sig_map.sum()),
        "head_rel_err": float(err_map[:, :t0].sum() / max(sig_map[:, :t0].sum(), 1e-12)),
        "tail_rel_err": float(err_map[:, t0:].sum() / max(sig_map[:, t0:].sum(), 1e-12)),
    }
    if need_raw:
        res["rho"] = float(compute_rho(preds, raws).item())

    print(f"NMSE {res['nmse_db']:.3f} dB" + (f"  rho {res['rho']:.6f}" if "rho" in res else ""))
    print(f"  median {res['median_ratio_db']:.2f} dB, p90 {res['p90_ratio_db']:.2f} dB, "
          f"worst5% share {res['worst5pct_share'] * 100:.1f}%")
    print(f"  head (tap<{t0}): sig {res['head_sig_frac'] * 100:.1f}%  "
          f"err {res['head_err_frac'] * 100:.1f}%  rel {res['head_rel_err']:.2e}")
    print(f"  tail (tap>={t0}): sig {100 - res['head_sig_frac'] * 100:.1f}%  "
          f"err {100 - res['head_err_frac'] * 100:.1f}%  rel {res['tail_rel_err']:.2e}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print("saved", args.out)


if __name__ == "__main__":
    main()
