#!/usr/bin/env python
"""从训练集计算角时延域的 PCA 合成基（用于 AxisCodec 的 PCA 初始化）。

输出 ``{"basis": (D, 2048) float32, "mean": (2048,)}``；basis 的行是前 D 个主成分，
可直接作为 ``P``：编码 ``z = P @ (x - 0.5)``，解码 ``x = 0.5 + P.T @ z``。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--scenario", default="indoor")
    ap.add_argument("--dim", type=int, required=True, help="输出的主成分个数（=encoded_dim）")
    ap.add_argument("--max-samples", type=int, default=100000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    suffix = {"indoor": "in", "outdoor": "out"}[args.scenario]
    path = os.path.join(args.data_dir, f"DATA_Htrain{suffix}.mat")
    x = np.asarray(sio.loadmat(path)["HT"], dtype=np.float32).reshape(-1, 2, 32, 32)
    if len(x) > args.max_samples:
        x = x[: args.max_samples]
    flat = x.reshape(len(x), -1).astype(np.float64) - 0.5

    dim = flat.shape[1]
    cov = np.zeros((dim, dim), dtype=np.float64)
    for i in range(0, len(flat), 5000):
        a = flat[i : i + 5000]
        cov += a.T @ a
    cov /= len(flat)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    basis = evecs[:, order[: args.dim]].T.astype(np.float32)  # (D, 2048)
    cum = np.cumsum(evals[order]) / evals.sum()
    print(f"{args.scenario} D={args.dim}: 累计能量 {cum[args.dim-1]:.6f}")
    torch.save(
        {"basis": torch.from_numpy(basis), "mean": torch.zeros(dim)},
        args.out,
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
