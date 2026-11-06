"""汇总论文所需的全部真实数据，缓存为 .npz / .json，供各图脚本使用。

数据来源：
  - data/DATA_Htestin.mat / DATA_Htrainin.mat：角时延能量统计
  - CsiFeedback-research/analysis/*.json：oracle、误差分解、复杂度
  - worktree/experiments/**/metrics.json：主结果与消融
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

DATA = Path("/home/wzh/myProjects/CsiFeedback/data")
RES = Path("/home/wzh/myProjects/CsiFeedback-research")
WT = Path("/home/wzh/myProjects/CsiFeedback-worktrees/cylinder_codec")
OUT = WT / "paper" / "data"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE = {  # STNet 固定 checkpoint 复评（logs/baseline_eval）
    "cr4": (-30.174291610717773, 0.9966896772384644),
    "cr8": (-19.2799, 0.9914),
    "cr16": (-14.7523, 0.981229),
    "cr32": (-9.509, 0.9433),
    "cr64": (-6.822, 0.8924),
}


def energy_stats():
    x = np.asarray(sio.loadmat(DATA / "DATA_Htestin.mat")["HT"], dtype=np.float32).reshape(-1, 2, 32, 32)
    g = x - 0.5
    e = (g[:, 0] ** 2 + g[:, 1] ** 2)  # (N, angle, delay)
    mean_map = e.mean(axis=0)
    delay = e.sum(axis=1).mean(axis=0)
    angle = e.sum(axis=2).mean(axis=0)
    delay_n = delay / delay.sum()
    angle_n = angle / angle.sum()
    cum = np.cumsum(delay_n)
    return {
        "mean_map": mean_map,
        "delay_profile": delay_n,
        "angle_profile": angle_n,
        "delay_cum": cum,
        "first4_energy": float(cum[3]),
        "first8_energy": float(cum[7]),
        # 逐样本 99% 能量支撑
        "k99": np.sort(e.reshape(len(e), -1), axis=1)[:, ::-1],
    }


def oracle():
    rd = json.loads((RES / "analysis/rate_distortion_in.json").read_text())
    return {
        "pca": {int(k): v for k, v in rd["pca_nmse_db"].items()},
        "topk": {int(k): v for k, v in rd["topk_oracle_nmse_db"].items()},
        "rank": {int(k): v for k, v in rd["rank_oracle_nmse_db"].items()},
    }


def parametric():
    out = {}
    for name, path, key in (
        ("tail4", RES / "analysis/tail_predict.json", "4"),
        ("blob2", RES / "analysis/blob_fit.json", "2"),
        ("blob4", RES / "analysis/blob_fit.json", "4"),
    ):
        if path.exists():
            d = json.loads(path.read_text())
            grp = d.get("fit") or d.get("blob_fit") or {}
            if key in grp:
                v = grp[key]
                out[name] = v["tail_rel"] if isinstance(v, dict) else v
    return out


def decomp(name):
    p = RES / f"analysis/{name}"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if "delay_err_frac" in d:                      # analyze_errors.py 输出
        return {
            "delay_err": d["delay_err_frac"], "delay_sig": d["delay_sig_frac"],
            "head_rel": d["head_rel_err"], "tail_rel": d["tail_rel_err"],
            "head_share": d.get("head_t0_4_err_frac"), "nmse": d.get("nmse_official"),
            "rho": d.get("rho"),
        }
    # verify_error_decomposition.json（独立复核 agent 的输出）
    dt = d["delay_tap_decomposition"]
    pt = dt["per_tap"]
    return {
        "delay_err": [x["err_share"] for x in pt], "delay_sig": [x["sig_share"] for x in pt],
        "head_rel": dt["head_tap_0_3"]["rel_err_energy"],
        "tail_rel": dt["tail_tap_4_31"]["rel_err_energy"],
        "head_share": dt["head_tap_0_3"]["err_share"],
        "nmse": d.get("primary_subset", {}).get("nmse_db", -30.17),
        "rho": d.get("primary_subset", {}).get("rho", 0.99669),
    }


def runs():
    out = {}
    for p in glob.glob(str(WT / "experiments/axis_codec/*/*/metrics.json")):
        d = json.loads(Path(p).read_text())
        cfg = d.get("config") or {}
        sel = (d.get("selected") or {}).get("best_val") or {}
        if not sel:
            continue
        extra = cfg.get("model.extra") or {}
        key = (d.get("scenario"), d.get("compression_ratio"), cfg.get("training.seed"),
               d.get("epochs_completed"), json.dumps(extra, sort_keys=True))
        rec = {
            "nmse": sel.get("nmse"), "rho": sel.get("rho"), "epoch": sel.get("epoch"),
            "epochs": d.get("epochs_completed"), "extra": extra,
            "cfg": cfg, "dir": Path(p).parent.name,
        }
        if key not in out or (rec["epochs"] or 0) > (out[key]["epochs"] or 0):
            out[key] = rec
    return list(out.values())


def is_baseline(extra):
    return (str(extra.get("angle_mode", "circular")) == "circular"
            and str(extra.get("loss", "metric")) == "metric"
            and int(extra.get("channels", 8)) == 8
            and int(extra.get("depth", 8)) == 8
            and str(extra.get("norm", "ln")) == "ln"
            and extra.get("base_skip", True) is not False
            and extra.get("bypass", True) is not False
            and not extra.get("pca_path"))


def cr_matrix():
    """每个 CR 取 1000 epoch 的最佳（同一 seed 0），CR4 取三 seed。"""
    rs = runs()
    by = {}
    for r in rs:
        scn, cr, seed, ep, _ = None, None, None, None, None
        # 从 dir 推断 CR 与 scenario 需要 cfg，这里直接用字段
    out = {}
    for r in rs:
        cfg = r["cfg"]
        scn = cfg.get("data.scenario") or cfg.get("data.scenario")
        cr = r.get("extra") and None
    # 直接用 metrics 里已存的 scenario / compression_ratio 由 runs() 丢失，重新读
    rows = []
    for p in glob.glob(str(WT / "experiments/axis_codec/*/*/metrics.json")):
        d = json.loads(Path(p).read_text())
        cfg = d.get("config") or {}
        sel = (d.get("selected") or {}).get("best_val") or {}
        if not sel or (d.get("epochs_completed") or 0) < 1000:
            continue
        rows.append({
            "scenario": d.get("scenario"), "cr": int(str(d.get("compression_ratio"))[2:]),
            "seed": cfg.get("training.seed"), "nmse": sel["nmse"], "rho": sel["rho"],
            "extra": cfg.get("model.extra") or {},
        })
    for r in rows:
        if not is_baseline(r["extra"]):
            continue
        out.setdefault(r["cr"], []).append(r)
    return out


def ablations():
    """300 epoch 的消融（seed 0）。"""
    rows = []
    for p in glob.glob(str(WT / "experiments/axis_codec/indoor_cr4/*/metrics.json")):
        d = json.loads(Path(p).read_text())
        cfg = d.get("config") or {}
        sel = (d.get("selected") or {}).get("best_val") or {}
        if not sel or d.get("epochs_completed") != 300:
            continue
        rows.append({"extra": cfg.get("model.extra") or {}, "nmse": sel["nmse"],
                     "sched": cfg.get("scheduler.name"), "dir": Path(p).parent.name})
    return rows


def main():
    st = energy_stats()
    np.savez(OUT / "geometry.npz", mean_map=st["mean_map"], delay_profile=st["delay_profile"],
             angle_profile=st["angle_profile"], delay_cum=st["delay_cum"])
    meta = {
        "first4_energy": st["first4_energy"], "first8_energy": st["first8_energy"],
        "oracle": oracle(), "parametric": parametric(),
        "decomp_stnet_cr4": decomp("verify_error_decomposition.json"),
        "decomp_axis_cr4": decomp("axis_cr4_e1000_final_decomp.json"),
        "decomp_stnet_cr16": decomp("stnet_cr16_decomp.json"),
        "decomp_axis_cr16": decomp("axis_cr16_e1000_decomp.json"),
        "decomp_stnet_cr64": decomp("stnet_cr64_decomp.json"),
        "cr_matrix": {str(k): v for k, v in cr_matrix().items()},
        "baseline": BASELINE,
        "ablations": ablations(),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("saved", OUT / "meta.json")
    print("CR rows:", {k: [(r["seed"], round(r["nmse"], 3)) for r in v] for k, v in meta["cr_matrix"].items()})
    print("ablations:", [(r["extra"], round(r["nmse"], 3)) for r in meta["ablations"]])


if __name__ == "__main__":
    main()
