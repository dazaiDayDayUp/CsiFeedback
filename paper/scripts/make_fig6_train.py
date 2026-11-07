"""Fig.7 训练动态：验证集 NMSE vs epoch（AxisCodec 三个种子 vs STNet 官方配方）。"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import style
from figqa import qa

style.setup()
WT = Path(__file__).resolve().parents[2]
MAIN = Path("/home/wzh/myProjects/CsiFeedback")

runs = [
    WT / "experiments/axis_codec/indoor_cr4/20261105_182644/metrics.json",
    WT / "experiments/axis_codec/indoor_cr4/20261105_235526/metrics.json",
    WT / "experiments/axis_codec/indoor_cr4/20261106_025326/metrics.json",
]
stnet_log = MAIN / "experiments/stnet/indoor_cr4/20260619_064715/logs/train.log"


def read_axis(path):
    import json
    h = json.loads(path.read_text())["history"]
    ep = [e["epoch"] for e in h if e["kind"] == "val"]
    nm = [e["nmse"] for e in h if e["kind"] == "val"]
    return np.array(ep), np.array(nm)


def read_stnet(path):
    pat = re.compile(r"Epoch \[(\d+)/\d+\] val_loss: .* val_NMSE: ([-\d.]+) dB")
    ep, nm = [], []
    for line in path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            ep.append(int(m.group(1)))
            nm.append(float(m.group(2)))
    return np.array(ep), np.array(nm)

ep_a, nm_a = zip(*[read_axis(p) for p in runs])
ep_a = ep_a[0]
nm_a = np.vstack(nm_a)
ep_s, nm_s = read_stnet(stnet_log)

fig, ax = plt.subplots(figsize=(style.COL, 1.98))
fig.subplots_adjust(left=0.155, right=0.985, top=0.965, bottom=0.215)

ax.fill_between(ep_a, nm_a.min(0), nm_a.max(0), color=style.C["blue"], alpha=0.20, lw=0)
ax.plot(ep_a, nm_a.mean(0), color=style.C["blue"], lw=1.15, label="AxisCodec (3 seeds)")
ax.plot(ep_s, nm_s, color=style.C["orange"], lw=1.0, label="STNet (published recipe)")

ax.axvline(300, color=style.C["grey"], lw=0.7, ls=(0, (2, 2)))
ax.text(286, -1.2, "300-epoch ablation", fontsize=6.6, color="#4D5656",
        va="top", ha="right")

ax.set_xlim(0, 1000)
ax.set_ylim(-34, 0)
ax.set_xlabel("epoch")
ax.set_ylabel("validation NMSE (dB)")
ax.set_xticks([0, 200, 400, 600, 800, 1000])
ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.995), fontsize=6.9)

for ext in ("png", "pdf"):
    fig.savefig(Path(__file__).resolve().parent.parent / "figures" / f"fig6_train.{ext}")
qa(fig, "Fig.6 training")
print("axis seeds final", nm_a[:, -1], "stnet final", nm_s[-1])
