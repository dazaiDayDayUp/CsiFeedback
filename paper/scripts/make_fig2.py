"""Fig.2 误差分布：逐时延 tap 的误差份额 vs 信号份额（STNet 与 AxisCodec，CR1/4）。"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import style
from figqa import qa

style.setup()
D = Path(__file__).resolve().parent.parent / "data"
meta = json.loads((D / "meta.json").read_text())
s = meta["decomp_stnet_cr4"]
a = meta["decomp_axis_cr4"]

fig, ax = plt.subplots(figsize=(style.COL, 2.05))
fig.subplots_adjust(left=0.155, right=0.985, top=0.90, bottom=0.215)
n = np.arange(16)
sig = np.array(s["delay_sig"][:16]) * 100
es = np.array(s["delay_err"][:16]) * 100
ea = np.array(a["delay_err"][:16]) * 100

ax.fill_between(n, 0, sig, step="mid", color="#D9D2C5", alpha=0.9, lw=0, zorder=1)
ax.plot(n, sig, drawstyle="steps-mid", color="#8A7F6A", lw=0.9, zorder=2)
ax.bar(n - 0.21, es, width=0.40, color="#B9B2A6", edgecolor="#6E675C", lw=0.4,
       hatch="////", label="STNet", zorder=3)
ax.bar(n + 0.21, ea, width=0.40, color=style.C["blue"], label="AxisCodec", zorder=3)

ax.axvline(3.5, color=style.C["red"], lw=0.8, ls=(0, (3, 2)), zorder=4)
ax.text(2.35, 42, "head", ha="center", fontsize=7.6, color=style.C["red"])
ax.text(9.6, 45, "tail", ha="center", fontsize=7.6, color=style.C["red"])

ax.set_xlim(-0.6, 15.6)
ax.set_ylim(0, 66)
ax.set_xlabel("delay tap $n_\\tau$  (taps 16–31 omitted)")
ax.set_ylabel("share of total (%)")
ax.set_xticks([0, 3, 7, 11, 15])
from matplotlib.patches import Patch
handles = [Patch(facecolor="#D9D2C5", edgecolor="#8A7F6A", lw=0.6, label="signal share"),
           Patch(facecolor="#B9B2A6", edgecolor="#6E675C", hatch="////", lw=0.4, label="STNet"),
           Patch(facecolor=style.C["blue"], label="AxisCodec")]
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=7.2)


for ext in ("png", "pdf"):
    fig.savefig(Path(__file__).resolve().parent.parent / "figures" / f"fig2_error.{ext}")
qa(fig, "Fig.2 error")
