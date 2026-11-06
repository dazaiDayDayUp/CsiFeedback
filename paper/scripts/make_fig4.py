"""Fig.4 率失真：STNet / AxisCodec / PCA / 自由位置稀疏 oracle vs 压缩比。"""

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

oracle = meta["oracle"]
base = {int(k[2:]): v[0] for k, v in meta["baseline"].items()}
crs = [4, 8, 16, 32, 64]
budget = {cr: 2048 // cr for cr in crs}

pca = [oracle["pca"][str(budget[cr])] for cr in crs]
# top-k oracle：k = budget/2（实数预算换成复系数个数）
topk = [oracle["topk"][str(budget[cr] // 2)] for cr in crs]
st = [base[cr] for cr in crs]

ours, ours_err, ours_x = [], [], []
cm = meta["cr_matrix"]
for cr in crs:
    rows = cm.get(str(cr), [])
    if not rows:
        continue
    v = np.array([r["nmse"] for r in rows], dtype=float)
    ours.append(v.mean())
    ours_err.append(v.std(ddof=1) if len(v) > 1 else 0.0)
    ours_x.append(cr)

fig, ax = plt.subplots(figsize=(style.COL, 2.35))
fig.subplots_adjust(left=0.165, right=0.975, top=0.955, bottom=0.235)

x = np.array(crs)
ax.plot(x, pca, "s--", color="#9A9A9A", ms=3.6, lw=1.0, mfc="w", mew=0.9, label="PCA (linear)")
ax.plot(x, topk, "v-", color="#1C1C1C", ms=3.8, lw=1.0, mfc="w", mew=0.9,
        label="sparse oracle")
ax.plot(x, st, "o-", color=style.C["orange"], ms=4.0, lw=1.1, label="STNet")
ax.errorbar(ours_x, ours, yerr=ours_err, fmt="D-", color=style.C["blue"], ms=4.2, lw=1.4,
            capsize=2.6, elinewidth=1.2, capthick=1.0, label="AxisCodec")

for cr, o, sb in zip(ours_x, ours, [base[c] for c in ours_x]):
    ax.text(cr, -0.145, f"{sb - o:+.2f}", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.0, color=style.C["blue"])

ax.set_xscale("log", base=2)
ax.set_xticks(crs)
ax.set_xticklabels([f"1/{c}" for c in crs])
ax.set_xlabel("compression ratio", labelpad=18)
ax.set_ylabel("NMSE (dB)")
ax.set_xlim(3.4, 78)
ax.set_ylim(-34.5, -2.2)
ax.grid(axis="y", color="#E8E8E8", lw=0.5)
ax.set_axisbelow(True)
leg = ax.legend(loc="lower right", ncol=1, fontsize=7.2, handlelength=1.7,
                borderaxespad=0.4, frameon=True, framealpha=0.92)
leg.get_frame().set_edgecolor(style.C["lightgrey"])
leg.get_frame().set_linewidth(0.5)


for ext in ("png", "pdf"):
    fig.savefig(Path(__file__).resolve().parent.parent / "figures" / f"fig4_rd.{ext}")
qa(fig, "Fig.4 rate-distortion")
