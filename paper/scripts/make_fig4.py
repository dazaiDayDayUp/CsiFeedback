"""Fig.4 率失真：主图（4 条曲线）+ 下方增益条（相对 STNet）+ 顶部码字维度轴。"""

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
pos = np.arange(len(crs))

pca = [oracle["pca"][str(budget[cr])] for cr in crs]
topk = [oracle["topk"][str(budget[cr] // 2)] for cr in crs]
st = np.array([base[cr] for cr in crs])

ours, ours_err = [], []
cm = meta["cr_matrix"]
for cr in crs:
    v = np.array([r["nmse"] for r in cm[str(cr)]], dtype=float)
    ours.append(v.mean())
    ours_err.append(v.std(ddof=1) if len(v) > 1 else 0.0)
ours = np.array(ours)
ours_err = np.array(ours_err)
gain = st - ours

fig = plt.figure(figsize=(style.COL, 2.30))
gs = fig.add_gridspec(2, 1, height_ratios=[3.45, 1.10], hspace=0.13,
                      left=0.155, right=0.975, top=0.875, bottom=0.175)
ax = fig.add_subplot(gs[0])
axb = fig.add_subplot(gs[1], sharex=ax)

# ---------------- 主图 ----------------
ax.plot(pos, pca, "--s", color="#9AA0A6", ms=3.4, lw=1.0, mfc="white", mew=0.9,
        label="PCA (linear)")
ax.plot(pos, topk, ":v", color="#3A3D40", ms=3.8, lw=1.1, mfc="white", mew=0.9,
        label="sparse oracle")
ax.plot(pos, st, "-o", color=style.C["orange"], ms=4.0, lw=1.25, label="STNet")
ax.errorbar(pos, ours, yerr=ours_err, fmt="-D", color=style.C["blue"], ms=4.2,
            lw=1.5, capsize=2.4, elinewidth=1.0, capthick=0.9, label="AxisCodec")

ax.set_ylabel("NMSE (dB)", fontsize=7.8)
ax.set_ylim(-35.0, -1.5)
ax.set_yticks([-30, -25, -20, -15, -10, -5])
ax.set_xlim(-0.45, 4.45)
ax.grid(axis="y", color="#ECECEC", lw=0.55)
ax.set_axisbelow(True)
ax.tick_params(axis="x", bottom=False, labelbottom=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=6.5, handlelength=1.6,
          columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)

# 顶部码字维度
axt = ax.secondary_xaxis("top")
axt.set_xticks(pos)
axt.set_xticklabels([str(budget[cr]) for cr in crs])
axt.tick_params(length=2.2, labelsize=6.5, pad=1.5)
axt.set_xlabel("latent dimension $D$", fontsize=7.2, labelpad=1.5)
axt.spines["top"].set_linewidth(0.7)

# ---------------- 增益条 ----------------
axb.bar(pos, gain, width=0.52, color=style.C["blue"], alpha=0.9, zorder=3)
axb.axhline(0, color="#1C1C1C", lw=0.8, zorder=4)
for x, g in zip(pos, gain):
    axb.text(x, g + 0.12, f"{g:+.2f}", ha="center", va="bottom", fontsize=6.8,
             color=style.C["blue"])
axb.set_ylim(0, 3.1)
axb.set_yticks([0, 1, 2, 3])
axb.set_ylabel("gain (dB)", fontsize=7.8, labelpad=2)
axb.set_xticks(pos)
axb.set_xticklabels([f"1/{c}" for c in crs])
axb.set_xlabel("compression ratio", labelpad=2)
axb.tick_params(length=2.4, pad=2, labelsize=7.0)
axb.grid(axis="y", color="#F0F0F0", lw=0.5)
axb.set_axisbelow(True)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)

for ext in ("png", "pdf"):
    fig.savefig(Path(__file__).resolve().parent.parent / "figures" / f"fig4_rd.{ext}")
qa(fig, "Fig.4 rate-distortion")
