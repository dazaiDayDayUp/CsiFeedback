"""Fig.1 几何：角时延能量图（前 8 tap）+ 角度圆周剖面 + 时延功率剖面。"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
import style
from figqa import qa

style.setup()
D = Path(__file__).resolve().parent.parent / "data"
g = np.load(D / "geometry.npz")
meta = json.loads((D / "meta.json").read_text())

mean_map = g["mean_map"]          # (angle, delay)
delay = g["delay_profile"]
angle = g["angle_profile"]
cum = g["delay_cum"]

fig = plt.figure(figsize=(style.DBL, 2.0))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.05, 0.95, 1.16], wspace=0.60,
                       left=0.072, right=0.972, top=0.86, bottom=0.215)

# ---------------- (a) energy map, first 8 taps ----------------
ax = fig.add_subplot(gs[0])
sub = mean_map[:, :8]
cmap_trunc = ListedColormap(plt.get_cmap("magma")(np.linspace(0.20, 1.0, 256)))
im = ax.imshow(sub, aspect="auto", origin="lower", cmap=cmap_trunc,
               norm=plt.matplotlib.colors.LogNorm(vmin=sub.max() / 200, vmax=sub.max()),
               extent=[-0.5, 7.5, -0.5, 31.5], interpolation="nearest")
ax.set_xlabel("delay tap $n_\\tau$", labelpad=2.0)
ax.set_ylabel("angle bin $n_a$", labelpad=3.0)
ax.set_xticks([0, 2, 4, 6])
ax.set_yticks([0, 8, 16, 24, 31])
ax.set_xlim(-0.5, 7.5)
ax.axvline(3.5, color="w", lw=0.9, ls=(0, (2.5, 1.8)), alpha=0.9)
ax.set_ylim(-0.5, 31.5)
ax.text(2.0, 27.5, f"{meta['first4_energy']*100:.0f}%", color="w", fontsize=8.0,
        ha="center", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.35))
ax.text(7.25, 1.6, "taps 8–31:\n0.9%", color="w", fontsize=7.0, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.16", fc="black", ec="none", alpha=0.3))
cb = fig.colorbar(im, ax=ax, pad=0.045, fraction=0.048)
cb.set_label("energy (a.u.)", fontsize=7.2, labelpad=9.0)
cb.ax.tick_params(labelsize=6.8)
cb.locator = plt.matplotlib.ticker.LogLocator(numticks=2)
cb.update_ticks()
style.panel_label(ax, "(a)", x=-0.30, y=1.03)

# ---------------- (b) circular angular heat-ring ----------------
axp = fig.add_subplot(gs[1])
axp.set_aspect("equal")
axp.set_xlim(-1.34, 1.34)
axp.set_ylim(-1.40, 1.34)
axp.axis("off")
from matplotlib.patches import Wedge
logp = np.log10(angle)
cn = plt.matplotlib.colors.Normalize(vmin=logp.min(), vmax=logp.max())
for b in range(32):
    # bin 0 在正上方，编号顺时针；楔形之间留 6% 缝隙
    center = 90 - 360 * b / 32
    half = 360 / 32 / 2 * 0.94
    axp.add_patch(Wedge((0, 0), 1.0, center - half, center + half, width=0.52,
                        fc=plt.get_cmap("magma")(cn(logp[b])), ec="white", lw=0.35))
# 接缝（bin 31 与 bin 0 之间）
sa = np.deg2rad(90 + 360 / 64)
axp.plot([0.36 * np.cos(sa), 1.12 * np.cos(sa)], [0.36 * np.sin(sa), 1.12 * np.sin(sa)],
         color=style.C["red"], lw=0.9, ls=(0, (2.4, 1.8)), zorder=6)
for b, txt in ((8, "8"), (16, "16"), (24, "24")):
    tb = np.deg2rad(90 - 360 * b / 32)
    axp.text(1.16 * np.cos(tb), 1.16 * np.sin(tb), txt, ha="center", va="center",
             fontsize=7.2, color="#33363A")
axp.text(0.0, 1.13, "seam 31$\,$|$\,$0", ha="center", va="bottom",
         fontsize=6.9, color=style.C["red"])
axp.text(0, -1.30, "angle bin (circular)", ha="center", va="top", fontsize=7.4)
style.panel_label(axp, "(b)", x=0.02, y=1.0)

# ---------------- (c) delay profile ----------------
axd = fig.add_subplot(gs[2])
n = np.arange(32)
axd.semilogy(n, delay, "o-", color=style.C["orange"], ms=2.3, lw=1.2, mfc="w", mew=0.9)
axd.set_xlabel("delay tap $n_\\tau$", labelpad=2.0)
axd.set_ylabel("normalized energy", labelpad=4.5)
axd.set_xlim(-1, 32)
axd.set_ylim(4e-5, 1.4)
axd.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])

ax2 = axd.twinx()
ax2.plot(n, cum * 100, color="#1E6B3A", lw=1.3, ls=(0, (4, 2)))
ax2.set_ylabel("cumulative (%)", color="#1E6B3A", fontsize=7.2, labelpad=2.0)
ax2.tick_params(axis="y", colors="#1E6B3A", labelsize=6.8)
ax2.set_ylim(0, 105)

ax2.set_yticks([0, 50, 100])
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color("#1E6B3A")
ax2.spines["right"].set_linewidth(0.8)
axd.axvline(3.5, color=style.C["grey"], lw=0.7, ls=(0, (3, 2)))
style.panel_label(axd, "(c)", x=-0.30, y=0.99)

for ext in ("png", "pdf"):
    fig.savefig(Path(__file__).resolve().parent.parent / "figures" / f"fig1_geometry.{ext}")
qa(fig, "Fig.1 geometry")
