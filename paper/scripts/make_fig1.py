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
gs = gridspec.GridSpec(1, 3, width_ratios=[1.05, 0.95, 1.16], wspace=0.52,
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
ax.text(7.25, 1.6, "taps 8–31:\n0.9%", color="w", fontsize=6.8, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.16", fc="black", ec="none", alpha=0.3))
cb = fig.colorbar(im, ax=ax, pad=0.045, fraction=0.048)
cb.set_label("energy (a.u.)", fontsize=7.2, labelpad=9.0)
cb.ax.tick_params(labelsize=6.8)
cb.locator = plt.matplotlib.ticker.LogLocator(numticks=2)
cb.update_ticks()
style.panel_label(ax, "(a)", x=-0.30, y=1.03)

# ---------------- (b) polar angular profile ----------------
axp = fig.add_subplot(gs[1], projection="polar")
th = np.linspace(0, 2 * np.pi, 33)
prof = np.concatenate([angle, angle[:1]])
axp.plot(th, prof, color=style.C["blue"], lw=1.3)
axp.fill(th, prof, color=style.C["blue"], alpha=0.18)
axp.set_theta_zero_location("N")
axp.set_theta_direction(-1)
axp.set_xticks(np.deg2rad([0, 90, 180, 270]))
axp.set_xticklabels(["0", "8", "16", "24"], fontsize=7)
axp.set_yticks([0.04, 0.08])
axp.set_yticklabels(["4%", "8%"], fontsize=6.2, color=style.C["grey"])
axp.set_rlabel_position(203)
axp.set_ylim(0, 0.135)
axp.tick_params(pad=-2.0)
axp.grid(color=style.C["lightgrey"], lw=0.5)
seam = -np.pi / 32
axp.plot([seam, seam], [0, 0.130], color=style.C["red"], lw=0.8, ls=(0, (2.2, 1.6)), zorder=6)
axp.plot([0], [angle[0]], "o", ms=2.4, color=style.C["blue"], zorder=7)
axp.plot([2 * np.pi * 31 / 32], [angle[31]], "o", ms=2.4, color=style.C["blue"], zorder=7)
axp.text(seam - 0.12, 0.030, "seam", ha="right", va="center",
         fontsize=6.6, color=style.C["red"])
axp.spines["polar"].set_linewidth(0.7)
axp.spines["polar"].set_color(style.C["grey"])
axp.text(0.5, -0.30, "angle bin (circular)", transform=axp.transAxes,
         ha="center", fontsize=7.4)
style.panel_label(axp, "(b)", x=-0.13, y=1.02)

# ---------------- (c) delay profile ----------------
axd = fig.add_subplot(gs[2])
n = np.arange(32)
axd.semilogy(n, delay, "o-", color=style.C["orange"], ms=2.3, lw=1.2, mfc="w", mew=0.9)
axd.set_xlabel("delay tap $n_\\tau$", labelpad=2.0)
axd.set_ylabel("normalized energy", labelpad=2.0)
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
