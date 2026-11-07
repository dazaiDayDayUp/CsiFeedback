"""Fig.3 AxisCodec 架构图：主流水线 + AxisBlock 展开细节。

上排：H → stem → AxisBlock×8 → FC → z →（线性基座 Pz / 残差 AxisBlock×8）→ + → Ĥ。
下排：单个 AxisBlock 的层顺序与参数量，含残差旁路。
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge

sys.path.insert(0, str(Path(__file__).parent))
import style
from figqa import qa

style.setup()
D = Path(__file__).resolve().parent.parent / "data"
g = np.load(D / "geometry.npz")

W, H = style.DBL, 2.45
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

BLUE = style.C["blue"]
ORANGE = style.C["orange"]
GREEN = style.C["green"]
INK = "#1C1C1C"
GREY = "#7F8C8D"


def rbox(x0, y0, x1, y1, fc="#F7F9FB", ec=GREY, lw=0.8, r=1.4, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(p0, p1, color=INK, lw=0.95, rad=0.0, z=4, ms=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


def circle_icon(cx, cy, r=2.4, color=BLUE):
    ax.add_patch(Circle((cx, cy), r, fill=False, ec=color, lw=0.85, zorder=5))
    for k in range(8):
        a = 2 * np.pi * k / 8
        ax.add_patch(Circle((cx + r * np.cos(a), cy + r * np.sin(a)), 0.22,
                            fc=color, ec="none", zorder=6))
    arrow((cx + r * 0.2, cy + r * 1.15), (cx - r * 0.8, cy + r * 1.15),
          color=color, lw=0.7, ms=5, rad=0.4)


def matrix_icon(x0, y0, w=4.4, h=4.6, color=ORANGE):
    for i in range(4):
        for j in range(4):
            fc = color if (i + j) % 3 == 0 else "#F5E6DA"
            ax.add_patch(plt.Rectangle((x0 + j * w / 4, y0 + i * h / 4), w / 4, h / 4,
                                       fc=fc, ec="white", lw=0.3, zorder=5))
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=color, lw=0.8, zorder=6))


def mlp_icon(cx, cy, color=GREY):
    for dx, hh in ((-1.6, 2.0), (0.0, 3.4), (1.6, 2.0)):
        ax.add_patch(plt.Rectangle((cx + dx - 0.45, cy - hh / 2), 0.9, hh,
                                   fc=color, ec="none", alpha=0.75, zorder=5))


def thumb(x0, y0, w, h, label):
    axi = fig.add_axes([x0 / 100, y0 / 100, w / 100, h / 100])
    axi.imshow(g["mean_map"][:, :8], aspect="auto", origin="lower", cmap="magma",
               interpolation="nearest")
    axi.set_xticks([])
    axi.set_yticks([])
    for s in axi.spines.values():
        s.set_edgecolor(GREY)
        s.set_linewidth(0.7)
    ax.text(x0 + w / 2, y0 - 2.4, label, ha="center", va="top", fontsize=9.0)


# ==================== 主流水线 ====================
thumb(2.0, 62.0, 8.5, 28.0, "$H$")
ax.text(6.25, 51.5, "$2\\times32\\times32$", ha="center", fontsize=6.8, color=INK)
arrow((11.0, 76.0), (13.4, 76.0))
rbox(13.8, 66.0, 19.6, 86.0, fc="#F2F4F6", ec=GREY)
ax.text(16.7, 79.0, "stem", ha="center", fontsize=7.6)
ax.text(16.7, 72.0, "$3\\times3$ conv\nLN + GELU", ha="center", va="center", fontsize=6.8, color=INK)

arrow((20.0, 76.0), (22.6, 76.0))
for dx, dy in ((1.6, 2.0), (0.8, 1.0)):
    rbox(23.2 + dx, 64.0 + dy, 36.6 + dx, 86.0 + dy, fc="#EAF1F7", ec=BLUE, lw=0.6)
rbox(23.2, 64.0, 36.6, 86.0, fc="#F5F9FD", ec=BLUE, lw=0.9)
ax.text(29.9, 78.5, "AxisBlock $\\times 8$", ha="center", fontsize=8.0, fontweight="bold", color=BLUE)
ax.text(29.9, 70.5, "angle $\\circlearrowright$  /  delay dense", ha="center", fontsize=6.8, color=INK)
arrow((37.4, 76.0), (40.0, 76.0))
rbox(40.4, 66.0, 46.0, 86.0, fc="#F2F4F6", ec=GREY)
ax.text(43.2, 79.0, "FC", ha="center", fontsize=7.6)
ax.text(43.2, 72.0, "$2048\\rightarrow D$", ha="center", fontsize=6.8, color=INK)

arrow((46.4, 76.0), (48.6, 76.0))
ax.add_patch(FancyBboxPatch((49.0, 64.0), 2.6, 24.0, boxstyle="round,pad=0,rounding_size=0.6",
                            fc="#E8EEF4", ec=GREY, lw=0.7, zorder=2))
for y in np.linspace(66.0, 86.0, 8):
    ax.plot([49.3, 51.3], [y, y], color=GREY, lw=0.35, zorder=3)
ax.text(50.3, 90.5, "$z$", ha="center", fontsize=9.0)
ax.text(50.3, 60.0, "$D=2048/\\mathrm{CR}$", ha="center", fontsize=6.8)

# 解码器分支
arrow((51.6, 76.0), (54.4, 76.0))
ax.add_patch(Circle((54.8, 76.0), 0.55, fc=INK, ec="none", zorder=6))
rbox(57.4, 78.0, 69.6, 94.0, fc="#F1F8F2", ec=GREEN, lw=0.9)
ax.text(63.5, 88.5, "linear base", ha="center", fontsize=7.6, color=GREEN)
ax.text(63.5, 82.5, "$Pz$", ha="center", fontsize=8.4, color=GREEN)
rbox(57.4, 52.0, 69.6, 68.0, fc="#FDF6F0", ec=ORANGE, lw=0.9)
ax.text(63.5, 63.0, "residual trunk", ha="center", fontsize=7.6, color=ORANGE)
ax.text(63.5, 57.0, "AxisBlock $\\times 8$", ha="center", fontsize=7.4, color=INK)
arrow((54.8, 76.0), (57.2, 86.0), color=GREEN, lw=1.0)
arrow((54.8, 76.0), (57.2, 60.0), color=ORANGE, lw=1.0)
ax.add_patch(Circle((73.4, 76.0), 2.2, fc="white", ec=INK, lw=0.9, zorder=6))
ax.text(73.4, 76.0, "$+$", ha="center", va="center", fontsize=10.0, zorder=7)
arrow((69.8, 86.0), (72.6, 78.0), color=GREEN, lw=1.0)
arrow((69.8, 60.0), (72.6, 73.6), color=ORANGE, lw=1.0)
ax.text(73.4, 69.5, "clip $[0,1]$", ha="center", fontsize=6.8, color=INK)
thumb(79.0, 62.0, 8.5, 28.0, "$\\hat H$")
arrow((75.8, 76.0), (78.4, 76.0))
ax.text(66.0, 47.0, "$\\hat H = Pz + g_\\theta(z)$", ha="center", fontsize=7.8, color=INK)

# ==================== AxisBlock 展开 ====================
rbox(2.0, 6.0, 97.0, 38.0, fc="#FBFBFC", ec="#B9BEC3", lw=0.7, r=1.6)
ax.text(4.6, 31.5, "AxisBlock", ha="left", fontsize=8.2, fontweight="bold", color=INK)
ax.text(41.0, 33.0, "per-position LN; global mixing on both axes", ha="left", fontsize=6.6, color=GREY)

cy = 17.0
mods = [
    (12.0, 19.0, "LN", "#F2F4F6", GREY, None),
    (21.4, 34.0, "circular angle mix", "#F0F5FB", BLUE, "circle"),
    (36.4, 42.0, "GELU", "#F2F4F6", GREY, None),
    (44.4, 57.0, "dense delay mix", "#FDF6F0", ORANGE, "matrix"),
    (59.4, 65.0, "GELU", "#F2F4F6", GREY, None),
    (67.4, 80.0, "$1\\times1$ MLP", "#F2F4F6", GREY, "mlp"),
    (82.4, 88.0, "LN", "#F2F4F6", GREY, None),
]
for x0, x1, name, fc, ec, icon in mods:
    rbox(x0, cy - 5.0, x1, cy + 5.0, fc=fc, ec=ec, lw=0.85)
    ax.text((x0 + x1) / 2, cy - 0.4, name, ha="center", va="center", fontsize=7.2, color=ec)
    if icon == "circle":
        circle_icon((x0 + x1) / 2, cy + 8.6, r=2.1, color=BLUE)
        ax.text((x0 + x1) / 2, cy - 8.6, "$w_c\\in\\mathbb{C}^{17}$ (32 real/ch)", ha="center", fontsize=6.6, color=BLUE)
    elif icon == "matrix":
        matrix_icon((x0 + x1) / 2 - 2.0, cy + 6.4, w=4.0, h=4.0, color=ORANGE)
        ax.text((x0 + x1) / 2, cy - 8.6, "$W_\\tau\\in\\mathbb{R}^{32\\times32}$", ha="center", fontsize=6.6, color=ORANGE)
    elif icon == "mlp":
        mlp_icon((x0 + x1) / 2, cy + 8.6, color=GREY)
        ax.text((x0 + x1) / 2, cy - 8.6, "$C\\rightarrow 2C\\rightarrow C$", ha="center", fontsize=6.6, color=GREY)
for (_, x1, *_), (nx0, *_rest) in zip(mods[:-1], mods[1:]):
    arrow((x1 + 0.25, cy), (nx0 - 0.25, cy), lw=0.8, ms=5)
ax.text(93.0, cy, "$+$", ha="center", va="center", fontsize=11.0, color=INK)
arrow((88.4, cy), (91.6, cy), lw=0.8, ms=5)
# 残差旁路
ax.text(4.6, 16.8, "$x$", fontsize=8.2, color=INK, ha="center")
arrow((5.4, cy), (11.4, cy), lw=0.8, ms=5)
arrow((93.0, cy + 3.0), (95.4, cy + 3.0), lw=0.8, ms=5)
ax.text(96.2, 16.8, "$x + \\mathrm{LN}(\\cdot)$", fontsize=7.2, color=INK, ha="left")
ax.plot([4.6, 4.6, 93.0, 93.0], [cy + 3.0, 29.0, 29.0, cy + 3.0], color=GREY, lw=0.7, ls=(0, (2, 1.6)), zorder=2)

for ext in ("png", "pdf"):
    fig.savefig(Path(__file__).resolve().parent.parent / "figures" / f"fig3_arch.{ext}")
qa(fig, "Fig.3 arch")
