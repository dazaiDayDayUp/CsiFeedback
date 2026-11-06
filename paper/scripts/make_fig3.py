"""Fig.3 架构图：圆周角度混合 + 稠密时延混合 + 线性基座/非线性残差解码器。"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

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

BLUE, ORANGE, GREEN, GREY = style.C["blue"], style.C["orange"], style.C["green"], style.C["grey"]


def rbox(x0, y0, x1, y1, fc="#F7F9FB", ec=GREY, lw=0.8, r=1.6, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(p0, p1, color="#1C1C1C", lw=0.9, style_="-|>", ls="-", rad=0.0, z=3, mutation_scale=7, shrinkA=0.0, shrinkB=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style_, mutation_scale=mutation_scale, lw=lw,
                                 color=color, ls=ls, shrinkA=shrinkA, shrinkB=shrinkB,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


def circle_mixer(cx, cy, r=5.2, color=BLUE):
    ax.add_patch(Circle((cx, cy), r, fill=False, ec=color, lw=0.9, zorder=4))
    n = 10
    for k in range(n):
        a = 2 * np.pi * k / n
        ax.add_patch(Circle((cx + r * np.cos(a), cy + r * np.sin(a)), 0.55,
                            fc=color, ec="none", zorder=5))
    for k in (0, 3, 6):
        a0 = 2 * np.pi * (k + 0.35) / n
        a1 = 2 * np.pi * (k + 0.9) / n
        arrow((cx + r * 1.28 * np.cos(a0), cy + r * 1.28 * np.sin(a0)),
              (cx + r * 1.28 * np.cos(a1), cy + r * 1.28 * np.sin(a1)),
              color=color, lw=0.8, mutation_scale=5)


def dense_mixer(x0, y0, w=13.0, h=9.0, color=ORANGE, n=5):
    xs0, xs1 = x0, x0 + w
    ys = np.linspace(y0, y0 + h, n)
    for y in ys:
        ax.add_patch(Circle((xs0, y), 0.5, fc=color, ec="none", zorder=5))
        ax.add_patch(Circle((xs1, y), 0.5, fc=color, ec="none", zorder=5))
        for y2 in ys:
            ax.plot([xs0, xs1], [y, y2], color=color, lw=0.18, alpha=0.55, zorder=4)


# ---------------- input / output images ----------------
for x0, key, title in ((1.0, "in", "$H$"), (85.0, "out", "$\\hat H$")):
    axi = fig.add_axes([x0 / 100, 30 / 100 * (H / H) * 0.999, 12.5 / 100, 34 / 100 * 0.999])
    axi.imshow(g["mean_map"][:, :8], aspect="auto", origin="lower", cmap="magma",
               interpolation="nearest")
    axi.set_xticks([]); axi.set_yticks([])
    for s in axi.spines.values():
        s.set_edgecolor(GREY); s.set_linewidth(0.7)
    # 放到正确位置（add_axes 用 figure 分数坐标）
    axi.set_position([x0 / 100, 0.34, 0.125, 0.42])
    ax.text(x0 + 6.2, 24, title, ha="center", fontsize=9.5)
ax.text(7.2, 20.0, "$32\\times32$", ha="center", fontsize=7.2, color="#33363A")

# ---------------- encoder ----------------
rbox(17.2, 8, 44.0, 92)
ax.text(30.6, 86, "8 $\\times$ encoder blocks", ha="center", fontsize=8.2, fontweight="bold")
circle_mixer(26.5, 66)
ax.text(39.0, 66, "circular\nangle mix", ha="center", va="center", fontsize=7.2, color=BLUE)
ax.text(39.0, 56.5, "32 params/ch", ha="center", va="center", fontsize=7.7, color=BLUE)
arrow((26.5, 59.6), (26.5, 32.6), color="#33363A", lw=0.9, mutation_scale=6)
ax.text(27.6, 46.0, "LN, GELU", ha="left", va="center", fontsize=7.6, color="#33363A")
dense_mixer(21.0, 22.0)
ax.text(39.0, 27.5, "dense\ndelay mix", ha="center", va="center", fontsize=7.2, color=ORANGE)
ax.text(39.0, 18.0, "$32^2$ params", ha="center", va="center", fontsize=7.7, color=ORANGE)

# ---------------- latent ----------------
lat_x = 48.5
ax.add_patch(FancyBboxPatch((lat_x, 26), 3.0, 48, boxstyle="round,pad=0,rounding_size=0.8",
                            fc="#E8EEF4", ec=GREY, lw=0.7, zorder=2))
for y in np.linspace(30, 70, 9):
    ax.plot([lat_x + 0.3, lat_x + 2.7], [y, y], color=GREY, lw=0.35, zorder=3)
ax.text(lat_x + 1.5, 78, "$z$", ha="center", fontsize=9.5)
ax.text(lat_x + 1.5, 20.5, "$D=2048/\\mathrm{CR}$", ha="center", fontsize=7.6)
ax.text(lat_x + 1.5, 16.0, "(512 at CR $1/4$)", ha="center", fontsize=7.7, color="#33363A")

# ---------------- decoder ----------------
rbox(55.5, 8, 82.5, 92)
ax.text(69.0, 86, "Decoder", ha="center", fontsize=8.2, fontweight="bold")

# 输入总线
arrow((52.0, 50), (57.0, 50), lw=1.0)
ax.plot([57.8, 57.8], [38.0, 62.0], color="#1C1C1C", lw=1.0, zorder=3)
for yy in (38.0, 50.0, 62.0):
    ax.add_patch(Circle((57.8, yy), 0.55, fc="#1C1C1C", ec="none", zorder=6))

# 上支路：线性基座
arrow((57.8, 62.0), (77.0, 52.0), color=GREEN, lw=1.1, rad=-0.12)
ax.text(66.0, 63.4, "linear base $Pz$", ha="center", fontsize=8.2, color=GREEN)

# 下支路：非线性残差
rbox(58.6, 31.5, 73.2, 44.5, fc="#FDF6F0", ec=ORANGE, lw=0.8, r=1.2)
circle_mixer(62.6, 38.2, r=2.6, color=BLUE)
dense_mixer(67.4, 35.2, w=4.6, h=6.0, color=ORANGE, n=4)
ax.text(65.9, 27.2, "8 $\\times$ residual blocks", ha="center", fontsize=7.7)
arrow((57.8, 38.0), (58.6, 38.0), lw=1.0)
arrow((73.2, 38.0), (77.24, 47.72), color="#1C1C1C", lw=1.0)
ax.text(75.4, 41.0, "$g_\\theta(z)$", ha="left", fontsize=7.9, color="#33363A")

# 合并与输出
ax.add_patch(Circle((78.0, 50), 2.4, fc="white", ec="#1C1C1C", lw=0.9, zorder=6))
ax.text(78.0, 50, "$+$", ha="center", va="center", fontsize=10, zorder=7)
arrow((80.4, 50), (84.7, 50), color="#1C1C1C", lw=1.0)

# ---------------- 流程箭头 ----------------
arrow((14.0, 50), (16.8, 50), lw=1.0)
arrow((44.4, 50), (48.0, 50), lw=1.0)


# 说明：边界条件
ax.text(30.6, 3.2, "angle: circular (periodic)   |   delay: zero-padded (half-line)",
        ha="center", fontsize=7.2, color="#1C1C1C")
ax.text(69.0, 3.2, "$\\hat H = Pz + g_\\theta(z)$, clipped to $[0,1]$",
        ha="center", fontsize=7.9)

for ext in ("png", "pdf"):
    fig.savefig(Path(__file__).resolve().parent.parent / "figures" / f"fig3_arch.{ext}")
qa(fig, "Fig.3 arch")
