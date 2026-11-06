"""IEEE 论文图的统一风格（字体、配色、线宽、尺寸）。"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE 单栏 3.5 in、双栏 7.16 in
COL = 3.5
DBL = 7.16

# 色盲友好配色（Okabe–Ito 变体，饱和度略降以适配印刷）
C = {
    "blue": "#0B5FA5",
    "orange": "#E07B39",
    "green": "#2E8B57",
    "purple": "#6A4C93",
    "red": "#C0392B",
    "grey": "#7F8C8D",
    "lightgrey": "#D5DBDB",
    "black": "#1C1C1C",
}


def setup():
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8.4,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8,
        "legend.fontsize": 7.4,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.4,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.1,
        "legend.labelspacing": 0.25,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
        "hatch.linewidth": 0.45,
        "ps.fonttype": 42,
    })


def panel_label(ax, text, x=-0.14, y=1.04, **kw):
    """统一的 (a)/(b) 面板标号。"""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", **kw)
