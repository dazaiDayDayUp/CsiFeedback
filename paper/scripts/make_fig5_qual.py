"""Fig.6 定性重建：同一样本在 CR1/4、CR1/16 下的幅度图与误差差图。

样本按 AxisCodec 在 CR1/4 测试集上的逐样本 NMSE 中位数选取，避免挑图。
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import style
from figqa import qa

from csifeedback.data.cost2100 import Cost2100Dataset
from csifeedback.models import get_model_from_config
from csifeedback.utils.checkpoint import load_checkpoint, load_model_state_dict
from csifeedback.utils.config import load_config

MAIN = Path("/home/wzh/myProjects/CsiFeedback")
WT = ROOT
AXIS_CKPT = {
    4: WT / "experiments/axis_codec/indoor_cr4/20261105_182644/checkpoints/best_val.pth",
    16: WT / "experiments/axis_codec/indoor_cr16/20261106_082212/checkpoints/best_val.pth",
}
STNET_CKPT = {
    4: MAIN / "experiments/stnet/indoor_cr4/20260619_064715/checkpoints/best_nmse.pth",
    16: MAIN / "experiments/stnet/indoor_cr16/20260620_030825/checkpoints/best_nmse.pth",
}
CFG = {
    ("axis", 4): WT / "csifeedback/configs/ideas/cylinder_codec/axis_codec_indoor_cr4_e1000.yaml",
    ("axis", 16): WT / "csifeedback/configs/ideas/cylinder_codec/axis_codec_indoor_cr16_e1000.yaml",
    ("stnet", 4): WT / "csifeedback/configs/baselines/stnet_indoor_cr4.yaml",
    ("stnet", 16): WT / "csifeedback/configs/baselines/stnet_indoor_cr16.yaml",
}


def load_model(kind, cr, device):
    cfg = load_config(str(CFG[(kind, cr)]))
    model = get_model_from_config(cfg.model).to(device)
    ckpt = load_checkpoint(str(AXIS_CKPT[cr] if kind == "axis" else STNET_CKPT[cr]),
                           map_location=str(device))
    load_model_state_dict(model, ckpt.model_state_dict)
    model.eval()
    return model


@torch.no_grad()
def infer(model, data, device, batch=200):
    return torch.cat([model(data[i:i + batch].to(device)).cpu()
                      for i in range(0, data.shape[0], batch)])


def ps_nmse(pred, gt):
    num = ((pred - gt) ** 2).sum(dim=(1, 2, 3))
    den = ((gt - 0.5) ** 2).sum(dim=(1, 2, 3))
    return (num / den).numpy()


def to_complex(x):
    return (x[0] - 0.5) + 1j * (x[1] - 0.5)


style.setup()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ds = Cost2100Dataset(str(MAIN / "data"), "indoor", "test")
x = ds.tensors[0]

models = {(k, cr): load_model(k, cr, device) for k in ("axis", "stnet") for cr in (4, 16)}
pred = {key: infer(m, x, device) for key, m in models.items()}
nmse = {key: ps_nmse(p, x) for key, p in pred.items()}
idx = int(np.argsort(nmse[("axis", 4)])[len(x) // 2])

gt = x[idx].numpy()
hgt = to_complex(gt)
peak = np.abs(hgt).max()
gt_db = 20 * np.log10(np.abs(hgt) / peak + 1e-9)

fig = plt.figure(figsize=(style.DBL, 2.80))
gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.865, bottom=0.225,
                      wspace=0.06, hspace=0.16)
mag_norm = PowerNorm(gamma=0.55, vmin=-40, vmax=0)
diff_norm = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12)
titles = ["Ground truth", "STNet", "AxisCodec"]
rows = [4, 16]
im_mag = im_diff = None
for r, cr in enumerate(rows):
    axes = [fig.add_subplot(gs[r, c]) for c in range(3)]
    for ax in axes:
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.set_yticks([0, 4, 8, 12, 15])
        ax.set_ylim(-0.5, 15.5)
        ax.tick_params(labelsize=7.2, length=2, pad=1.4)
        ax.axhline(3.5, color="w", lw=0.55, ls=(0, (3, 2)), alpha=0.9)
        for s in ax.spines.values():
            s.set_linewidth(0.55)
            s.set_visible(True)
    im_mag = axes[0].imshow(gt_db[:16], origin="lower", cmap="magma", norm=mag_norm,
                            interpolation="nearest", aspect="auto")
    e = {}
    for ax, kind in zip(axes[1:], ("stnet", "axis")):
        h = to_complex(pred[(kind, cr)][idx].numpy())
        db = 20 * np.log10(np.abs(h) / peak + 1e-9)
        im_mag = ax.imshow(db[:16], origin="lower", cmap="magma", norm=mag_norm,
                           interpolation="nearest", aspect="auto")
        e[kind] = 20 * np.log10(np.abs(h - hgt) / peak + 1e-9)
        val = 10 * np.log10(nmse[(kind, cr)][idx])
        ax.text(0.97, 0.95, f"NMSE {val:.1f} dB", transform=ax.transAxes, ha="right",
                va="top", fontsize=6.8, color="w",
                bbox=dict(boxstyle="round,pad=0.14", fc="black", ec="none", alpha=0.5))
    axes[0].set_ylabel("delay tap $n_\\tau$", fontsize=7.2)

    # GT 面板右缘画一条 3.5 tap 的短标注
    for ax in axes[1:]:
        ax.set_yticklabels([])
    if r == 0:
        for ax, t in zip(axes, titles):
            ax.set_title(t, fontsize=7.7, pad=3.2)
    else:
        for ax in axes:
            ax.set_xlabel("angle bin $n_a$", fontsize=7.6)
    for ax in axes:
        if r == 0:
            ax.set_xticklabels([])
    axes[0].text(0.97, 0.95, f"({chr(97 + r)})  CR 1/{cr}", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=7.4, color="w", fontweight="bold")

cax = fig.add_axes([0.30, 0.055, 0.42, 0.032])
cb = fig.colorbar(im_mag, cax=cax, orientation="horizontal")
cb.set_label("$|H|$ (dB re sample peak)", fontsize=6.8, labelpad=4.5)
cb.ax.tick_params(labelsize=6.8, length=2, pad=1.2)
cb.outline.set_linewidth(0.5)
cb.set_ticks([-40, -30, -20, -10, 0])


for ext in ("png", "pdf"):
    fig.savefig(Path(__file__).resolve().parent.parent / "figures" / f"fig5_qual.{ext}")
qa(fig, "Fig.5 qualitative")
print("median sample", idx, {k: float(10 * np.log10(v[idx])) for k, v in nmse.items()})
