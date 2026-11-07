"""图质检查：ASCII 预览 + 文字重叠/贴边/压线/marker 裁切检测。

用法（作为库）:
    from figqa import qa
    qa(fig, "Fig1")

检查项：
  OVERLAP        两个文字 bbox 显著相交
  SMALL-FONT     字号低于阈值
  EDGE-FLUSH     墨迹距画布边缘 < 3 px（印刷时像被裁切）
  TEXT-ON-INK    文字 bbox 下方（隐藏该文字后）仍有 >2% 墨迹（压线/压色块）
  MARKER-CLIPPED 数据点落在 axes 边界上，marker 会被裁掉一半
  OUT-OF-CANVAS  artist 超出 figure 画布
"""

import io

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402


def _ascii_lum(png_bytes, width=84, height=42, charset=" .:-=+*#%@"):
    from PIL import Image

    im = Image.open(io.BytesIO(png_bytes)).convert("L")
    im = im.resize((width, height), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = 1.0 - a  # 墨量越高字符越重
    idx = np.clip((a * (len(charset) - 1)).round().astype(int), 0, len(charset) - 1)
    return "\n".join("".join(charset[i] for i in row) for row in idx), float(a.mean())


def _texts(fig):
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        out.append((t, bb))
    return out


def _render_ink(fig, dpi=200):
    """把 figure 渲染为灰度墨迹图（1=有墨）。"""
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None, pad_inches=0)
    buf.seek(0)
    a = np.asarray(Image.open(buf).convert("L"), dtype=np.float32)
    return (a < 245).astype(np.float32), a.shape


def _seg_hits_rect(p, q, bb):
    """Liang-Barsky：线段 pq 是否与矩形 bb 相交。"""
    x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if abs(pp) < 1e-12:
            if qq < 0:
                return False
        else:
            r = qq / pp
            if pp < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
    return True


def _line_through_text(fig, texts):
    """压线检测：任意数据折线是否穿过文字 bbox（收缩 15% 的核心区）。"""
    issues = []
    for t, bb in texts:
        pad = 0.15 * min(bb.width, bb.height)
        core = type(bb).from_extents(bb.x0 + pad, bb.y0 + pad, bb.x1 - pad, bb.y1 - pad)
        hit = False
        for ax in fig.axes:
            for line in ax.lines:
                if not line.get_visible():
                    continue
                xy = line.get_xydata()
                if len(xy) < 2:
                    continue
                disp = ax.transData.transform(xy)
                for i in range(len(disp) - 1):
                    if _seg_hits_rect(disp[i], disp[i + 1], core):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            issues.append((1.0, t.get_text()[:26]))
    return issues


def _marker_clipped(fig, tol_pt=0.8):
    issues = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        bb = ax.bbox
        for line in ax.get_lines():
            if line.get_marker() in (None, "None", "", " ") or not line.get_visible():
                continue
            xy = line.get_xydata()
            if len(xy) == 0:
                continue
            disp = ax.transData.transform(xy)
            ms = float(np.atleast_1d(line.get_markersize())[0]) / 2
            near = ((np.abs(disp[:, 0] - bb.x0) < ms + tol_pt) | (np.abs(disp[:, 0] - bb.x1) < ms + tol_pt)
                    | (np.abs(disp[:, 1] - bb.y0) < ms + tol_pt) | (np.abs(disp[:, 1] - bb.y1) < ms + tol_pt))
            outside = ((disp[:, 0] < bb.x0) | (disp[:, 0] > bb.x1)
                       | (disp[:, 1] < bb.y0) | (disp[:, 1] > bb.y1))
            near = near & ~outside
            if near.any():
                issues.append((int(near.sum()), line.get_label()[:20]))
    return issues


def qa(fig, name="fig", width=84, height=42, min_pt=6.5):
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", pad_inches=0.04)
    art, ink = _ascii_lum(buf.getvalue(), width, height)

    print(f"\n===== {name}: ASCII ({width}x{height}), ink={ink*100:.1f}% =====")
    print(art)

    fig.canvas.draw()
    figbb = fig.bbox
    texts = _texts(fig)
    problems = []

    # 字号
    for t, bb in texts:
        if t.get_fontsize() < min_pt:
            problems.append(f"SMALL-FONT {t.get_fontsize():.1f}pt: {t.get_text()[:30]!r}")

    # 两两重叠
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (t1, b1), (t2, b2) = texts[i], texts[j]
            dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if dx > 1.5 and dy > 1.5:
                area = dx * dy
                if area > 0.18 * min(b1.width * b1.height, b2.width * b2.height):
                    problems.append(
                        f"OVERLAP {area:.0f}px²: {t1.get_text()[:22]!r} <> {t2.get_text()[:22]!r}"
                    )

    # 压线：文字下方仍有墨迹
    for _, txt in _line_through_text(fig, texts):
        problems.append(f"LINE-THROUGH-TEXT: {txt!r}")

    # marker 贴边被裁
    for n, lab in _marker_clipped(fig):
        problems.append(f"MARKER-CLIPPED x{n}: {lab!r}")

    # 墨迹贴边（tight 保存后距边 < 3 px）
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("L"))
    on = im < 245
    pads = {"top": int(np.argmax(on.any(axis=1))),
            "bottom": int(np.argmax(on.any(axis=1)[::-1])),
            "left": int(np.argmax(on.any(axis=0))),
            "right": int(np.argmax(on.any(axis=0)[::-1]))}
    for k, p in pads.items():
        if p < 3:
            problems.append(f"EDGE-FLUSH {k}: {p}px")

    if ink < 0.035:
        problems.append(f"TOO-EMPTY ink={ink*100:.1f}%")
    has_image = any(ax.get_images() for ax in fig.axes)
    if ink > 0.45 and not has_image:
        problems.append(f"TOO-DENSE ink={ink*100:.1f}%")

    if problems:
        print("---- issues ----")
        for p in problems[:18]:
            print("  *", p)
        if len(problems) > 18:
            print(f"  ... {len(problems)-18} more")
    else:
        print("---- no text/layout issues ----")
    return {"ink": ink, "issues": problems}
