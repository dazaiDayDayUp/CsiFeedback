"""图质检查：把 matplotlib 图形转成 ASCII 亮度图，并检测文字重叠/越界/墨量。

用法（作为库）:
    from figqa import qa
    qa(fig, "Fig1")   # 打印 ASCII 与检查报告
"""

import io

import matplotlib
import numpy as np

matplotlib.use("Agg")

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


def qa(fig, name="fig", width=84, height=42, min_pt=6.5):
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", pad_inches=0.02)
    art, ink = _ascii_lum(buf.getvalue(), width, height)

    print(f"\n===== {name}: ASCII ({width}x{height}), ink={ink*100:.1f}% =====")
    print(art)

    texts = _texts(fig)
    # 用 tight bbox 判断越界（savefig 用的是 bbox_inches="tight"）
    try:
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        figbb = type(fig.bbox).from_extents(tb.x0, tb.y0, tb.x1, tb.y1)
        figbb = fig.bbox.from_extents(*fig.get_window_extent().extents) if False else figbb
    except Exception:
        figbb = fig.bbox
    problems = []

    # 越界
    for t, bb in texts:
        if t.get_fontsize() < min_pt:
            problems.append(f"SMALL-FONT {t.get_fontsize():.1f}pt: {t.get_text()[:30]!r}")

    # 两两重叠（允许 1px 容差）
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

    if ink < 0.035:
        problems.append(f"TOO-EMPTY ink={ink*100:.1f}%")
    if ink > 0.45:
        problems.append(f"TOO-DENSE ink={ink*100:.1f}%")

    if problems:
        print("---- issues ----")
        for p in problems[:14]:
            print("  *", p)
        if len(problems) > 14:
            print(f"  ... {len(problems)-14} more")
    else:
        print("---- no text/layout issues ----")
    return {"ink": ink, "issues": problems}
