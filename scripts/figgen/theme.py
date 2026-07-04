"""配图设计系统（matplotlib）——scAtlasVAE 复现报告统一视觉。

所有报告配图（结构图 + 数据图）都从这里取设计令牌与组件，保证风格统一、专业、好看。
- 中文：自动探测并注册 Noto Sans CJK 字体；SVG 导出时把文字转为矢量路径（svg.fonttype='path'），
  这样在任何机器/浏览器打开都不依赖字体、不会出现方框。
- 坐标：采用"左上角为原点、y 向下"的排版坐标（与直觉一致），helpers 内部处理翻转。
- 产物：每张图同时导出 reports/<name>.svg（报告嵌入用）与 scratch 里的 <name>.png（我目检用）。

用法见 build_structures.py / build_data.py。
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ----------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
PNG_DIR = os.environ.get(
    "FIG_PNG_DIR",
    "/tmp/claude-1000/-home-vpnadmin-PRO1/874d7d0f-bb66-4795-8b46-dc5fb739b0cb/scratchpad/figpng",
)
os.makedirs(PNG_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 中文字体探测与注册
# ----------------------------------------------------------------------
def setup_cjk():
    """找到一款中文字体、注册进 matplotlib，返回字体族名。

    优先用本仓库 figgen/fonts/ 下从 Noto CJK .ttc 提取的**简体 SC** 面
    （matplotlib 无法直接从 .ttc 选 SC，故预先提取），保证简体字形正确。
    """
    local = os.path.join(os.path.dirname(__file__), "fonts")
    for fname in ("NotoSansCJKsc-Regular.otf", "NotoSansCJKsc-Bold.otf"):
        fp = os.path.join(local, fname)
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
            except Exception:
                pass
    if os.path.exists(os.path.join(local, "NotoSansCJKsc-Regular.otf")):
        try:
            name = fm.FontProperties(
                fname=os.path.join(local, "NotoSansCJKsc-Regular.otf")).get_name()
            _apply_font(name)
            return name
        except Exception:
            pass
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC",
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Source Han Sans SC",
        "Droid Sans Fallback",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in installed:
            _apply_font(c)
            return c
    # 按文件名兜底搜一遍常见路径
    for path in fm.findSystemFonts(fontpaths=None, fontext="ttf") + \
            fm.findSystemFonts(fontpaths=None, fontext="otf"):
        low = path.lower()
        if any(k in low for k in ["notosanscjk", "notoserifcjk", "wqy", "sourcehan", "droidsansfallback"]):
            try:
                fm.fontManager.addfont(path)
                name = fm.FontProperties(fname=path).get_name()
                _apply_font(name)
                return name
            except Exception:
                continue
    _apply_font("DejaVu Sans")
    return "DejaVu Sans"


def _apply_font(name):
    plt.rcParams["font.family"] = [name, "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"   # 文字转矢量路径，脱离字体依赖
    plt.rcParams["pdf.fonttype"] = 42


FONT = setup_cjk()

# ----------------------------------------------------------------------
# 设计令牌（配色 / 字号）
# ----------------------------------------------------------------------
INK = "#1b2432"      # 主文字（近黑蓝）
MUTED = "#5b6577"    # 次要文字
FAINT = "#8791a1"    # 弱文字 / 箭头
BG = "#ffffff"
PANEL = "#fbfcfe"    # 浅底面板
GRID = "#e6eaf1"

# 语义色板：每类一组 face(浅底)/edge(描边)/ink(强调)
PAL = {
    "input":     dict(face="#f4f6f9", edge="#c4ccd8", ink="#54606f"),
    "encoder":   dict(face="#eaf2ff", edge="#5b8bd6", ink="#2f6fd0"),
    "latent":    dict(face="#f1ecfb", edge="#8b6fd0", ink="#6a49c0"),
    "decoder":   dict(face="#e8f5ec", edge="#57b473", ink="#2f8f52"),
    "loss":      dict(face="#fff6e6", edge="#e3ba57", ink="#b9821f"),
    "cls":       dict(face="#fdeede", edge="#e39a63", ink="#c56f2a"),
    "batch":     dict(face="#eef1f5", edge="#9aa4b2", ink="#4a5563"),
    "accentA":   dict(face="#e6f6f8", edge="#4bb1bd", ink="#1f8c99"),  # teal
    "accentB":   dict(face="#fdeef2", edge="#e08aa4", ink="#c04f72"),  # rose
}

# 字号
FS_TITLE = 19
FS_SUB = 12.5
FS_BOXT = 13.5
FS_BOXS = 10.8
FS_ANN = 10.2
FS_SMALL = 9.4
FS_TAG = 10.5

# ----------------------------------------------------------------------
# 画布与组件
# ----------------------------------------------------------------------
def canvas(w=1000, h=560, dpi=150):
    """建一张以 (0,0) 左上角为原点、y 向下的排版画布。返回 (fig, ax)。
    w/h 以"排版单位"计（≈像素），1 单位 ≈ 1/100 英寸。"""
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)          # 翻转 y：顶部在上
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), w, h, facecolor=BG, edgecolor="none", zorder=-10))
    return fig, ax


def title(ax, x, y, main, sub=None):
    ax.text(x, y, main, fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")
    if sub:
        ax.text(x, y + 24, sub, fontsize=FS_SUB, color=MUTED, va="top")


def card(ax, x, y, w, h, kind="input", title_t="", subs=None, radius=12,
         shadow=True, title_fs=FS_BOXT, lw=1.5, zorder=3):
    """圆角卡片 + 可选标题/多行副文字 + 轻投影。subs 为字符串列表。"""
    p = PAL[kind]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        mutation_aspect=1, linewidth=lw,
        facecolor=p["face"], edgecolor=p["edge"], zorder=zorder,
    )
    if shadow:
        box.set_path_effects([pe.withSimplePatchShadow(
            offset=(0.8, 1.6), shadow_rgbFace="#1b2432", alpha=0.12, rho=0.5)])
    ax.add_patch(box)
    cx = x + w / 2
    subs = subs or []
    # 垂直排布：标题 + 副文字
    n = len(subs)
    if title_t and n:
        ax.text(cx, y + h / 2 - (n * 8), title_t, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=p["ink"], zorder=zorder + 1)
        for i, s in enumerate(subs):
            ax.text(cx, y + h / 2 - (n * 8) + 19 + i * 15.5, s, ha="center", va="center",
                    fontsize=FS_BOXS, color=MUTED, zorder=zorder + 1)
    elif title_t:
        ax.text(cx, y + h / 2, title_t, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=p["ink"], zorder=zorder + 1)
    else:
        for i, s in enumerate(subs):
            ax.text(cx, y + h / 2 - (n - 1) * 8 + i * 16, s, ha="center", va="center",
                    fontsize=FS_BOXS, color=MUTED, zorder=zorder + 1)
    return (cx, y + h / 2)


def arrow(ax, x1, y1, x2, y2, color=FAINT, lw=1.7, style="-|>", ls="-", zorder=2, rad=0.0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        lw=lw, color=color, linestyle=ls, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1,
    )
    ax.add_patch(a)


def label(ax, x, y, text, color=MUTED, fs=FS_ANN, ha="center", va="center", weight="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, fontweight=weight)


def badge(ax, x, y, w, h, text, kind="loss", fs=FS_TAG):
    """圆角药丸标记（用于'题眼'一类高亮）。"""
    p = PAL[kind]
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h/2}",
                       linewidth=1.3, facecolor=p["face"], edgecolor=p["edge"], zorder=5)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=p["ink"], zorder=6)


def save(fig, name):
    """导出 reports/<name>.svg（报告用）与 PNG（目检用）。"""
    svg = os.path.join(REPORTS_DIR, f"{name}.svg")
    png = os.path.join(PNG_DIR, f"{name}.png")
    fig.savefig(svg, format="svg", bbox_inches=None, pad_inches=0)
    fig.savefig(png, format="png", dpi=150, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return svg, png


if __name__ == "__main__":
    print("CJK font in use:", FONT)
    print("REPORTS_DIR:", REPORTS_DIR)
    print("PNG_DIR:", PNG_DIR)
