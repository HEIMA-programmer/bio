"""生成全部"结构图"（架构 / 流程 / 概念）到 reports/*.svg，并输出 PNG 供目检。

运行：python3 build_structures.py [fig_name ...]   # 不带参数=全部
每个 fig_* 函数画一张图，风格统一取自 theme.py。
"""
import sys
import numpy as np
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle
import theme as T
from theme import PAL, INK, MUTED, FAINT


# ---- 通用小组件 ---------------------------------------------------------------
def panel(ax, x, y, w, h, fc=None, ec=None, r=12, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=1.2, facecolor=fc or T.PANEL,
                                edgecolor=ec or T.GRID, zorder=z))


def chip(ax, x, y, w, h, text, kind, fs=T.FS_ANN):
    p = PAL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                linewidth=1.3, facecolor=p["face"], edgecolor=p["edge"], zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=p["ink"], zorder=4)


def tag(ax, cx, y, text, kind="loss"):
    """小标签徽标（居中于 cx）。"""
    w = 12 + len(text) * 8.2
    T.badge(ax, cx - w / 2, y, w, 22, text, kind=kind, fs=T.FS_SMALL)


# ======================================================================
def fig_scatlasvae_architecture():
    fig, ax = T.canvas(1000, 560)
    T.title(ax, 40, 28, "scAtlasVAE 架构",
            "批不变编码器 · 批条件解码器 · ZINB 重构 · 半监督分类头")
    T.badge(ax, 182, 84, 204, 26, "题眼 · 只吃 X，不看 batch", kind="loss")
    T.arrow(ax, 284, 110, 284, 122, color=PAL["loss"]["edge"], style="-", ls=(0, (3, 2)), lw=1.3)

    T.card(ax, 36, 124, 118, 90, "input", "基因表达 X", ["原始计数矩阵", "N × 4000"])
    T.card(ax, 196, 124, 176, 90, "encoder", "批不变编码器", ["F(X) → μ, σ²", "log1p 后送入 MLP"])
    T.card(ax, 414, 124, 150, 90, "latent", "潜向量 z", ["z = μ + σ·ε", "N × 10"])
    T.card(ax, 606, 124, 176, 90, "decoder", "批条件解码器", ["F(z, B) → MLP", "batch 仅在此注入"])
    T.card(ax, 824, 116, 150, 106, "decoder", "ZINB 三头",
           ["scale×文库 = 均值 μ", "离散度 θ", "零膨胀门控 π", "重构 N × 4000"])

    T.arrow(ax, 154, 169, 194, 169)
    T.arrow(ax, 372, 169, 412, 169)
    T.label(ax, 393, 158, "μ,σ²", color=PAL["latent"]["ink"], fs=T.FS_SMALL)
    T.arrow(ax, 564, 169, 604, 169)
    T.arrow(ax, 782, 169, 822, 169)

    T.card(ax, 606, 300, 176, 60, "batch", "批次 B → 嵌入 (dim 8)", ["与 z 拼接后解码"])
    T.arrow(ax, 694, 300, 694, 216, color=PAL["batch"]["edge"])
    T.card(ax, 414, 300, 150, 68, "cls", "分类头（线性）", ["z → 细胞类型", "半监督 · 可选"])
    T.arrow(ax, 489, 214, 489, 298)

    panel(ax, 36, 430, 938, 92)
    T.label(ax, 56, 452, "总损失", color=INK, fs=13.5, ha="left", weight="bold")
    chip(ax, 56, 468, 262, 36, "① ZINB 负对数似然（重构）", "decoder")
    T.label(ax, 328, 486, "+", color=FAINT, fs=15)
    chip(ax, 344, 468, 236, 36, "② λ_KL · KL(z ‖ N(0,I))", "latent")
    T.label(ax, 590, 486, "+", color=FAINT, fs=15)
    chip(ax, 606, 468, 228, 36, "③ λ_ct · 交叉熵（分类）", "cls")
    T.label(ax, 846, 486, "λ_KL 预热缓升", color=FAINT, fs=T.FS_SMALL, ha="left")
    T.save(fig, "fig_scatlasvae_architecture")


# ======================================================================
def fig_pipeline_overview():
    fig, ax = T.canvas(1140, 400)
    T.title(ax, 40, 26, "复现全流程：五个阶段",
            "军师 VM 负责读/写/画/填预期 · 你的 4060 负责实跑/替换")
    stages = [
        ("input",   "① 环境搭建",     ["摸清陌生库", "torch 换 cu118"], "L0→L1"),
        ("encoder", "② 端到端整合",   ["预处理→整合→评测", "scib-metrics"], "L1"),
        ("latent",  "③ 核心 VAE 重写 ★", ["手写最小 VAE", "对照官方源码"], "L2 重点"),
        ("decoder", "④ 消融实验",     ["每次只改一个旋钮", "验证设计必要性"], "L3"),
        ("cls",     "⑤ 汇总报告",     ["组会汇报稿", "诚实声明局限"], "交付"),
    ]
    x0, w, gap, y, h = 40, 190, 22, 150, 118
    centers = []
    for i, (k, t, subs, tg) in enumerate(stages):
        x = x0 + i * (w + gap)
        tag(ax, x + w / 2, 108, tg, kind="loss" if "★" in t or "重点" in tg else "batch")
        T.card(ax, x, y, w, h, k, t, subs)
        centers.append((x, x + w))
        if i:
            T.arrow(ax, centers[i - 1][1] + 2, y + h / 2, x - 2, y + h / 2)
    panel(ax, 40, 312, 1060, 52)
    T.label(ax, 60, 338, "判定成功看结论与趋势（批次校正、亚型分离、方法相对排序），不追求与论文像素/数字一致。",
            color=MUTED, fs=T.FS_ANN, ha="left")
    T.save(fig, "fig_pipeline_overview")


# ======================================================================
def fig_repo_map():
    fig, ax = T.canvas(1140, 560)
    T.title(ax, 40, 26, "仓库地图：核心其实只有一个文件",
            "命令 find scatlasvae -name '*.py' | 按重要性分三层")
    bands = [
        ("核心（真正要精读）", "encoder", 96, [
            ("model/_gex_model.py", "2156 行 · 编码/解码/ZINB/损失/训练全在这", 470),
        ]),
        ("相关支撑（用到时翻）", "latent", 250, [
            ("utils/_loss.py", "212 · ZINB/KL/MMD 损失", 235),
            ("utils/_distributions.py", "311 · ZINB 分布", 250),
            ("model/_primitives.py", "845 · SAE/FC 积木", 235),
            ("preprocessing/", "742 · _preprocess.py", 220),
        ]),
        ("外围 / vendored（本次忽略）", "input", 404, [
            ("pipeline/ · tools/ · data/", "训练封装/UMAP/加载", 300),
            ("externals/taming/", "VQGAN，第三方，无关", 220),
            ("externals/tabnet/", "可选编码器，无关", 210),
        ]),
    ]
    for name, kind, y, items in bands:
        T.label(ax, 40, y + 4, name, color=INK, fs=12.5, ha="left", weight="bold")
        x = 40
        for t, sub, w in items:
            h = 96 if kind == "encoder" else 62
            T.card(ax, x, y + 22, w, h, kind, t, [sub],
                   title_fs=14 if kind == "encoder" else 12)
            x += w + 20
    T.badge(ax, 560, 96, 300, 30, "结论：读懂它 ≈ 读懂 scAtlasVAE", kind="loss")
    T.save(fig, "fig_repo_map")


# ======================================================================
def fig_paper_story():
    fig, ax = T.canvas(1160, 430)
    T.title(ax, 40, 26, "论文的科学故事（读 Fig 1 就能串起来）",
            "scAtlasVAE 是贯穿全流程的方法引擎；本次复现聚焦这台引擎")
    steps = [
        ("input",   "数据",     ["68 studies", "961 样本 · 42 疾病"]),
        ("encoder", "构建图谱", ["115 万", "CD8⁺ T 细胞"]),
        ("latent",  "18 个亚型", ["无监督聚类", "+ 人工注释"]),
        ("decoder", "3 个 Tex 亚型", ["GZMK⁺ / ITGAE⁺", "XBP1⁺"]),
        ("accentA", "TCR 克隆",  ["克隆扩增", "跨亚型分享"]),
        ("cls",     "迁移注释", ["query 数据", "自动打标签"]),
    ]
    x0, w, gap, y, h = 30, 172, 16, 170, 108
    xs = []
    for i, (k, t, subs) in enumerate(steps):
        x = x0 + i * (w + gap)
        T.card(ax, x, y, w, h, k, t, subs)
        xs.append((x, x + w))
        if i:
            T.arrow(ax, xs[i - 1][1] + 1, y + h / 2, x - 1, y + h / 2)
    panel(ax, 30, 322, 1100, 56)
    T.label(ax, 50, 350, "复现路线由此定：不追全 115 万图谱，而是复刻让这一切成立的“整合+迁移”方法内核（第 3 阶段手写）。",
            color=MUTED, fs=T.FS_ANN, ha="left")
    T.save(fig, "fig_paper_story")


# ======================================================================
def fig_batch_invariant_vs_scvi():
    fig, ax = T.canvas(1060, 500)
    T.title(ax, 40, 26, "为什么只有 scAtlasVAE 能 zero-shot 迁移",
            "关键差别只在一处：编码器看不看 batch")
    cols = ["方法", "编码器输入", "解码器", "重构"]
    rows = [
        ("scAtlasVAE", "F(X) 只吃 X", "F(z, B)", "ZINB", True),
        ("scVI / scANVI", "F(X, B, S)", "F(z, zₗ, B)", "ZINB", False),
        ("SCALEX", "F(X)", "F(z, B)", "BCE", False),
        ("scPoli", "F(X, B)", "F(z, B)", "ZINB", False),
    ]
    x0, y0 = 40, 92
    cw = [190, 220, 180, 120]
    rh = 52
    # 表头
    cx = x0
    for j, c in enumerate(cols):
        ax.text(cx + cw[j] / 2, y0 - 12, c, ha="center", va="center",
                fontsize=T.FS_ANN, color=MUTED, fontweight="bold")
        cx += cw[j]
    for i, (m, enc, dec, rec, hi) in enumerate(rows):
        y = y0 + i * (rh + 12)
        cx = x0
        vals = [m, enc, dec, rec]
        for j, v in enumerate(vals):
            hlcell = hi and j == 1
            k = "encoder" if hlcell else ("latent" if (hi and j == 0) else "input")
            chip(ax, cx + 4, y, cw[j] - 8, rh, v, k,
                 fs=T.FS_ANN if len(v) < 12 else T.FS_SMALL)
            cx += cw[j]
    # 右侧结论
    panel(ax, 40, 372, 980, 96, fc=PAL["loss"]["face"], ec=PAL["loss"]["edge"])
    T.label(ax, 60, 400, "编码器不依赖 batch 的后果", color=PAL["loss"]["ink"], fs=12.5, ha="left", weight="bold")
    T.label(ax, 60, 428, "新查询数据不必重训、不必改架构——直接过同一个编码器就映射进参考图谱（zero-shot）。",
            color=MUTED, fs=T.FS_ANN, ha="left")
    T.label(ax, 60, 450, "而 scVI 编码器吃了 batch，来新批次就得做“架构手术”或重训。",
            color=MUTED, fs=T.FS_ANN, ha="left")
    T.save(fig, "fig_batch_invariant_vs_scvi")


# ======================================================================
def fig_vae_dataflow_shapes():
    fig, ax = T.canvas(1120, 640)
    T.title(ax, 40, 26, "前向计算：张量形状怎么流动",
            "对照源码 encode()（_gex_model.py:964）与 decode()（:992）")
    # Row 1: encode
    T.label(ax, 40, 96, "encode()：只用 X", color=PAL["encoder"]["ink"], fs=12.5, ha="left", weight="bold")
    e = [
        ("input", "X", ["N×4000", "原始计数"]),
        ("input", "log1p(X)", ["N×4000"]),
        ("encoder", "SAE 编码器", ["→ h  N×128"]),
        ("latent", "z_mean / z_var", ["μ, σ²  N×10"]),
        ("latent", "rsample", ["z = μ+σε", "N×10"]),
    ]
    _snake_row(ax, e, y=118, h=82, x0=40, w=180, gap=26)
    # Row 2: decode
    T.label(ax, 40, 320, "decode()：上一行的 z 拼接 batch，再出 ZINB 三头",
            color=PAL["decoder"]["ink"], fs=12.5, ha="left", weight="bold")
    d = [
        ("latent", "z ⊕ batch_emb", ["N×(10+8)"]),
        ("decoder", "解码器 MLP", ["→ h'  N×128"]),
    ]
    _snake_row(ax, d, y=342, h=80, x0=40, w=200, gap=26)
    # three heads
    heads = [
        ("decoder", "scale (softmax)×文库", "→ μ 均值  N×4000", 470),
        ("decoder", "rate", "→ θ 离散度  N×4000", 534),
        ("decoder", "dropout", "→ π 门控  N×4000", 598),
    ]
    hx = 300
    for k, t, s, yy in heads:
        T.card(ax, hx, yy - 26, 330, 52, k, t, [s], title_fs=12)
        T.arrow(ax, 266, 382, hx - 4, yy, rad=-0.12)
    T.card(ax, 700, 470, 190, 128, "loss", "ZINB(μ, θ, π)", ["重构原始计数", "x̂  N×4000"])
    for yy in (470, 534, 598):
        T.arrow(ax, 630, yy, 698, 534, color=PAL["decoder"]["edge"], rad=0.05)
    T.save(fig, "fig_vae_dataflow_shapes")


def _snake_row(ax, items, y, h, x0, w, gap):
    xs = []
    for i, (k, t, subs) in enumerate(items):
        x = x0 + i * (w + gap)
        T.card(ax, x, y, w, h, k, t, subs, title_fs=12)
        xs.append((x, x + w))
        if i:
            T.arrow(ax, xs[i - 1][1] + 1, y + h / 2, x - 1, y + h / 2)
    return xs


# ======================================================================
def fig_semisupervised_transfer():
    fig, ax = T.canvas(1080, 520)
    T.title(ax, 40, 26, "半监督训练 + zero-shot 迁移",
            "编码器全程共享、不重训——这是迁移能力的根")
    # 训练
    T.label(ax, 40, 96, "训练（参考图谱）", color=INK, fs=12.5, ha="left", weight="bold")
    T.card(ax, 40, 116, 170, 78, "input", "参考数据", ["部分带标签"])
    T.card(ax, 250, 116, 150, 78, "encoder", "编码器", ["F(X)"])
    T.card(ax, 440, 116, 130, 78, "latent", "z", ["N×10"])
    T.card(ax, 610, 92, 190, 52, "cls", "分类头 · 交叉熵", ["仅对有标签细胞"])
    T.card(ax, 610, 162, 190, 52, "decoder", "解码器 · 重构", ["对全部细胞"])
    T.arrow(ax, 210, 155, 248, 155)
    T.arrow(ax, 400, 155, 438, 155)
    T.arrow(ax, 570, 145, 608, 122)
    T.arrow(ax, 570, 165, 608, 188)
    # 迁移
    T.label(ax, 40, 300, "迁移（新 query 数据）", color=INK, fs=12.5, ha="left", weight="bold")
    T.card(ax, 40, 320, 170, 78, "input", "新 query 数据", ["无标签 · 新批次"])
    T.card(ax, 250, 320, 150, 78, "encoder", "同一个编码器", ["❄ 不重训"])
    T.card(ax, 440, 320, 130, 78, "latent", "z", ["落入同一图谱"])
    T.card(ax, 610, 320, 190, 78, "cls", "分类头 → 预测标签", ["自动注释"])
    T.arrow(ax, 210, 359, 248, 359)
    T.arrow(ax, 400, 359, 438, 359)
    T.arrow(ax, 570, 359, 608, 359)
    # 强调同一编码器
    T.badge(ax, 250, 236, 150, 26, "同一套编码器", kind="loss")
    T.arrow(ax, 325, 236, 325, 196, color=PAL["loss"]["edge"], style="-", ls=(0, (3, 2)), lw=1.2)
    T.arrow(ax, 325, 262, 325, 318, color=PAL["loss"]["edge"], style="-", ls=(0, (3, 2)), lw=1.2)
    panel(ax, 40, 430, 1000, 66)
    T.label(ax, 60, 462, "因为编码器只吃 X（见架构图“题眼”），它对“来自哪个批次”一无所知，",
            color=MUTED, fs=T.FS_ANN, ha="left")
    T.label(ax, 60, 482, "所以任何新数据都能直接喂进来、映射到参考坐标系——无需微调。",
            color=MUTED, fs=T.FS_ANN, ha="left")
    T.save(fig, "fig_semisupervised_transfer")


# ======================================================================
def fig_ablation_design():
    fig, ax = T.canvas(1000, 460)
    T.title(ax, 40, 26, "消融 = 控制变量：每次只拧一个旋钮",
            "其余全按默认，才能把结论变化归因到这一个改动")
    T.card(ax, 380, 92, 240, 76, "latent", "默认配置",
           ["n_latent=10 · KL 预热", "batch 在解码器注入"])
    branches = [
        ("input",   "潜维度", ["n_latent → 2 / 50"], "预期：2 太小信息丢，50 无收益", 40),
        ("encoder", "KL 预热", ["关掉 / 缩短预热"], "预期：过早收紧→后验坍缩", 380),
        ("decoder", "batch 位置", ["移到编码器"], "预期：退化成 scVI，迁移变差", 720),
    ]
    for k, t, subs, exp, x in branches:
        T.arrow(ax, 500, 168, x + 120, 232, rad=0.0)
        T.card(ax, x, 232, 240, 76, k, t, subs)
        T.label(ax, x + 120, 336, exp, color=MUTED, fs=T.FS_SMALL)
    T.save(fig, "fig_ablation_design")


# ======================================================================
def fig_ae_vs_vae_latent_space():
    fig, ax = T.canvas(1000, 480)
    T.title(ax, 40, 26, "AE vs VAE 的潜空间",
            "左：AE 是孤立的点、点间留洞；右：VAE 是重叠的云、填满原点附近")
    rng = np.random.default_rng(7)
    cols = [PAL["encoder"]["ink"], PAL["decoder"]["ink"], PAL["cls"]["ink"], PAL["latent"]["ink"]]

    # 左：AE 面板
    panel(ax, 40, 96, 440, 356, fc="#ffffff", ec=T.GRID)
    T.label(ax, 60, 120, "自编码器 (AE)", color=INK, fs=12.5, ha="left", weight="bold")
    ae_centers = [(140, 180), (380, 170), (150, 380), (390, 380)]
    for c, ctr in zip(cols, ae_centers):
        pts = rng.normal(ctr, 26, size=(40, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=c, alpha=0.85, edgecolors="none", zorder=4)
    T.label(ax, 260, 285, "空洞", color=FAINT, fs=T.FS_SMALL)
    ax.add_patch(Circle((260, 285), 40, fill=False, ls=(0, (2, 2)), ec=FAINT, lw=1.1, zorder=3))

    # 右：VAE 面板
    panel(ax, 520, 96, 440, 356, fc="#ffffff", ec=T.GRID)
    T.label(ax, 540, 120, "变分自编码器 (VAE)", color=INK, fs=12.5, ha="left", weight="bold")
    ax.add_patch(Circle((740, 290), 150, fill=False, ls=(0, (3, 3)), ec=PAL["latent"]["edge"], lw=1.2, zorder=2))
    T.label(ax, 740, 150, "N(0, I) 有界区域", color=PAL["latent"]["ink"], fs=T.FS_SMALL)
    vae_centers = [(700, 250), (790, 250), (700, 330), (790, 330)]
    for c, ctr in zip(cols, vae_centers):
        ax.add_patch(Ellipse(ctr, 130, 110, facecolor=c, alpha=0.20, edgecolor=c, lw=1.0, zorder=3))
        pts = rng.normal(ctr, 22, size=(35, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=c, alpha=0.8, edgecolors="none", zorder=4)
    T.save(fig, "fig_ae_vs_vae_latent_space")


REGISTRY = {
    "fig_scatlasvae_architecture": fig_scatlasvae_architecture,
    "fig_pipeline_overview": fig_pipeline_overview,
    "fig_repo_map": fig_repo_map,
    "fig_paper_story": fig_paper_story,
    "fig_batch_invariant_vs_scvi": fig_batch_invariant_vs_scvi,
    "fig_vae_dataflow_shapes": fig_vae_dataflow_shapes,
    "fig_semisupervised_transfer": fig_semisupervised_transfer,
    "fig_ablation_design": fig_ablation_design,
    "fig_ae_vs_vae_latent_space": fig_ae_vs_vae_latent_space,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(REGISTRY)
    for n in names:
        REGISTRY[n]()
        print("built", n)
