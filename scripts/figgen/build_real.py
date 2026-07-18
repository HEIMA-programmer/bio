"""用**真实实跑结果**重绘数据图，替换 build_data.py 的「示意/预期」占位。

与 build_data.py 的区别：那份用合成的示意数值、角落盖「示意」章；这份读 4060 实跑产物
（loss npz / scib-metrics csv / 处理好的 h5ad 里的嵌入），画真图、不盖示意章。

数据来源（默认在 bio/data/，可用 REAL_DATA_DIR 覆盖）：
    phase2_scatlasvae_loss.npz      ← phase2_run_scatlasvae.py 训练时保存（loss + λ_KL）
    phase2_benchmark_results.csv    ← phase2_benchmark_scib.py
    phase4_ablation_results.csv     ← phase4_ablations.py
    tcell_processed.h5ad            ← 含 obsm 各嵌入 + X_umap_*

运行（在 scib 环境，需 matplotlib + 中文字体，theme.py 已处理）：
    python build_real.py loss                     # 重绘 fig_phase2_loss_curve
    python build_real.py bench                     # 重绘 fig_phase2_benchmark_bars
    python build_real.py ablation                  # 重绘 fig_phase4_ablation_bars
    python build_real.py umap_integration          # 重绘 fig_phase2_integration_umap
    python build_real.py umap_compare              # 重绘 fig_phase3_umap_compare
    python build_real.py all
"""
import os
import sys
import hashlib
import json
import struct
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import theme as T
from theme import PAL, INK, MUTED, FAINT

plt.rcParams.update({
    "axes.edgecolor": "#c4ccd8", "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": "#eef1f6", "grid.linewidth": 1.0,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

DATA_DIR = os.environ.get("REAL_DATA_DIR",
                          os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")))
# F 项改造：结果图统一输出 PNG 到 reports/figures/（不再往 reports/ 顶层散落 SVG）。
FIG_DIR = os.path.join(T.REPORTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TARGET_SPECS = {
    "loss": ("fig_phase2_loss_curve", ["phase2_scatlasvae_loss.npz"]),
    "bench": ("fig_phase2_benchmark_bars", ["phase2_benchmark_results.csv"]),
    "ablation": ("fig_phase4_ablation_bars", ["phase4_ablation_results.csv"]),
    "umap_integration": ("fig_phase2_integration_umap", ["tcell_processed.h5ad"]),
    "umap_compare": ("fig_phase3_umap_compare", ["tcell_processed.h5ad"]),
    "bench_minimal": ("fig_phase5_minimal_bench", ["phase5_minimal_bench.csv"]),
    "transfer": (
        "fig_phase5_transfer",
        ["phase5_transfer_results_paper.csv", "phase5_transfer_results_patient_paper.csv",
         "phase5_fair_knn_results.csv"],
    ),
    "transfer_protocol_p": (
        "fig_phase5_transfer_patient_protocol",
        ["phase5_transfer_results_patient_paper.csv", "phase5_transfer_results_patient_fulltime.csv"],
    ),
    "invariance": (
        "fig_phase5_invariance",
        ["phase5_invariance_scatlasvae.csv", "phase5_invariance_scvi.csv"],
    ),
    "scalability": ("fig_phase5_scalability", ["phase5_scalability.csv"]),
    "cross_atlas": ("fig_phase5_cross_atlas", ["phase5_cross_atlas_head_alignment.csv"]),
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(path):
    with open(path, "rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"不是有效 PNG：{path}")
    return struct.unpack(">II", header[16:24])


def _write_figure_manifest(targets):
    """记录图、生成器和输入产物哈希，使“已重绘”可由门禁独立复核。"""
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    source_cache = {}
    figures = []
    for target in targets:
        figure_name, source_names = TARGET_SPECS[target]
        figure_path = os.path.join(FIG_DIR, f"{figure_name}.png")
        width, height = _png_dimensions(figure_path)
        source_keys = []
        for source_name in source_names:
            source_path = os.path.join(DATA_DIR, source_name)
            if not os.path.exists(source_path):
                raise FileNotFoundError(source_path)
            source_key = os.path.relpath(source_path, project_dir).replace("\\", "/")
            source_keys.append(source_key)
            if source_key not in source_cache:
                source_cache[source_key] = {
                    "bytes": os.path.getsize(source_path),
                    "sha256": _sha256(source_path),
                }
        figures.append({
            "target": target,
            "path": os.path.relpath(figure_path, project_dir).replace("\\", "/"),
            "bytes": os.path.getsize(figure_path),
            "width": width,
            "height": height,
            "sha256": _sha256(figure_path),
            "sources": source_keys,
        })

    generator_path = os.path.abspath(__file__)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "path": os.path.relpath(generator_path, project_dir).replace("\\", "/"),
            "sha256": _sha256(generator_path),
        },
        "sources": source_cache,
        "figures": figures,
    }
    manifest_path = os.path.join(DATA_DIR, "figure_manifest.json")
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, manifest_path)
    print("wrote", manifest_path)


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _save(fig, name):
    """只写 PNG 到 reports/figures/（结果图光栅化，GitHub 直接渲染、体积小、中文稳定）。"""
    png = f"{FIG_DIR}/{name}.png"
    fig.savefig(png, format="png", dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", png)


# ======================================================================
def loss_curve():
    """监督主模型的真实训练动态：总/重构损失下降 + λ_KL 预热 0→~1。"""
    d = np.load(os.path.join(DATA_DIR, "phase2_scatlasvae_loss.npz"))
    total = d["epoch_total_loss_list"]
    recon = d["epoch_reconstruction_loss_list"]
    klw = d["kl_weight"]
    ep = np.arange(1, len(total) + 1)

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.plot(ep, total, color=PAL["encoder"]["ink"], lw=2.2, label="总损失")
    ax.plot(ep, recon, color=PAL["decoder"]["ink"], lw=1.8, ls="--", label="重构损失(ZINB)")
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("训练损失（每 epoch 累加）", fontsize=10)
    ax.set_title(f"监督版 scAtlasVAE 训练损失曲线（实跑，{len(ep)} epoch）",
                 fontsize=13, fontweight="bold", pad=10)

    ax2 = ax.twinx()
    ax2.plot(ep, klw, color=PAL["loss"]["ink"], lw=2.0, label="KL 权重 λ_KL(预热)")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("KL 权重 λ_KL", color=PAL["loss"]["ink"], fontsize=10)
    ax2.tick_params(axis="y", colors=PAL["loss"]["ink"])
    ax2.grid(False)
    ax2.annotate(
        f"λ_KL 因 min(max_epoch,400) 截断，\n整个训练 0→~1 爬满，末轮≈{klw[-1]:.2f}\n"
        f"（旧稿误作\"只到0.18\"）",
        xy=(ep[-1], klw[-1]), xytext=(ep[int(len(ep) * 0.12)], 0.62),
        fontsize=8.8, color=PAL["loss"]["ink"],
        arrowprops=dict(arrowstyle="->", color=PAL["loss"]["ink"], lw=1))
    _clean(ax)
    ax2.spines["top"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="center right")
    _save(fig, "fig_phase2_loss_curve")


def _bars_from_csv(csv_name, order, title, outname, note=None):
    """从 scib-metrics 的结果 csv 画分组条形（每个嵌入的 批次校正/生物保留/总分）。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name), index_col=0)
    df = df[df.index != "Metric Type"]                 # 去掉描述行
    aggs = ["Batch correction", "Bio conservation", "Total"]
    zh = {"Batch correction": "批次校正", "Bio conservation": "生物保留", "Total": "总分 Overall"}
    order = [e for e in order if e in df.index]
    mcol = [PAL["input"]["ink"], PAL["accentA"]["ink"], PAL["encoder"]["ink"],
            PAL["cls"]["ink"], PAL["latent"]["ink"]]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(aggs)); w = 0.8 / len(order)
    for i, emb in enumerate(order):
        vals = [float(df.loc[emb, a]) for a in aggs]
        b = ax.bar(x + (i - (len(order) - 1) / 2) * w, vals, w,
                   color=mcol[i % len(mcol)], label=emb, edgecolor="white", linewidth=0.8)
        ax.bar_label(b, fmt="%.2f", fontsize=8, color=MUTED, padding=2)
    ax.set_xticks(x); ax.set_xticklabels([zh[a] for a in aggs], fontsize=10.5)
    ax.set_ylim(0, max(0.7, float(df[aggs].values.max()) * 1.18))
    ax.set_ylabel("分数（越高越好）", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    _clean(ax)
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=8.6, color=MUTED)
    _save(fig, outname)


def bench():
    # 阶段 5 · E2：同一数据、batch 与指标口径下的四方内部比较。
    _bars_from_csv(
        "phase2_benchmark_results.csv",
        ["X_pca", "X_scVI", "X_scAtlasVAE_unsup", "X_scAtlasVAE_sup"],
        "整合评测：PCA / scVI / scAtlasVAE(无监督) / scAtlasVAE(监督)（真实，scib-metrics）",
        "fig_phase2_benchmark_bars",
        note="注：scib-metrics 与论文旧 scib 数值不可直接比，只看内部相对排序；监督版更高的方向与论文一致。")


def bench_minimal():
    # 阶段 5 · E4：把手写最小 VAE 放上同一把标尺。
    _bars_from_csv(
        "phase5_minimal_bench.csv",
        ["X_pca", "X_scVI", "X_scAtlasVAE_sup", "X_minimal"],
        "手写最小 VAE 上标尺：与 PCA / scVI / 官方监督版并列（真实，scib-metrics）",
        "fig_phase5_minimal_bench",
        note="注：X_minimal = 从零手写的最小 scAtlasVAE。看它相对 PCA 与 scVI/官方落在哪，量化手写实现的水平。")


def transfer():
    """阶段 5 · E1：注释迁移的 acc/macroF1/AUROC 分组条形（按设计×方法）。
    scAtlasVAE 使用官方源码默认日程；kNN 使用真正 reference-only frozen encoder，
    不再把 full-data X_scVI 的 transductive 诊断画成公平基线。
    """
    import pandas as pd
    result_paths = [os.path.join(DATA_DIR, "phase5_transfer_results_paper.csv")]
    patient_path = os.path.join(DATA_DIR, "phase5_transfer_results_patient_paper.csv")
    if os.path.exists(patient_path):
        result_paths.append(patient_path)
    df = pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    df = df[df["method"] != "kNN on scVI latent"].copy()
    fair_path = os.path.join(DATA_DIR, "phase5_fair_knn_results.csv")
    if not os.path.exists(fair_path):
        raise FileNotFoundError(fair_path)
    fair = pd.read_csv(fair_path)
    fair = fair[fair["kind"].str.startswith("fair-inductive", na=False)].copy()
    fair["method"] = "scVI kNN (ref-only frozen)"
    df = pd.concat(
        [df, fair[["design", "method", "accuracy", "macro_f1", "macro_ovr_auc"]]],
        ignore_index=True,
    )
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "macro-F1"), ("macro_ovr_auc", "macro OVR-AUC")]
    designs = {
        "A": "设计A(随机5%)",
        "B": "设计B(整癌种UCEC)",
        "P": "设计P(整位patient)",
    }
    mcol = {"scAtlasVAE (zero-shot)": PAL["encoder"]["ink"],
            "scAtlasVAE (full-shot)": PAL["latent"]["ink"],
            "scVI kNN (ref-only frozen)": PAL["accentA"]["ink"]}
    dlist = [d for d in ["A", "B", "P"] if d in set(df["design"])]
    fig, axes = plt.subplots(1, len(dlist), figsize=(5.6 * len(dlist), 4.7), squeeze=False)
    for ax, d in zip(axes[0], dlist):
        sub = df[df["design"] == d]
        methods = list(sub["method"])
        x = np.arange(len(metrics)); w = 0.8 / max(1, len(methods))
        for i, mth in enumerate(methods):
            row = sub[sub["method"] == mth].iloc[0]
            vals = [float(row[k]) for k, _ in metrics]
            b = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w,
                       color=mcol.get(mth, MUTED), label=mth, edgecolor="white", linewidth=0.8)
            ax.bar_label(b, fmt="%.2f", fontsize=7.6, color=MUTED, padding=2)
        ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in metrics], fontsize=9.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(designs.get(d, d), fontsize=12, fontweight="bold", pad=8)
        ax.legend(fontsize=8.3, frameon=False, loc="upper left")
        _clean(ax)
    axes[0][0].set_ylabel("分数（越高越好）", fontsize=10)
    fig.suptitle("注释迁移（Task 3）：scAtlasVAE 源码默认日程 + reference-only frozen scVI kNN",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_phase5_transfer")


def transfer_protocol_p():
    """设计 P 的 zero-shot 分类头训练日程对照；fulltime 不是 full-shot。"""
    import pandas as pd
    paper_path = os.path.join(DATA_DIR, "phase5_transfer_results_patient_paper.csv")
    fulltime_path = os.path.join(DATA_DIR, "phase5_transfer_results_patient_fulltime.csv")
    for path in (paper_path, fulltime_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    rows = []
    for path, label in (
        (paper_path, "zero-shot (源码默认: 自动总轮数/末10轮)"),
        (fulltime_path, "zero-shot (fulltime: 150/150轮)"),
    ):
        frame = pd.read_csv(path)
        row = frame[frame["method"] == "scAtlasVAE (zero-shot)"].iloc[0].copy()
        row["method"] = label
        rows.append(row)
    df = pd.DataFrame(rows)
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "macro-F1"),
               ("macro_ovr_auc", "macro OVR-AUC")]
    colors = [PAL["encoder"]["ink"], PAL["accentB"]["ink"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(metrics)); w = 0.34
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [float(row[k]) for k, _ in metrics]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=colors[i],
                      label=row["method"], edgecolor="white", linewidth=0.9)
        ax.bar_label(bars, fmt="%.3f", fontsize=8.5, color=MUTED, padding=2)
    ax.set_xticks(x); ax.set_xticklabels([label for _, label in metrics], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("分数（越高越好）", fontsize=10)
    ax.set_title("设计 P：整位 patient RC.P20190923 的 zero-shot 训练日程敏感性",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=8.7, loc="upper left")
    fig.text(
        0.5, -0.01,
        "fulltime 同时改变总训练轮数与分类头启用轮数；这里只能视为联合日程对照，且不是 query/reference 共训的 full-shot。",
        ha="center", fontsize=8.3, color=MUTED,
    )
    _clean(ax)
    _save(fig, "fig_phase5_transfer_patient_protocol")


def invariance():
    """阶段 5 · E3：默认两模型≈0；仅显式 encode_covariates=True 的 scVI 明显漂移。"""
    import pandas as pd
    parts = []
    for fn in ("phase5_invariance_scatlasvae.csv", "phase5_invariance_scvi.csv"):
        p = os.path.join(DATA_DIR, fn)
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError("phase5_invariance_*.csv")
    df = pd.concat(parts, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    models = list(df["model"]); drift = [float(v) for v in df["mean_l2_drift_perm"]]
    # 按漂移大小上色：≈0（batch-invariant）用蓝、明显>0（batch-variant）用玫红。
    cols = [PAL["encoder"]["ink"] if d < 1e-3 else PAL["accentB"]["ink"] for d in drift]
    b = ax.bar(range(len(models)), drift, width=0.55, color=cols, edgecolor="white", linewidth=1)
    ax.bar_label(b, fmt="%.3f", fontsize=10, color=INK, padding=3)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9.5)
    ax.set_ylabel("全局置换 batch 后 z 的平均 L2 漂移", fontsize=10)
    ax.set_ylim(0, max(0.01, max(drift) * 1.35))
    ax.set_title("批不变编码器探针：同一批细胞、全局置换 batch，潜向量动不动？（真实）",
                 fontsize=12, fontweight="bold", pad=10)
    changed = float(df["batch_changed_fraction"].min()) if "batch_changed_fraction" in df else float("nan")
    n_batches = int(df["n_batches_probed"].min()) if "n_batches_probed" in df else 0
    sub = (f"覆盖 {n_batches} 位患者；保持 batch 边际计数并实际改变 {changed:.1%} 的探针标签。"
           "蓝=encoder 不显式接收 batch 元数据→本探针漂移≈0；玫红=显式接收 batch→漂移。"
           "这不保证表达矩阵 X 中的批次信号自动消失；统计混合仍看 patient-based 指标。")
    fig.text(0.5, -0.02, sub, ha="center", fontsize=8.2, color=MUTED)
    _clean(ax)
    _save(fig, "fig_phase5_invariance")


def ablation():
    """监督模型消融：左=潜维度 2/10/50，右=KL 预热开/关。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_DIR, "phase4_ablation_results.csv"), index_col=0)
    df = df[df.index != "Metric Type"]
    aggs = ["Batch correction", "Bio conservation", "Total"]
    zh = {"Batch correction": "批次校正", "Bio conservation": "生物保留", "Total": "总分"}
    panels = [
        ("潜维度 n_latent", [("X_nlat2", "n=2"), ("X_nlat10", "n=10(默认)"), ("X_nlat50", "n=50")]),
        ("KL 预热", [("X_nlat10", "有预热(0→1)"), ("X_nowarmup", "关(恒1.0)")]),
    ]
    mcol = [PAL["input"]["ink"], PAL["encoder"]["ink"], PAL["accentB"]["ink"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [3, 2]})
    for ax, (ptitle, items) in zip(axes, panels):
        items = [(k, lab) for k, lab in items if k in df.index]
        x = np.arange(len(aggs)); w = 0.8 / max(1, len(items))
        for i, (k, lab) in enumerate(items):
            vals = [float(df.loc[k, a]) for a in aggs]
            b = ax.bar(x + (i - (len(items) - 1) / 2) * w, vals, w, color=mcol[i % len(mcol)],
                       label=lab, edgecolor="white", linewidth=0.8)
            ax.bar_label(b, fmt="%.2f", fontsize=7.5, color=MUTED, padding=2)
        ax.set_xticks(x); ax.set_xticklabels([zh[a] for a in aggs], fontsize=10)
        ax.set_ylim(0, max(0.7, float(df[aggs].values.max()) * 1.18))
        ax.set_title(ptitle, fontsize=12, fontweight="bold", pad=8)
        ax.legend(fontsize=9, frameon=False, loc="upper left")
        _clean(ax)
    axes[0].set_ylabel("分数（越高越好）", fontsize=10)
    fig.suptitle("监督版 scAtlasVAE 消融：潜维度与 KL 预热（真实，scib-metrics）",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_phase4_ablation_bars")


def _categorical_palette(n_colors):
    """只用 Matplotlib 内置定性色板，返回不重复的类别颜色。"""
    from matplotlib.colors import to_rgba

    if n_colors <= 10:
        candidates = list(plt.get_cmap("tab10").colors)
    elif n_colors <= 20:
        candidates = list(plt.get_cmap("tab20").colors)
    else:
        # 交错三个 20 色定性表，避免相邻类别集中落在同一组近似色中。
        pools = [list(plt.get_cmap(name).colors) for name in ("tab20", "tab20b", "tab20c")]
        candidates = [pools[j][i] for i in range(20) for j in range(3)]

    colors = []
    for candidate in candidates:
        rgba = to_rgba(candidate)
        if rgba not in colors:
            colors.append(rgba)
        if len(colors) == n_colors:
            return colors
    # 当前项目最多 45 类；此分支仅为未来更多类别保底。
    for candidate in plt.get_cmap("hsv")(np.linspace(0, 1, n_colors, endpoint=False)):
        rgba = to_rgba(candidate)
        if rgba not in colors:
            colors.append(rgba)
        if len(colors) == n_colors:
            return colors
    raise ValueError(f"无法生成 {n_colors} 个不重复颜色")


def _scatter_by(ax, xy, labels, title, palette=None, s=2.3, legend=False,
                draw_order=None, alpha=0.55):
    """按类别上色后一次性散点；固定打乱绘制顺序，避免类别覆盖偏差。"""
    import pandas as pd
    from matplotlib.lines import Line2D

    cats = pd.Categorical(labels)
    uniq = list(cats.categories)
    colors = list(palette) if palette is not None else _categorical_palette(len(uniq))
    if len(colors) < len(uniq):
        raise ValueError(f"palette 只有 {len(colors)} 色，但需要 {len(uniq)} 色")
    colors = colors[:len(uniq)]
    point_colors = np.asarray([colors[code] for code in cats.codes])
    if draw_order is None:
        draw_order = np.random.default_rng(0).permutation(len(cats))
    draw_order = np.asarray(draw_order)
    ax.scatter(
        xy[draw_order, 0], xy[draw_order, 1], s=s,
        color=point_colors[draw_order], alpha=alpha,
        edgecolors="none", rasterized=True,
    )
    ax.set_title(title, fontsize=10.5, color=INK, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    _clean(ax)
    if legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=4,
                   markerfacecolor=colors[i], markeredgewidth=0, label=str(category))
            for i, category in enumerate(uniq)
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                  fontsize=6.5, frameon=False, ncol=1, handletextpad=0.2)


def umap_integration():
    """真实整合 UMAP：X_pca vs 监督版 X_scAtlasVAE_sup，按 patient/cell_type 上色。"""
    import h5py
    # 该图只需 obs/obsm；直接切片 HDF5，避免导入 Scanpy 或加载 10.5 万 × 4000 的表达矩阵。
    with h5py.File(os.path.join(DATA_DIR, "tcell_processed.h5ad"), "r") as h5:
        up = h5["obsm"]["X_umap_pca"][...]
        us = h5["obsm"]["X_umap_scatlasvae"][...]

        def read_categorical_obs(key):
            node = h5["obs"][key]
            categories = node["categories"].asstr()[...]
            return categories[node["codes"][...]]

        batch = read_categorical_obs("patient")    # 模型训练与 scIB 评测实际使用的 batch（45 位患者）
        ctype = read_categorical_obs("cell_type")  # 17 个 CD8 亚型
    draw_order = np.random.default_rng(0).permutation(len(batch))
    patient_palette = _categorical_palette(len(np.unique(batch)))
    celltype_palette = _categorical_palette(len(np.unique(ctype)))
    fig, ax = plt.subplots(2, 2, figsize=(11.2, 9.2))
    _scatter_by(ax[0, 0], up, batch, "未校正 X_pca · 按患者(batch)",
                palette=patient_palette, draw_order=draw_order)
    _scatter_by(ax[0, 1], up, ctype, "未校正 X_pca · 按细胞类型", legend=True,
                palette=celltype_palette, draw_order=draw_order)
    _scatter_by(ax[1, 0], us, batch, "监督版 scAtlasVAE · 按患者(batch)",
                palette=patient_palette, draw_order=draw_order)
    _scatter_by(ax[1, 1], us, ctype, "监督版 scAtlasVAE · 按细胞类型", legend=True,
                palette=celltype_palette, draw_order=draw_order)
    fig.suptitle("整合前(上) vs 整合后(下)：批次混合改善、细胞类型结构总体保留（真实结果）",
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.008,
        "患者面板使用 45 种不重复颜色，上下映射与全局随机绘制顺序一致；颜色仅作定性辅助，结论以 patient-based 指标为准。",
        ha="center", fontsize=8.0, color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    _save(fig, "fig_phase2_integration_umap")


def scalability():
    """可扩展性：时间、进程内存与 CUDA allocator 分口径展示。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_DIR, "phase5_scalability.csv")).sort_values("n_cells")
    required = {
        "load_setup_fit_seconds", "peak_process_rss_mb", "peak_process_private_mb",
        "peak_cuda_allocated_mb", "peak_cuda_reserved_mb",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(
            f"phase5_scalability.csv 仍是旧内存口径，缺少 {missing}；"
            "请先运行升级后的 phase5_scalability.py"
        )
    n = df["n_cells"].values / 1000.0   # 千细胞
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    # 左：时间
    axes[0].plot(n, df["fit_seconds"].values, "o-", color=PAL["encoder"]["ink"],
                 lw=2.2, ms=7, label="model.fit")
    axes[0].plot(n, df["load_setup_fit_seconds"].values, "s--", color=PAL["latent"]["ink"],
                 lw=1.8, ms=6, label="读取+初始化+fit")
    axes[0].set_xlabel("训练细胞数（千）", fontsize=10)
    axes[0].set_ylabel("固定 epoch 墙钟时间 (s)", fontsize=10)
    axes[0].set_title("时间（fresh worker）", fontsize=12, fontweight="bold", pad=8)
    axes[0].legend(frameon=False, fontsize=8)

    # 中：完整 Python 进程内存，覆盖 backed 读取、最小 AnnData、模型初始化和 fit。
    axes[1].plot(n, df["peak_process_rss_mb"].values, "o-", color=PAL["accentA"]["ink"],
                 lw=2.2, ms=7, label="RSS / working set")
    axes[1].plot(n, df["peak_process_private_mb"].values, "s--", color=PAL["accentB"]["ink"],
                 lw=1.8, ms=6, label="private bytes")
    axes[1].set_xlabel("训练细胞数（千）", fontsize=10)
    axes[1].set_ylabel("峰值进程内存 (MiB)", fontsize=10)
    axes[1].set_title("CPU/进程总口径", fontsize=12, fontweight="bold", pad=8)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=8)

    # 右：仅 PyTorch CUDA allocator。allocated 与 reserved 都不等于进程总显存。
    axes[2].plot(n, df["peak_cuda_allocated_mb"].values, "o-", color=PAL["encoder"]["ink"],
                 lw=2.2, ms=7, label="CUDA allocated")
    axes[2].plot(n, df["peak_cuda_reserved_mb"].values, "s--", color=PAL["latent"]["ink"],
                 lw=1.8, ms=6, label="CUDA reserved")
    axes[2].set_xlabel("训练细胞数（千）", fontsize=10)
    axes[2].set_ylabel("峰值 CUDA allocator (MiB)", fontsize=10)
    axes[2].set_title("GPU allocator 口径", fontsize=12, fontweight="bold", pad=8)
    axes[2].set_ylim(bottom=0)
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        _clean(ax)
        ax.set_xlim(0, n.max() * 1.08)
    fig.suptitle("scAtlasVAE 可扩展性：时间 / 进程内存 / CUDA 分口径（本机实测）",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_phase5_scalability")


def cross_atlas():
    """Task 2：两个分类头对所有细胞的预测标签共现矩阵。"""
    import pandas as pd
    matrix_path = os.path.join(DATA_DIR, "phase5_cross_atlas_head_alignment.csv")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(
            f"缺少论文式多分类头对齐结果：{matrix_path}。请先重新运行 "
            "phase5_cross_atlas.py；旧 phase5_cross_atlas_alignment.csv 是 "
            "latent-kNN 结果，不能用于这张多分类头预测共现图。"
        )
    M = pd.read_csv(matrix_path, index_col=0)
    fig, ax = plt.subplots(figsize=(max(8.5, 0.55 * M.shape[1]), 0.7 * M.shape[0] + 2.2))
    im = ax.imshow(M.values, aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=10)
    # 每格标占比（>0.15 才标，避免糊）
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if v > 0.15:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.6 else "black")
    ax.set_xlabel("主分类头预测的 Zheng 标签", fontsize=10)
    ax.set_ylabel("附加分类头预测的 Yost 标签", fontsize=10)
    ax.set_title("Task 2 跨图谱标签对齐：两分类头对所有细胞的预测共现（行归一化）\n"
                 "每行表示 P（Zheng 头预测标签 | Yost 头预测标签）；图内仅标注占比 > 0.15 的格子",
                 fontsize=10.5, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="预测共现占比")
    _save(fig, "fig_phase5_cross_atlas")


def umap_compare():
    """官方监督版 vs 带分类头的手写 VAE UMAP 对照（按细胞类型上色）。

    UMAP 的朝向本身任意。为便于并排阅读，这里只把手写面板的 UMAP1 取反；
    这不是两个二维嵌入的定量配准，也不代表两张图逐点等价。该图只用于观察谱系富集和
    亚型连续重叠；复现是否成立仍以高维 latent benchmark、近邻纯度和类中心距离关系为准。
    """
    # 只读取这张图需要的缓存 UMAP 与 obs，避免为绘图加载约 856 MB 的完整表达矩阵。
    # 数组与此前 scanpy.read_h5ad 路径完全相同，因此不会重算或改变 UMAP 布局。
    import h5py
    with h5py.File(os.path.join(DATA_DIR, "tcell_processed.h5ad"), "r") as h5:
        cell_type_node = h5["obs"]["cell_type"]
        categories = cell_type_node["categories"].asstr()[...]
        ctype = categories[cell_type_node["codes"][...]]
        u_off = h5["obsm"]["X_umap_official"][...]
        u_mine = h5["obsm"]["X_umap_mine"][...].copy()
    u_mine[:, 0] = -u_mine[:, 0]          # 水平镜像，抵消两套独立 UMAP 的任意左右朝向
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 5.0))
    _scatter_by(ax[0], u_off, ctype, "官方监督版 scAtlasVAE latent")
    _scatter_by(ax[1], u_mine, ctype, "手写最小 VAE latent（UMAP1 已水平镜像对齐）", legend=True)
    fig.suptitle("监督机制对照：官方 vs 手写 VAE；细粒度亚型连续重叠（UMAP 仅作定性）",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 0.985, 0.96])
    _save(fig, "fig_phase3_umap_compare")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["all"]
    requested_all = "all" in targets
    if requested_all:
        targets = ["loss", "bench", "ablation", "umap_integration", "umap_compare",
                   "bench_minimal", "transfer", "transfer_protocol_p", "invariance",
                   "scalability", "cross_atlas"]
    fns = {"loss": loss_curve, "bench": bench, "ablation": ablation,
            "umap_integration": umap_integration, "umap_compare": umap_compare,
            "bench_minimal": bench_minimal, "transfer": transfer,
            "transfer_protocol_p": transfer_protocol_p, "invariance": invariance,
           "scalability": scalability, "cross_atlas": cross_atlas}
    failures = []
    for t in targets:
        if t in fns:
            try:
                fns[t]()
            except FileNotFoundError as e:
                failures.append(t)
                print(f"[error] {t}: 缺数据 {e}", file=sys.stderr)
        else:
            failures.append(t)
            print(f"[error] 未知绘图目标：{t}", file=sys.stderr)
    if failures:
        raise SystemExit(f"绘图未完整完成：{', '.join(failures)}")
    if requested_all:
        _write_figure_manifest(targets)
