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
    """真实训练动态：总/重构损失下降 + λ_KL 预热 0→~1（**纠正旧稿"只到0.18"**）。"""
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
    ax.set_title(f"scAtlasVAE 训练损失曲线（实跑，{len(ep)} epoch）",
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
    # 阶段 5 · E2：四方对比（复现论文 Ext. Data Fig. 2a 的 无监督/监督 两根柱）。
    _bars_from_csv(
        "phase2_benchmark_results.csv",
        ["X_pca", "X_scVI", "X_scAtlasVAE_unsup", "X_scAtlasVAE_sup"],
        "整合评测：PCA / scVI / scAtlasVAE(无监督) / scAtlasVAE(监督)（真实，scib-metrics）",
        "fig_phase2_benchmark_bars",
        note="注：scib-metrics 与论文旧 scib 数值不可直接比，只看相对排序。无监督 scAtlasVAE≈scVI、监督版更高——复现论文 Ext. Data Fig. 2a。")


def bench_minimal():
    # 阶段 5 · E4：把手写最小 VAE 放上同一把标尺。
    _bars_from_csv(
        "phase5_minimal_bench.csv",
        ["X_pca", "X_scVI", "X_scAtlasVAE_sup", "X_minimal"],
        "手写最小 VAE 上标尺：与 PCA / scVI / 官方监督版并列（真实，scib-metrics）",
        "fig_phase5_minimal_bench",
        note="注：X_minimal = 从零手写的最小 scAtlasVAE。看它相对 PCA 与 scVI/官方落在哪，量化手写实现的水平。")


def transfer():
    """阶段 5 · E1：注释迁移的 acc/macroF1/AUROC 分组条形（按设计×方法）。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_DIR, "phase5_transfer_results.csv"))
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "macro-F1"), ("macro_ovr_auc", "macro OVR-AUC")]
    designs = {"A": "设计A(随机5%)", "B": "设计B(整癌种UCEC)"}
    mcol = {"scAtlasVAE (zero-shot)": PAL["encoder"]["ink"],
            "scAtlasVAE (full-shot)": PAL["latent"]["ink"],
            "kNN on scVI latent": PAL["accentA"]["ink"]}
    dlist = [d for d in ["A", "B"] if d in set(df["design"])]
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
        ax.legend(fontsize=8.3, frameon=False, loc="lower left")
        _clean(ax)
    axes[0][0].set_ylabel("分数（越高越好）", fontsize=10)
    fig.suptitle("注释迁移（Task 3）：zero-shot / full-shot / kNN 对照（真实）",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_phase5_transfer")


def invariance():
    """阶段 5 · E3：批不变探针——打乱 batch 后潜向量漂移，scAtlasVAE≈0 vs scVI>0。"""
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
    ax.set_ylabel("打乱 batch 后 z 的平均 L2 漂移", fontsize=10)
    ax.set_ylim(0, max(0.05, max(drift) * 1.25))
    ax.set_title("批不变编码器探针：同一批细胞、打乱 batch，潜向量动不动？（真实）",
                 fontsize=12, fontweight="bold", pad=10)
    sub = ("蓝=编码器不吃 batch→漂移≈0；玫红=编码器吃 batch→明显漂移。scAtlasVAE **结构上永不**编码 batch；"
           "scVI 默认也不编码(细节)，仅 encode_covariates=True 时才 batch-variant。")
    fig.text(0.5, -0.02, sub, ha="center", fontsize=8.2, color=MUTED)
    _clean(ax)
    _save(fig, "fig_phase5_invariance")


def ablation():
    """消融两联图：左=潜维度 2/10/50，右=KL 预热 开/关；各画 批次校正/生物保留/总分。"""
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
    fig.suptitle("消融：潜维度 与 KL 预热（真实，scib-metrics）", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_phase4_ablation_bars")


def _scatter_by(ax, xy, labels, title, palette=None, s=3, legend=False, max_leg=8):
    """按类别上色的散点。labels 为分类数组。"""
    import pandas as pd
    cats = pd.Categorical(labels)
    uniq = list(cats.categories)
    cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
    for i, c in enumerate(uniq):
        m = cats.codes == i
        col = (palette[i % len(palette)] if palette else cmap(i % cmap.N))
        ax.scatter(xy[m, 0], xy[m, 1], s=s, color=col, alpha=0.6,
                   edgecolors="none", label=str(c), rasterized=True)
    ax.set_title(title, fontsize=10.5, color=INK, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    _clean(ax)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.5,
                  frameon=False, ncol=1, markerscale=2, handletextpad=0.2)


def umap_integration():
    """真实整合 UMAP：X_pca(未校正) vs X_scAtlasVAE(整合后)，按 batch(癌种) 与 cell_type 上色。"""
    import scanpy as sc
    a = sc.read_h5ad(os.path.join(DATA_DIR, "tcell_processed.h5ad"))
    up = a.obsm["X_umap_pca"]; us = a.obsm["X_umap_scatlasvae"]
    batch = a.obs["cancerType"].values     # 8 类，可读；作 batch/技术轴的可视化
    ctype = a.obs["cell_type"].values      # 17 个 CD8 亚型
    fig, ax = plt.subplots(2, 2, figsize=(11.2, 9.2))
    _scatter_by(ax[0, 0], up, batch, "未校正 X_pca · 按癌种(batch)")
    _scatter_by(ax[0, 1], up, ctype, "未校正 X_pca · 按细胞类型", legend=True)
    _scatter_by(ax[1, 0], us, batch, "scAtlasVAE · 按癌种(batch)")
    _scatter_by(ax[1, 1], us, ctype, "scAtlasVAE · 按细胞类型", legend=True)
    fig.suptitle("整合前(上) vs 整合后(下)：批次混合↑、细胞类型仍分得开（真实结果）",
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig_phase2_integration_umap")


def scalability():
    """可扩展性：训练时间 / 峰值显存 随细胞数增长（对标论文 Ext. Data Fig. 4e,f）。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_DIR, "phase5_scalability.csv")).sort_values("n_cells")
    n = df["n_cells"].values / 1000.0   # 千细胞
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    # 左：时间
    axes[0].plot(n, df["fit_seconds"].values, "o-", color=PAL["encoder"]["ink"], lw=2.2, ms=7)
    axes[0].set_xlabel("训练细胞数（千）", fontsize=10)
    axes[0].set_ylabel("固定 epoch 训练墙钟时间 (s)", fontsize=10)
    axes[0].set_title("训练时间 vs 细胞数", fontsize=12, fontweight="bold", pad=8)
    # 右：显存。**y 轴从 0 起**，避免把 ~110MB 的 0.1MB 抖动放大成"断崖"假象——
    # 峰值显存实测几乎恒定（分批训练，GPU 上只驻留一个 minibatch），要如实画成一条平线。
    axes[1].plot(n, df["peak_gpu_mb"].values, "s-", color=PAL["accentB"]["ink"], lw=2.2, ms=7)
    axes[1].set_xlabel("训练细胞数（千）", fontsize=10)
    axes[1].set_ylabel("峰值显存 (MB)", fontsize=10)
    axes[1].set_title("峰值显存 vs 细胞数", fontsize=12, fontweight="bold", pad=8)
    axes[1].set_ylim(0, float(df["peak_gpu_mb"].max()) * 1.5)
    axes[1].annotate("几乎恒定 ~110 MB（分批训练，GPU 只驻留 1 个 minibatch）",
                     xy=(n.mean(), float(df["peak_gpu_mb"].max())),
                     xytext=(n.min(), float(df["peak_gpu_mb"].max()) * 1.18),
                     fontsize=8.4, color=MUTED)
    for ax in axes:
        _clean(ax)
        ax.set_xlim(0, n.max() * 1.08)
    fig.suptitle("scAtlasVAE 可扩展性（本机 4060 实测，对标 Ext. Data Fig. 4e,f）",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_phase5_scalability")


def cross_atlas():
    """Task 2：跨图谱标签对齐矩阵热图（Yost CD8 亚型 × Zheng 亚型，行归一化占比）。"""
    import pandas as pd
    M = pd.read_csv(os.path.join(DATA_DIR, "phase5_cross_atlas_alignment.csv"), index_col=0)
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
    ax.set_xlabel("Zheng 亚型（我们的图谱，meta.cluster）", fontsize=10)
    ax.set_ylabel("Yost CD8 亚型", fontsize=10)
    ax.set_title("Task 2 跨图谱标签对齐：每个 Yost 亚型的最近邻 Zheng 亚型分布（行归一化，真实）\n"
                 "耗竭态 CD8_ex/CD8_ex_act → Tex，记忆 CD8_mem → Tem/Tm——生物学对上了",
                 fontsize=10.5, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="占比")
    _save(fig, "fig_phase5_cross_atlas")


def umap_compare():
    """官方 vs 手写 VAE 的 UMAP 对照（按细胞类型上色）。

    注意 UMAP 朝向是任意的：两套 latent 各自独立跑 UMAP，实测两图"各亚型横坐标"
    相关 r≈-0.895——即整张图互为**左右镜像**（红 Temra 官方在最左、手写在最右）。
    镜像/旋转无科学含义，为便于并排对照，这里把手写面板的 UMAP1 取反（水平镜像回来），
    让同一亚型落到同侧；判定"趋势一致"看的是拓扑邻接，不是绝对左右。
    """
    import scanpy as sc
    a = sc.read_h5ad(os.path.join(DATA_DIR, "tcell_processed.h5ad"))
    ctype = a.obs["cell_type"].values
    u_off = np.asarray(a.obsm["X_umap_official"])
    u_mine = np.asarray(a.obsm["X_umap_mine"]).copy()
    u_mine[:, 0] = -u_mine[:, 0]          # 水平镜像，抵消两套独立 UMAP 的任意左右朝向
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 5.0))
    _scatter_by(ax[0], u_off, ctype, "官方 scAtlasVAE latent")
    _scatter_by(ax[1], u_mine, ctype, "手写最小 VAE latent（UMAP1 已水平镜像对齐）", legend=True)
    fig.suptitle("官方 vs 手写 VAE：UMAP 按细胞类型上色（趋势一致即成功）",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig_phase3_umap_compare")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["all"]
    if "all" in targets:
        targets = ["loss", "bench", "ablation", "umap_integration", "umap_compare",
                   "bench_minimal", "transfer", "invariance", "scalability", "cross_atlas"]
    fns = {"loss": loss_curve, "bench": bench, "ablation": ablation,
           "umap_integration": umap_integration, "umap_compare": umap_compare,
           "bench_minimal": bench_minimal, "transfer": transfer, "invariance": invariance,
           "scalability": scalability, "cross_atlas": cross_atlas}
    for t in targets:
        if t in fns:
            try:
                fns[t]()
            except FileNotFoundError as e:
                print(f"[skip] {t}: 缺数据 {e}")
        else:
            print(f"[todo] {t}: 该图函数将在拿到对应实跑产物后补上")
