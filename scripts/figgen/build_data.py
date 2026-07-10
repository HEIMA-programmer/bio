"""生成"数据图/定量图"（预期·示意）到 reports/*.svg，并输出 PNG 供目检。

这些图用于占位——数值是按"复现顺利、符合论文趋势"填的示意值；
你在 4060 实跑后用真实数据替换。所有图都在角落标注「示意 / 预期」。

运行：python3 build_data.py [fig_name ...]
"""
import sys
import math
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

CAT = [PAL["encoder"]["ink"], PAL["decoder"]["ink"], PAL["cls"]["ink"],
       PAL["latent"]["ink"], PAL["accentA"]["ink"], PAL["accentB"]["ink"]]
TYPE_NAMES = ["Tn", "Tcm", "GZMK⁺ Tem", "ITGAE⁺ Tex", "XBP1⁺ Tex", "MAIT"]


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _stamp(fig, text="示意 / 预期占位 · 待 4060 实跑替换"):
    fig.text(0.995, 0.012, text, ha="right", va="bottom",
             fontsize=8.5, color=FAINT, style="italic")


def save(fig, name):
    import os
    svg = f"{T.REPORTS_DIR}/{name}.svg"
    png = f"{T.PNG_DIR}/{name}.png"
    fig.savefig(svg, format="svg", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(png, format="png", dpi=145, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


# ======================================================================
def fig_phase2_integration_umap():
    rng = np.random.default_rng(3)
    n_type, n_batch, per = 6, 3, 60
    type_centers = np.array([[0, 6], [5, 5.5], [7.5, 1], [4.5, -3.5], [-1, -3], [-3.5, 2.2]]) * 1.5
    batch_shift = np.array([[-9, 8], [10, 7], [1, -11]]) * 1.0

    def make(mix):  # mix=0 无校正(批次主导)，mix=1 已校正
        xs, cty, cba = [], [], []
        for t in range(n_type):
            for b in range(n_batch):
                c = type_centers[t] + (1 - mix) * batch_shift[b]
                p = rng.normal(c, 0.9, size=(per, 2))
                xs.append(p); cty += [t] * per; cba += [b] * per
        return np.vstack(xs), np.array(cty), np.array(cba)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7))
    before, ty0, ba0 = make(0.0)
    after, ty1, ba1 = make(1.0)
    bcols = [PAL["input"]["ink"], PAL["accentA"]["ink"], PAL["cls"]["ink"]]

    for ax, data, ba, title in [
        (axes[0], before, ba0, "整合前（按批次上色）：三个批次各自成团"),
        (axes[1], after, ba1, "整合后（按批次上色）：批次在类型簇内充分混合"),
    ]:
        for b in range(n_batch):
            m = ba == b
            ax.scatter(data[m, 0], data[m, 1], s=9, color=bcols[b], alpha=0.75,
                       edgecolors="none", label=f"批次 {b+1}")
        ax.set_title(title, fontsize=11.5, color=INK, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP1", fontsize=9.5); ax.set_ylabel("UMAP2", fontsize=9.5)
        _clean(ax)
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False, ncol=1)
    fig.suptitle("scAtlasVAE 整合效果（示意）", fontsize=14, fontweight="bold", y=1.02)
    _stamp(fig)
    save(fig, "fig_phase2_integration_umap")


# ======================================================================
def fig_phase2_benchmark_bars():
    methods = ["X_pca\n(未校正)", "scVI", "scAtlasVAE"]
    metrics = ["批次校正", "生物保留", "总分 Overall"]
    vals = np.array([[0.55, 0.80, 0.68], [0.88, 0.78, 0.83], [0.89, 0.82, 0.86]])
    mcol = [PAL["input"]["ink"], PAL["accentA"]["ink"], PAL["encoder"]["ink"]]

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    x = np.arange(len(metrics)); w = 0.25
    for i, m in enumerate(methods):
        b = ax.bar(x + (i - 1) * w, vals[i], w, color=mcol[i], label=m, edgecolor="white", linewidth=0.8)
        ax.bar_label(b, fmt="%.2f", fontsize=8.5, color=MUTED, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10.5)
    ax.set_ylim(0, 1.0); ax.set_ylabel("分数（越高越好）", fontsize=10)
    ax.set_title("整合评测：三种嵌入对比（示意，scib-metrics）", fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    _clean(ax)
    fig.text(0.5, -0.02, "注：scib-metrics 与论文旧 scib 数值不可直接比，只看方法间相对排序。",
             ha="center", fontsize=8.8, color=MUTED)
    _stamp(fig)
    save(fig, "fig_phase2_benchmark_bars")


# ======================================================================
def fig_phase2_loss_curve():
    ep = np.arange(1, 74)
    recon = 1400 * np.exp(-ep / 18) + 340 + rng_noise(ep, 6)
    total = recon + 40 * np.minimum(1, ep / 400) * (ep / 73)
    klw = np.minimum(1, ep / 400)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(ep, total, color=PAL["encoder"]["ink"], lw=2.2, label="总损失")
    ax.plot(ep, recon, color=PAL["decoder"]["ink"], lw=1.8, ls="--", label="重构损失")
    ax.set_xlabel("epoch", fontsize=10); ax.set_ylabel("损失（每细胞）", fontsize=10)
    ax.set_title("训练损失曲线（示意，约 73 epoch）", fontsize=13, fontweight="bold", pad=10)
    ax2 = ax.twinx()
    ax2.plot(ep, klw, color=PAL["loss"]["ink"], lw=1.8, label="KL 权重(预热)")
    ax2.set_ylim(0, 1.02); ax2.set_ylabel("KL 权重 λ_KL", color=PAL["loss"]["ink"], fontsize=10)
    ax2.tick_params(axis="y", colors=PAL["loss"]["ink"]); ax2.grid(False)
    ax2.axhline(0.18, color=PAL["loss"]["ink"], ls=":", lw=1.1, alpha=0.7)
    ax2.annotate("默认 warmup=400，到 73 epoch 时 λ_KL≈0.18（从没到 1）",
                 xy=(73, 0.18), xytext=(20, 0.5), fontsize=8.6, color=PAL["loss"]["ink"],
                 arrowprops=dict(arrowstyle="->", color=PAL["loss"]["ink"], lw=1))
    _clean(ax); ax2.spines["top"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="center right")
    _stamp(fig)
    save(fig, "fig_phase2_loss_curve")


def rng_noise(ep, s):
    r = np.random.default_rng(1)
    return r.normal(0, s, size=len(ep))


# ======================================================================
def fig_phase3_umap_compare():
    rng = np.random.default_rng(11)
    centers = np.array([[0, 6], [5, 5], [7, 0.5], [4, -4], [-1, -3], [-3.5, 2]]) * 1.4
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7))
    for ax, seed, title in [(axes[0], 0, "官方 scAtlasVAE 潜空间"),
                            (axes[1], 5, "你的手写最小版潜空间")]:
        rr = np.random.default_rng(seed + 20)
        rot = 0.0 if seed == 0 else 0.28  # 手写版整体略有旋转/形变
        R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        for t, c in enumerate(centers):
            p = rr.normal(c, 0.95, size=(55, 2)) @ R.T
            ax.scatter(p[:, 0], p[:, 1], s=10, color=CAT[t], alpha=0.8,
                       edgecolors="none", label=TYPE_NAMES[t])
        ax.set_title(title, fontsize=11.5, color=INK, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP1", fontsize=9.5); ax.set_ylabel("UMAP2", fontsize=9.5)
        _clean(ax)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, frameon=False)
    fig.suptitle("官方 vs 手写：结构相似即成功（kNN 邻域 Jaccard ≈ 0.4–0.6）",
                 fontsize=13, fontweight="bold", y=1.02)
    _stamp(fig)
    save(fig, "fig_phase3_umap_compare")


# ======================================================================
def fig_phase4_ablation_bars():
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
    # 左：潜维度
    labs = ["n_latent=2", "n_latent=10\n(默认)", "n_latent=50"]
    vals = [0.72, 0.86, 0.85]
    cols = [PAL["input"]["ink"], PAL["encoder"]["ink"], PAL["accentA"]["ink"]]
    b = axes[0].bar(labs, vals, color=cols, edgecolor="white", width=0.62)
    axes[0].bar_label(b, fmt="%.2f", fontsize=9, color=MUTED, padding=2)
    axes[0].set_ylim(0, 1.0); axes[0].set_ylabel("总分 Overall", fontsize=10)
    axes[0].set_title("潜维度消融", fontsize=12, fontweight="bold", pad=8)
    _clean(axes[0])
    # 右：KL 预热
    labs2 = ["预热 开\n(默认)", "预热 关"]
    vals2 = [0.86, 0.70]
    cols2 = [PAL["decoder"]["ink"], PAL["accentB"]["ink"]]
    b2 = axes[1].bar(labs2, vals2, color=cols2, edgecolor="white", width=0.5)
    axes[1].bar_label(b2, fmt="%.2f", fontsize=9, color=MUTED, padding=2)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("KL 预热消融", fontsize=12, fontweight="bold", pad=8)
    _clean(axes[1])
    fig.suptitle("消融结果（示意）：作者的设计选择是否必要", fontsize=13.5, fontweight="bold", y=1.03)
    _stamp(fig)
    save(fig, "fig_phase4_ablation_bars")


# ======================================================================
def _poisson_pmf(k, lam):
    return np.exp(-lam) * lam ** k / np.array([math.factorial(int(i)) for i in k])


def _nb_pmf(k, mu, theta):
    # NB: r=theta, p=theta/(theta+mu)
    p = theta / (theta + mu)
    coef = np.array([math.lgamma(int(i) + theta) - math.lgamma(theta) - math.lgamma(int(i) + 1) for i in k])
    return np.exp(coef + theta * math.log(p) + k * math.log(1 - p))


def fig_zinb_construction():
    k = np.arange(0, 16)
    lam, theta, pi = 3.0, 1.4, 0.35
    pois = _poisson_pmf(k, lam)
    nb = _nb_pmf(k, lam, theta)
    zinb = (1 - pi) * nb
    zinb[0] += pi
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharey=True)
    data = [
        (pois, "① 泊松 Poisson", "方差 = 均值（太死板）", PAL["encoder"]["ink"]),
        (nb, "② 负二项 NB", "加“离散度 θ”→ 方差 > 均值", PAL["latent"]["ink"]),
        (zinb, "③ 零膨胀 ZINB", "再并联“补零开关 π”治 dropout", PAL["decoder"]["ink"]),
    ]
    for ax, (p, t, sub, c) in zip(axes, data):
        bars = ax.bar(k, p, color=c, alpha=0.85, edgecolor="white", width=0.82)
        if t.startswith("③"):
            bars[0].set_color(PAL["cls"]["ink"])  # 高亮 0 处的尖峰
            ax.annotate("额外补的零", xy=(0, p[0]), xytext=(3.2, p[0] * 0.92),
                        fontsize=8.8, color=PAL["cls"]["ink"],
                        arrowprops=dict(arrowstyle="->", color=PAL["cls"]["ink"], lw=1))
        ax.set_title(t, fontsize=12, fontweight="bold", color=c, pad=6)
        ax.set_xlabel(sub, fontsize=9.5, color=MUTED)
        _clean(ax)
    axes[0].set_ylabel("概率 P(计数=k)", fontsize=10)
    fig.suptitle("解码器为什么用 ZINB：从泊松三步搭起来", fontsize=13.5, fontweight="bold", y=1.04)
    save(fig, "fig_zinb_construction")


# ======================================================================
def fig_loss_and_warmup():
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), gridspec_kw={"width_ratios": [1, 1.15]})
    # 左：几块损失的相对量级（示意；含读源码才看到的"门控稀疏"项）
    comps = ["ZINB 重构", "λ_KL·KL", "λ_ct·交叉熵", "门控稀疏(sigmoid π)"]
    mag = [1.0, 0.16, 0.22, 0.10]
    cols = [PAL["decoder"]["ink"], PAL["latent"]["ink"], PAL["cls"]["ink"], PAL["batch"]["ink"]]
    axes[0].barh(comps[::-1], mag[::-1], color=cols[::-1], edgecolor="white", height=0.62)
    axes[0].set_xlim(0, 1.15)
    axes[0].set_title("总损失 = 几块相加（相对量级·示意）", fontsize=12, fontweight="bold", pad=8)
    axes[0].set_xlabel("相对贡献", fontsize=9.5)
    _clean(axes[0]); axes[0].grid(axis="y", visible=False)
    # 右：warmup 的**正确**故事——因 min(max_epoch,400) 截断，λ_KL 在 max_epoch 内 0→1 爬满
    for me, c, lab in [(200, PAL["loss"]["ink"], "4 万细胞 max_epoch=200"),
                       (73, PAL["accentB"]["ink"], "11 万细胞 max_epoch≈73")]:
        ep = np.arange(0, me + 1)
        axes[1].plot(ep, np.minimum(1, ep / min(me, 400)), color=c, lw=2.2,
                     label=f"λ_KL=min(1, epoch/min(max_epoch,400))·{lab}")
        axes[1].scatter([me], [1.0], color=c, zorder=5, s=36)
    axes[1].annotate("关键：源码有 n_epochs_kl_warmup=min(max_epoch,400)\n"
                     "→ 预热长度=max_epoch，λ_KL 整个训练 0→1 爬满、末轮≈1\n"
                     "（旧图误作\"只到0.18\"，是漏读这行 min）",
                     xy=(200, 1.0), xytext=(18, 0.42), fontsize=8.4, color=INK,
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    axes[1].set_xlim(0, 260); axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("epoch", fontsize=10); axes[1].set_ylabel("KL 权重 λ_KL", fontsize=10)
    axes[1].set_title("KL 预热真相（读全 fit 源码才看到）", fontsize=12, fontweight="bold", pad=8)
    axes[1].legend(fontsize=7.6, frameon=False, loc="lower right")
    _clean(axes[1])
    _stamp(fig, "左：损失量级示意 · 右：预热为真实调度公式")
    save(fig, "fig_loss_and_warmup")


REGISTRY = {
    "fig_phase2_integration_umap": fig_phase2_integration_umap,
    "fig_phase2_benchmark_bars": fig_phase2_benchmark_bars,
    "fig_phase2_loss_curve": fig_phase2_loss_curve,
    "fig_phase3_umap_compare": fig_phase3_umap_compare,
    "fig_phase4_ablation_bars": fig_phase4_ablation_bars,
    "fig_zinb_construction": fig_zinb_construction,
    "fig_loss_and_warmup": fig_loss_and_warmup,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(REGISTRY)
    for n in names:
        REGISTRY[n]()
        print("built", n)
