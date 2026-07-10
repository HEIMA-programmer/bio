"""阶段二 · 步骤 5（baseline）：Harmony 批次校正基线（用 harmonypy，别自己手写）。

为什么用 Harmony 而不是 scVI
    原计划的 baseline 是 scVI（"编码器 batch-variant"的经典 VAE，作为 scAtlasVAE
    "编码器 batch-invariant"的对照）。但 scvi-tools 在**本机 Windows** 上装不上——
    它依赖 JAX 生态的 orbax-checkpoint，其包内有一个超长路径的测试文件，触发 Windows
    260 字符路径上限（需管理员开 LongPathsEnabled 才能装）。为不改系统设置、又能给出
    一个**真实可跑的批次校正基线**，改用 **Harmony**（Korsunsky et al. 2019）：
    它是单细胞整合最经典的基线之一（论文 benchmark 也含它），纯 Python、在 PCA 嵌入上做
    迭代式批次校正，轻量稳定。scVI 那个"编码器是否看 batch"的**架构对照**仍在文档里保留
    （见 01 文档 §1.5 / phase2 §5）——那是概念对比，不依赖实跑。

产出
    obsm['X_harmony']：在未校正的 obsm['X_pca'] 上做 Harmony 得到的批次校正嵌入。

用法（在环境 B `scib` 中，需 harmonypy==0.0.9）
    python phase2_baseline_harmony.py

对应报告
    reports/phase2_integration_and_benchmark.md 步骤 5。
"""
import scanpy as sc

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"

adata = sc.read_h5ad(PROC_PATH)

# 在未校正的 X_pca 上做 Harmony；结果写进 obsm['X_pca_harmony']
sc.external.pp.harmony_integrate(adata, key=BATCH_KEY, basis="X_pca",
                                 adjusted_basis="X_pca_harmony", max_iter_harmony=20)
adata.obsm["X_harmony"] = adata.obsm["X_pca_harmony"]

adata.write_h5ad(PROC_PATH)
print("Harmony 完成，已写入 obsm['X_harmony']（在 X_pca 上做批次校正）")
