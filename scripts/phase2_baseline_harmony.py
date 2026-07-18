"""阶段二 · 步骤 5（baseline）：Harmony 批次校正基线（用 harmonypy，别自己手写）。

为什么仍保留 Harmony
    项目早期因 Windows 长路径限制暂时装不上 scvi-tools，曾用 Harmony 作为替代；现在
    scVI 已在独立 CPU 环境成功实跑，并是主 baseline。Harmony 仅作为可选的第二基线。
    **Harmony**（Korsunsky et al. 2019）是单细胞整合最经典的基线之一（论文 benchmark 也含它），
    纯 Python、在 PCA 嵌入上做
    迭代式批次校正，轻量稳定。注意 scvi-tools 默认 ``encode_covariates=False``，本项目
    实跑的默认 scVI encoder 也不显式接收 batch；不能把它写成 scAtlasVAE 的
    “batch-variant encoder”反面。可配置版本的差异见 01 文档 §1.5 / phase5 E3。

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
