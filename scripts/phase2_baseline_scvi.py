"""阶段二 · 步骤 5：scVI baseline（用官方 scvi-tools，别自己手写）。

为什么要它
    scVI 是"编码器 batch-variant"的经典 VAE（编码器吃 batch），
    正好作为 scAtlasVAE（编码器 batch-invariant）的对照（见 01 文档 §1.5）。

用法（需装了 scvi-tools 的环境，环境 A 或 B 均可）
    python phase2_baseline_scvi.py

对应报告
    reports/phase2_integration_and_benchmark.md 步骤 5。
"""
import scanpy as sc
import scvi

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "study_name"

adata = sc.read_h5ad(PROC_PATH)

# scVI 也要原始整数计数：用预处理时备份的 layers['counts']
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=BATCH_KEY)
model = scvi.model.SCVI(adata)     # 默认参数，与论文 baseline 一致

# 论文对 scVI 固定 max_epochs=10；这里沿用以对齐设置
model.train(max_epochs=10)

adata.obsm["X_scVI"] = model.get_latent_representation()
adata.write_h5ad(PROC_PATH)
print("scVI 训练完成，已写入 obsm['X_scVI']")
