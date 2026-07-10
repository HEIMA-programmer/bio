"""阶段二 · 步骤 5：scVI baseline（用官方 scvi-tools，别自己手写）。

为什么要它
    scVI 是"编码器 batch-variant"的经典 VAE（编码器吃 batch），
    正好作为 scAtlasVAE（编码器 batch-invariant）的对照（见 01 文档 §1.5）。

⚠️ 本机（Windows）跑不了 —— 已改用 Harmony 基线
    scvi-tools 依赖 JAX 生态的 orbax-checkpoint，其包内有超长路径的测试文件，
    触发 Windows 260 字符路径上限（需管理员开 LongPathsEnabled 才能装）。为不改系统设置，
    本次改用 phase2_baseline_harmony.py（Harmony，同样是经典批次校正基线）作为可跑的对照。
    本脚本保留给能装 scvi-tools 的环境（Linux / 已开长路径的 Windows）参考。
    scVI 与 scAtlasVAE 的"编码器是否看 batch"的**架构对照**仍在文档保留（概念对比，不依赖实跑）。

用法（需装了 scvi-tools 的环境；本机不可用，见上）
    python phase2_baseline_scvi.py

对应报告
    reports/phase2_integration_and_benchmark.md 步骤 5。
"""
import scanpy as sc
import scvi

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"

adata = sc.read_h5ad(PROC_PATH)

# scVI 也要原始整数计数：用预处理时备份的 layers['counts']
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=BATCH_KEY)
model = scvi.model.SCVI(adata)     # 默认参数，与论文 baseline 一致

# 论文对 scVI 固定 max_epochs=10；这里沿用以对齐设置
model.train(max_epochs=10)

adata.obsm["X_scVI"] = model.get_latent_representation()
adata.write_h5ad(PROC_PATH)
print("scVI 训练完成，已写入 obsm['X_scVI']")
