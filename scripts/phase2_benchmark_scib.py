"""阶段二 · 步骤 7：用 scib-metrics 定量对比三种嵌入。

对比对象
    obsm['X_pca']（未校正基线） / obsm['X_scVI'] / obsm['X_scAtlasVAE']
指标
    Benchmarker 会分别算"批次校正"和"生物保留"两类指标并汇总排名。
提醒
    scib-metrics 的数值与论文旧 scib(1.1.4) 不可直接比——只看方法间相对排序。

用法（在环境 B `scib`，py3.10）
    python phase2_benchmark_scib.py

对应报告
    reports/phase2_integration_and_benchmark.md 步骤 7 与第 7 节。
"""
import scanpy as sc
from scib_metrics.benchmark import Benchmarker

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"
LABEL_KEY = "cell_type"
# 三个对照：X_pca(未校正) / X_scVI(经典 batch-variant VAE) / X_scAtlasVAE(本方法)。
# scvi-tools 在本机 Windows 开启长路径(LongPathsEnabled)后可正常安装，故 baseline 用回 scVI
# （scAtlasVAE 编码器 batch-invariant 的正牌对照）。Harmony 结果仍可选（见 phase2_baseline_harmony.py）。
EMBEDDINGS = ["X_pca", "X_scVI", "X_scAtlasVAE"]

adata = sc.read_h5ad(PROC_PATH)

bm = Benchmarker(
    adata,
    batch_key=BATCH_KEY,
    label_key=LABEL_KEY,
    embedding_obsm_keys=EMBEDDINGS,
    n_jobs=-1,
)
bm.benchmark()

# 表格：每种嵌入的各项指标 + 两类汇总 + 总分
results = bm.get_results(min_max_scale=False)
print(results)
results.to_csv("phase2_benchmark_results.csv")

# 排名图（Batch correction / Bio conservation / Total 三栏）
bm.plot_results_table(min_max_scale=False, save_dir=".")
print("完成：见 phase2_benchmark_results.csv 与排名图")
