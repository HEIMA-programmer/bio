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
# 阶段 5 · E2：四方对比，复现论文 Ext. Data Fig. 2a 把 scAtlasVAE 分"无监督/监督"两根柱的核心论点。
#   X_pca              未校正基线
#   X_scVI             经典 batch-variant VAE（编码器吃 batch）
#   X_scAtlasVAE_unsup 无监督 scAtlasVAE（不带 label_key，纯整合）—— 预期 ≈ scVI
#   X_scAtlasVAE_sup   监督 scAtlasVAE（带 label_key、半监督分类头）—— 预期最高
# 旧名 X_scAtlasVAE 即监督版；--mode unsup 那趟已补别名 X_scAtlasVAE_sup。
# Harmony 结果仍可选（见 phase2_baseline_harmony.py）。
EMBEDDINGS = ["X_pca", "X_scVI", "X_scAtlasVAE_unsup", "X_scAtlasVAE_sup"]

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
