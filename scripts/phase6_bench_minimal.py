"""阶段六 · E4：把"从零手写的最小 VAE"放上同一把 scib 标尺。

动机
    阶段三我们手写了 minimal_scatlasvae.py（批不变编码器 / 重参数化 / 批条件解码器 /
    ZINB / KL 预热 / 单分类头），此前只有"官方 vs 手写 UMAP 定性一致 + kNN Jaccard=0.235"。
    这里把手写实现产出的 obsm['X_minimal'] 与 PCA / scVI / 官方监督版 **并列打分**，
    给出"我的实现落在什么水平"的**定量**结论。

对照
    X_pca(未校正) / X_scVI / X_scAtlasVAE_sup(官方监督) / X_minimal(手写)
用法（环境 B `scib`，py3.10）
    python phase6_bench_minimal.py
产出
    phase6_minimal_bench.csv
对应报告
    reports/phase6_deeper_validation.md（E4）与 reports/phase3_reimplement_vae.md。
"""
import scanpy as sc
from scib_metrics.benchmark import Benchmarker

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"
LABEL_KEY = "cell_type"
# 手写实现和三个参照并列；把 X_minimal 放最后，读表时一眼看它落在哪。
EMBEDDINGS = ["X_pca", "X_scVI", "X_scAtlasVAE_sup", "X_minimal"]

adata = sc.read_h5ad(PROC_PATH)
missing = [k for k in EMBEDDINGS if k not in adata.obsm]
if missing:
    raise KeyError(f"缺少 obsm: {missing}（X_scAtlasVAE_sup 需先跑 phase2_run --mode unsup 补别名；"
                   f"X_minimal 需先跑 phase3_train_and_compare.py 写入）")

bm = Benchmarker(
    adata,
    batch_key=BATCH_KEY,
    label_key=LABEL_KEY,
    embedding_obsm_keys=EMBEDDINGS,
    n_jobs=-1,
)
bm.benchmark()
results = bm.get_results(min_max_scale=False)
print(results)
results.to_csv("phase6_minimal_bench.csv")
print("完成：见 phase6_minimal_bench.csv")
