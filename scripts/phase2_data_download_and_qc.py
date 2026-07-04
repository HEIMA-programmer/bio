"""阶段二 · 步骤 2–3：数据检查 + 预处理（QC / HVG / 归一化 / PCA 基线）。

用途
    把 TCellLandscape（GEO GSE156728）组装/载入为 AnnData，先按格式清单检查，
    再做标准预处理，产出供 scAtlasVAE 与评测使用的 h5ad。

用法（在环境 A `scatlasvae` 中）
    python phase2_data_download_and_qc.py --stage check       # 只检查格式、打印列名
    python phase2_data_download_and_qc.py --stage preprocess  # QC/HVG/归一化/PCA 并保存

前置
    先把下载好的原始数据整理成一个 .h5ad（X 为原始整数计数），路径填到 CONFIG。

对应报告
    reports/phase2_integration_and_benchmark.md 第 3、6 节。
"""
import argparse
import numpy as np
import scanpy as sc

# ============================================================
# CONFIG —— 用 --stage check 打印出真实列名后，回来改这里
# ============================================================
RAW_PATH = "TCellLandscape_raw.h5ad"      # 原始数据（X = 整数计数）
OUT_PATH = "tcell_processed.h5ad"          # 预处理输出
BATCH_KEY = "study_name"                    # 批次列（可能是 study_name/patient/cancerType）
LABEL_KEY = "cell_type"                     # 细胞类型列（可能是 meta.cluster/cell_type）
N_HVG = 4000                                # 论文对所有数据集统一取 4000 个高变基因
MITO_PREFIX = "MT-"                         # 线粒体基因前缀（人类为 MT-）


def _is_integer_matrix(X):
    """判断稀疏/稠密矩阵是否为整数计数（ZINB 重构要求 X 是原始 count）。"""
    sample = X[:200].toarray() if hasattr(X, "toarray") else np.asarray(X[:200])
    return bool(np.allclose(sample, np.round(sample)))


def check(adata):
    """§3 格式检查清单：把数据长什么样彻底打印出来，再决定怎么用。"""
    print(adata)
    print("\nobs 列（在这里找 batch 键和 cell type 列）:\n", list(adata.obs.columns))
    print("\nX 是否为整数计数 :", _is_integer_matrix(adata.X))
    total = np.asarray(adata.X.sum(axis=1)).ravel()
    print("每个细胞总计数 > 0 :", bool((total > 0).all()),
          " | 最小总计数 =", float(total.min()))
    for key in (BATCH_KEY, LABEL_KEY):
        if key in adata.obs:
            print(f"'{key}' 取值数 =", adata.obs[key].nunique())
        else:
            print(f"[注意] 未找到列 '{key}'，请从上面 obs 列里挑对的填回 CONFIG")


def preprocess(adata):
    """标准预处理：QC → 备份 counts → 归一化 → HVG(4000) → PCA(未校正基线)。"""
    # --- QC：过滤过少基因的细胞、过少细胞表达的基因，并按线粒体比例去将死细胞 ---
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith(MITO_PREFIX)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()  # 线粒体>20% 视为将死细胞

    # --- 关键：先备份原始计数（scAtlasVAE 的 ZINB 要用它），再做归一化 ---
    adata.layers["counts"] = adata.X.copy()

    # --- 归一化（供 PCA 与作为 log 表达参考；原理见 01 文档 §1.4f）---
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # --- HVG：只留 4000 个高变基因（与论文一致），并把矩阵裁到这些基因 ---
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, batch_key=BATCH_KEY)
    adata = adata[:, adata.var["highly_variable"]].copy()

    # --- 未校正基线 X_pca：在 log 归一化数据上做 PCA（不含任何批次校正）---
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)
    adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :50]

    # 恢复 X 为 log 归一化值（scale 后的 X 只用于 PCA），counts 仍在 layers 里
    adata.X = adata.layers["counts"].copy()   # 交给下游：X=raw counts, 需要时再取 layers
    print("预处理完成:", adata.shape, "| 已存 layers['counts'] 与 obsm['X_pca']")
    return adata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["check", "preprocess"], required=True)
    args = ap.parse_args()

    adata = sc.read_h5ad(RAW_PATH)
    if args.stage == "check":
        check(adata)
    else:
        adata = preprocess(adata)
        adata.write_h5ad(OUT_PATH)
        print("已保存 ->", OUT_PATH)


if __name__ == "__main__":
    main()
