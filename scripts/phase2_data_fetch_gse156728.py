"""阶段二 · 步骤 2（前置）：从 GEO 下载并组装 TCellLandscape（GSE156728）为原始计数 h5ad。

用途
    phase2_data_download_and_qc.py 从一个已存在的 TCellLandscape_raw.h5ad 起步，
    但"那份 h5ad 从哪来"一直缺一环。本脚本把这一环补上：直接从 GEO 拉取
    Zheng et al. 2021 泛癌 T 细胞图谱（GSE156728）的 10X CD8 计数矩阵与元数据，
    组装成 scAtlasVAE 能直接吃的原始整数计数 AnnData，并下采样到可在 4060 上舒适训练的规模。

数据结构（侦查 GEO suppl 得到，见 reports/phase2 §3）
    - GSE156728_metadata.txt.gz：全体细胞的元数据。列 =
        cellID, cancerType, patient, libraryID, loc, meta.cluster, platform
      其中 meta.cluster 形如 'CD8.c01.Tex.CXCL13'（CD8 亚型）或 'CD4.c...'。
    - GSE156728_<CANCER>_10X.CD8.counts.txt.gz：按癌种的 CD8 计数矩阵，
        **行 = 基因（~24148），列 = 细胞**（列名即 cellID，如 'AAACGGGAGCCACCTG.1'），值 = 原始 UMI 计数。
      可用的 10X CD8 癌种：BCL, BC, ESCA, MM, PACA, RC, THCA, UCEC。

    组装映射：计数矩阵的列名（cellID） == 元数据的 cellID，据此把 cancerType / patient /
    loc / meta.cluster 贴到每个细胞。batch 取 **patient**（多样本整合的真实挑战），
    cell_type 取 **meta.cluster**（CD8 亚型，供半监督分类头与评测的 label）。

用法（在环境 A `scatlasvae` 中，于一个数据工作目录下运行）
    python phase2_data_fetch_gse156728.py                 # 默认下采样到 ~40000 细胞
    python phase2_data_fetch_gse156728.py --target 15000  # 更小规模先跑通
    python phase2_data_fetch_gse156728.py --cancers BC ESCA THCA   # 只用部分癌种

内存友好设计
    dense 文本矩阵很大，若整读再筛选会吃爆内存。本脚本先读**表头**拿到该文件的细胞列，
    与元数据里该癌种的 CD8 细胞求交集并封顶，再用 pandas 的 `usecols` **只读选中的列**，
    从源头把内存/耗时压下来。

产出
    TCellLandscape_raw.h5ad：X = 原始整数计数（CSR 稀疏），
    obs 含 cancerType / patient / loc / cell_type / platform。
    随后交给 phase2_data_download_and_qc.py --stage check / preprocess。

对应报告
    reports/phase2_integration_and_benchmark.md 第 3 节（数据侦查与验货）。
"""
import argparse
import gzip
import os
import urllib.request

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156728/suppl"
META_FILE = "GSE156728_metadata.txt.gz"
# 8 个有独立 10X CD8 计数文件的癌种（GEO suppl 侦查所得）
ALL_CANCERS = ["BCL", "BC", "ESCA", "MM", "PACA", "RC", "THCA", "UCEC"]
OUT_PATH = "TCellLandscape_raw.h5ad"


def _download(fname: str):
    """下载单个 GEO suppl 文件到当前目录；已存在则跳过（便于断点续跑）。"""
    if os.path.exists(fname):
        print(f"  [跳过] 已有 {fname}（{os.path.getsize(fname) / 1048576:.1f} MB）")
        return fname
    url = f"{GEO_BASE}/{fname}"
    print(f"  [下载] {url}")
    tmp = fname + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, fname)
    print(f"         -> {fname}（{os.path.getsize(fname) / 1048576:.1f} MB）")
    return fname


def load_metadata():
    """读元数据，返回以 cellID 为索引的 DataFrame（只保留 CD8 细胞）。"""
    _download(META_FILE)
    meta = pd.read_csv(META_FILE, sep="\t")
    meta = meta.set_index("cellID")
    cd8 = meta[meta["meta.cluster"].astype(str).str.startswith("CD8")]
    print(f"元数据：共 {len(meta)} 细胞，其中 CD8 亚型 {len(cd8)} 细胞；"
          f"平台分布 = {dict(meta['platform'].value_counts())}")
    return cd8


def load_cancer_counts(cancer: str, cd8_meta: pd.DataFrame, max_per_cancer: int, seed: int):
    """载入某癌种的 10X CD8 计数矩阵，只读选中的 CD8 细胞列，返回 cells×genes 的 AnnData。"""
    fname = f"GSE156728_{cancer}_10X.CD8.counts.txt.gz"
    _download(fname)

    # 1) 只读表头，拿到该文件里所有细胞列名（cellID）
    with gzip.open(fname, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    file_cells = header[1:]  # header[0] 是基因列的**空表头**（pandas 会把它重命名为 'Unnamed: 0'）

    # 2) 与元数据的 CD8 细胞求交集，并封顶到 max_per_cancer（分层下采样在此发生）
    cancer_cd8 = set(cd8_meta.index[cd8_meta["cancerType"] == cancer])
    name_to_pos = {c: i + 1 for i, c in enumerate(file_cells)}  # 列名 -> 文件里的列位置（基因列占 0）
    keep = [c for c in file_cells if c in cancer_cd8]
    if len(keep) == 0:
        print(f"  [警告] {cancer}: 计数矩阵与元数据无交集 CD8 细胞，跳过")
        return None
    rng = np.random.default_rng(seed)
    if len(keep) > max_per_cancer:
        keep = list(rng.choice(keep, size=max_per_cancer, replace=False))

    # 3) 用**整数位置** usecols 只读选中的列（基因列 0 + 选中细胞列）。
    #    关键：基因列表头为空，pandas 会改名成 'Unnamed: 0'，用列名匹配会漏掉它；用位置最稳。
    usecols = [0] + [name_to_pos[c] for c in keep]
    df = pd.read_csv(fname, sep="\t", header=0, usecols=usecols)
    df = df.set_index(df.columns[0])  # 第 0 列（基因）作行索引

    # 不同癌种文件基因集不同（有的 24148、有的 28855），且个别基因符号在文件内**重复**，
    # 会让后面按基因求交集的 concat 报 "duplicate labels"。这里先按基因符号去重（保留首次出现），
    # 使每个文件的 var_names 唯一，再交由 concat 取交集。
    df = df[~df.index.duplicated(keep="first")]
    # df: 基因(行) × 细胞(列)。转成 细胞×基因 的稀疏 AnnData。
    X = csr_matrix(df.T.values.astype(np.float32))
    adata = sc.AnnData(X=X)
    adata.obs_names = df.columns
    adata.var_names = df.index.astype(str)
    print(f"  {cancer}: 载入 {adata.n_obs} 细胞 × {adata.n_vars} 基因（去重后）")
    return adata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancers", nargs="+", default=ALL_CANCERS,
                    help="用哪些癌种（默认 8 个全用）")
    ap.add_argument("--target", type=int, default=40000,
                    help="最终下采样目标细胞数（默认 40000）")
    ap.add_argument("--max-per-cancer", type=int, default=7000,
                    help="每个癌种最多取多少 CD8 细胞（控内存）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cd8_meta = load_metadata()

    adatas = []
    for cancer in args.cancers:
        a = load_cancer_counts(cancer, cd8_meta, args.max_per_cancer, args.seed)
        if a is not None:
            adatas.append(a)
    if not adatas:
        raise RuntimeError("没有成功载入任何癌种的数据")

    # 各癌种共享基因集时 inner==outer；用 inner 避免零填充膨胀
    print("拼接各癌种（按共享基因）...")
    adata = sc.concat(adatas, join="inner", index_unique=None)

    # 贴回元数据：cancerType / patient / loc / cell_type / platform
    m = cd8_meta.reindex(adata.obs_names)
    adata.obs["cancerType"] = pd.Categorical(m["cancerType"].values)
    adata.obs["patient"] = pd.Categorical(m["patient"].values)
    adata.obs["loc"] = pd.Categorical(m["loc"].values)
    adata.obs["cell_type"] = pd.Categorical(m["meta.cluster"].values)  # CD8 亚型
    adata.obs["platform"] = pd.Categorical(m["platform"].values)

    # 全局下采样到 target（分层按 patient，保住 batch 多样性）
    if adata.n_obs > args.target:
        rng = np.random.default_rng(args.seed)
        frac = args.target / adata.n_obs
        idx = []
        for _, grp in adata.obs.groupby("patient", observed=True):
            n = max(1, int(round(len(grp) * frac)))
            idx.extend(rng.choice(grp.index, size=min(n, len(grp)), replace=False))
        adata = adata[np.array(idx)].copy()

    # 兜底：确保每细胞总计数 > 0（否则 ZINB 训练出 NaN，见 README/phase1）
    tot = np.asarray(adata.X.sum(1)).ravel()
    if not (tot > 0).all():
        adata = adata[tot > 0].copy()

    # X 存为 int32（scAtlasVAE 对非 int32 会 warn；这里就是原始计数）
    adata.X = csr_matrix(adata.X, dtype=np.int32)

    adata.write_h5ad(OUT_PATH)
    print("\n组装完成 ->", OUT_PATH)
    print(f"  形状: {adata.shape}")
    print(f"  batch(patient) 类别数: {adata.obs['patient'].nunique()}")
    print(f"  cancerType 类别数    : {adata.obs['cancerType'].nunique()}")
    print(f"  cell_type 类别数     : {adata.obs['cell_type'].nunique()}")
    print(f"  每细胞总计数>0        : {bool((np.asarray(adata.X.sum(1)).ravel() > 0).all())}")


if __name__ == "__main__":
    main()
