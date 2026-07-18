"""阶段二 · 步骤 2（前置）：从 GEO 下载并组装 Zheng/GSE156728 10X CD8 原始计数 h5ad。

用途
    phase2_data_download_and_qc.py 从一个已存在的 TCellLandscape_raw.h5ad 起步，
    但"那份 h5ad 从哪来"一直缺一环。本脚本把这一环补上：直接从 GEO 拉取
    Zheng et al. 2021 泛癌 T 细胞图谱（GSE156728）的 10X CD8 计数矩阵与元数据，
    组装成 scAtlasVAE 能直接吃的原始整数计数 AnnData。

规模（2026-07 更新：默认改为"全量"）
    经核对元数据，GSE156728 的 8 个 10X CD8 癌种共 ~104,805 细胞（THCA 33450 / UCEC 19926 /
    RC 16544 / ESCA 12526 / MM 8629 / PACA 5957 / BC 4291 / BCL 3482；45 个病人），
    量级与论文 110,218-cell benchmark 接近。此前默认下采样到 4 万只是为了快，
    故**默认改为不封顶、不下采样、取当前 8 癌种全量**。这只是同量级真实数据上的
    近似复现，不是带 28 个 ``study_name`` 的论文成品 TCellLandscape。仍保留 --target / --max-per-cancer
    以便需要小规模先跑通时手动下采样。

内存安全（关键）
    本机 15.6GB RAM。dense 文本矩阵若整读会吃爆内存（THCA 24148×33450×8 ≈ 6GB）。
    本脚本改为**分块流式读取**（pandas chunksize 每次 2000 个基因行，逐块转稀疏后 vstack），
    峰值内存只有一个块（<1.5GB），与总细胞数无关，故全量也安全。

数据结构（侦查 GEO suppl 得到，见 reports/phase2 §3）
    - GSE156728_metadata.txt.gz：全体细胞的元数据。列 =
        cellID, cancerType, patient, libraryID, loc, meta.cluster, platform
      其中 meta.cluster 形如 'CD8.c01.Tex.CXCL13'（CD8 亚型）或 'CD4.c...'。
    - GSE156728_<CANCER>_10X.CD8.counts.txt.gz：按癌种的 CD8 计数矩阵，
        **行 = 基因（~24148），列 = 细胞**（列名即 cellID），值 = 原始 UMI 计数。
      可用的 10X CD8 癌种：BCL, BC, ESCA, MM, PACA, RC, THCA, UCEC。

    组装映射：计数矩阵的列名（cellID） == 元数据的 cellID，据此把 cancerType / patient /
    loc / meta.cluster 贴到每个细胞。batch 取 **patient**（多样本整合的真实挑战），
    cell_type 取 **meta.cluster**（CD8 亚型，供半监督分类头与评测的 label）。

用法（在环境 A `scatlasvae` 中，于数据工作目录下运行）
    python phase2_data_fetch_gse156728.py                 # 默认：全量（~10.5 万）
    python phase2_data_fetch_gse156728.py --target 40000  # 想要小规模：随机下采样到 4 万
    python phase2_data_fetch_gse156728.py --cancers BC ESCA THCA   # 只用部分癌种

产出
    TCellLandscape_raw.h5ad：X = 原始整数计数（CSR 稀疏）。文件名为历史兼容名，
    内容实际是本脚本重建的 Zheng/GSE156728 8 癌种 10X CD8 对象，并非论文成品 TCellLandscape；
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
from scipy.sparse import csr_matrix, vstack

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156728/suppl"
META_FILE = "GSE156728_metadata.txt.gz"
# 8 个有独立 10X CD8 计数文件的癌种（GEO suppl 侦查所得）
ALL_CANCERS = ["BCL", "BC", "ESCA", "MM", "PACA", "RC", "THCA", "UCEC"]
OUT_PATH = "TCellLandscape_raw.h5ad"
BIG = 10_000_000  # "不封顶/不下采样"的哨兵值


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
    """流式载入某癌种 10X CD8 计数矩阵（分块读，内存安全），返回 cells×genes 稀疏 AnnData。"""
    fname = f"GSE156728_{cancer}_10X.CD8.counts.txt.gz"
    _download(fname)

    # 1) 只读表头，拿到该文件里所有细胞列名（cellID）
    with gzip.open(fname, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    file_cells = header[1:]  # header[0] 是基因列的空表头

    # 2) 与元数据的 CD8 细胞求交集；默认不封顶（max_per_cancer=BIG），需要时才随机下采样
    cancer_cd8 = set(cd8_meta.index[cd8_meta["cancerType"] == cancer])
    keep = [c for c in file_cells if c in cancer_cd8]
    if len(keep) == 0:
        print(f"  [警告] {cancer}: 计数矩阵与元数据无交集 CD8 细胞，跳过")
        return None
    if len(keep) > max_per_cancer:
        rng = np.random.default_rng(seed)
        keep = list(rng.choice(keep, size=max_per_cancer, replace=False))
    keep_set = set(keep)
    name_to_pos = {c: i + 1 for i, c in enumerate(file_cells)}  # 列名 -> 文件列位置（基因列占 0）
    usecols = [0] + [name_to_pos[c] for c in keep]

    # 3) 分块流式读：每次读 2000 个基因行，逐块转稀疏后 vstack。
    #    峰值内存只有一个块（~500MB），与细胞数无关，故全量也安全。
    #    注意：read_csv(usecols=...) 返回的列顺序 = 文件原始列序（与 usecols 顺序无关）。
    gene_names, blocks = [], []
    for chunk in pd.read_csv(fname, sep="\t", header=0, usecols=usecols, chunksize=2000):
        chunk = chunk.set_index(chunk.columns[0])  # 第 0 列（基因）作行索引
        gene_names.extend(chunk.index.astype(str).tolist())
        blocks.append(csr_matrix(chunk.values.astype(np.float32)))
    mat = vstack(blocks).tocsr()  # 基因 × 细胞

    # 个别基因符号在文件内重复，会让后面按基因求交集的 concat 报 "duplicate labels"；
    # 按基因符号去重（保留首次出现），使 var_names 唯一。
    gn = pd.Index(gene_names)
    dup = gn.duplicated(keep="first")
    if dup.any():
        mat = mat[np.where(~dup)[0]]
        gn = gn[~dup]

    X = csr_matrix(mat.T)  # 细胞 × 基因
    adata = sc.AnnData(X=X)
    adata.obs_names = [c for c in file_cells if c in keep_set]  # 与列序（文件序）对齐
    adata.var_names = gn.astype(str)
    print(f"  {cancer}: 载入 {adata.n_obs} 细胞 × {adata.n_vars} 基因（流式/去重后）")
    return adata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancers", nargs="+", default=ALL_CANCERS,
                    help="用哪些癌种（默认 8 个全用）")
    ap.add_argument("--target", type=int, default=BIG,
                    help="最终下采样目标细胞数（默认不下采样=全量；想小规模就设如 40000）")
    ap.add_argument("--max-per-cancer", type=int, default=BIG,
                    help="每个癌种最多取多少 CD8 细胞（默认不封顶=全量）")
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
    del adatas

    # 贴回元数据：cancerType / patient / loc / cell_type / platform
    m = cd8_meta.reindex(adata.obs_names)
    adata.obs["cancerType"] = pd.Categorical(m["cancerType"].values)
    adata.obs["patient"] = pd.Categorical(m["patient"].values)
    adata.obs["loc"] = pd.Categorical(m["loc"].values)
    adata.obs["cell_type"] = pd.Categorical(m["meta.cluster"].values)  # CD8 亚型
    adata.obs["platform"] = pd.Categorical(m["platform"].values)

    # 可选全局下采样到 target（默认 BIG=不触发；分层按 patient 保住 batch 多样性）
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
