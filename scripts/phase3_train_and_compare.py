"""阶段三 · 训练手写 VAE 并与官方实现定量对照。

流程
    1) 载入阶段二处理好的 h5ad（含 counts / batch / label / 官方 obsm['X_scAtlasVAE']）；
    2) 用手写的 MinimalScAtlasVAE 训练，得到 obsm['X_minimal']；
    3) 定性对比：两者各出一张 UMAP，并排看；
    4) 定量对比：算两套嵌入的 kNN 邻域重叠（Jaccard），衡量"是否抓到相似结构"。

用法（在环境 A `scatlasvae` 中，需 torch/scanpy/sklearn）
    python phase3_train_and_compare.py

对应报告：reports/phase3_reimplement_vae.md
"""
import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.neighbors import NearestNeighbors

from minimal_scatlasvae import MinimalScAtlasVAE

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"
LABEL_KEY = "cell_type"


def knn_overlap(emb_a, emb_b, k=30, n_sample=2000, seed=0):
    """两套嵌入的邻域一致性：随机取若干细胞，比较它们在 A、B 中的 k 近邻集合的平均 Jaccard。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(emb_a.shape[0], size=min(n_sample, emb_a.shape[0]), replace=False)
    nn_a = NearestNeighbors(n_neighbors=k + 1).fit(emb_a)
    nn_b = NearestNeighbors(n_neighbors=k + 1).fit(emb_b)
    _, ia = nn_a.kneighbors(emb_a[idx])
    _, ib = nn_b.kneighbors(emb_b[idx])
    jacc = []
    for ra, rb in zip(ia, ib):
        sa, sb = set(ra[1:]), set(rb[1:])          # 去掉自身
        jacc.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(jacc))


def train_minimal(adata):
    """训练手写实现并写入 ``X_minimal``。"""
    # 取原始计数、把 batch/label 字符串转成整数索引（手写模型吃索引）
    X = np.asarray(adata.layers["counts"].todense() if hasattr(adata.layers["counts"], "todense")
                   else adata.layers["counts"]).astype("float32")
    batch_idx = adata.obs[BATCH_KEY].astype("category").cat.codes.to_numpy()
    label_cat = adata.obs[LABEL_KEY].astype("category")
    label_idx = label_cat.cat.codes.to_numpy()

    # seed 必须在模型构造前设置，才能同时固定初始权重与 fit 内随机过程。
    seed = 12
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 训练手写模型（默认超参与官方一致：lr=5e-5, bs=128, seed=12, KL 预热）
    model = MinimalScAtlasVAE(
        n_genes=X.shape[1],
        n_batch=int(batch_idx.max()) + 1,
        n_label=int(label_idx.max()) + 1,
    )
    model.fit(X, batch_idx, labels=label_idx, seed=seed, device="cuda")
    adata.obsm["X_minimal"] = model.get_latent_embedding(X, device="cuda")


def compare_existing_embeddings(adata):
    """用当前官方嵌入重新计算 UMAP 与邻域重叠，不重训手写模型。"""
    official_key = (
        "X_scAtlasVAE_sup"
        if "X_scAtlasVAE_sup" in adata.obsm
        else "X_scAtlasVAE"
    )
    if "X_minimal" not in adata.obsm:
        raise KeyError("缺少 X_minimal；请先用 --stage train 训练手写模型")

    # 定性：两套嵌入各出 UMAP（按 cell type 上色）
    for rep, tag in [(official_key, "official"), ("X_minimal", "mine")]:
        sc.pp.neighbors(adata, use_rep=rep, key_added=tag)
        sc.tl.umap(adata, neighbors_key=tag)
        adata.obsm[f"X_umap_{tag}"] = adata.obsm["X_umap"]

    # 定量：邻域重叠
    ov = knn_overlap(adata.obsm[official_key], adata.obsm["X_minimal"])
    print(f"官方 vs 手写 的 kNN 邻域平均 Jaccard = {ov:.3f} （越高说明结构越一致）")
    pd.DataFrame([{
        "official_embedding": official_key,
        "minimal_embedding": "X_minimal",
        "k": 30,
        "n_sample": min(2000, adata.n_obs),
        "mean_knn_jaccard": ov,
    }]).to_csv("phase3_knn_overlap.csv", index=False)


def main(stage):
    adata = sc.read_h5ad(PROC_PATH)
    if stage == "train":
        train_minimal(adata)
    compare_existing_embeddings(adata)
    adata.write_h5ad(PROC_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["train", "compare"], default="train",
        help="train=重训手写模型后比较；compare=复用 X_minimal，仅刷新官方对比",
    )
    args = parser.parse_args()
    main(args.stage)
