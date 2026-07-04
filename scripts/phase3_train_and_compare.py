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
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from minimal_scatlasvae import MinimalScAtlasVAE

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "study_name"
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


def main():
    adata = sc.read_h5ad(PROC_PATH)

    # 取原始计数、把 batch/label 字符串转成整数索引（手写模型吃索引）
    X = np.asarray(adata.layers["counts"].todense() if hasattr(adata.layers["counts"], "todense")
                   else adata.layers["counts"]).astype("float32")
    batch_idx = adata.obs[BATCH_KEY].astype("category").cat.codes.to_numpy()
    label_cat = adata.obs[LABEL_KEY].astype("category")
    label_idx = label_cat.cat.codes.to_numpy()

    # 训练手写模型（默认超参与官方一致：lr=5e-5, bs=128, seed=12, KL 预热）
    model = MinimalScAtlasVAE(
        n_genes=X.shape[1],
        n_batch=int(batch_idx.max()) + 1,
        n_label=int(label_idx.max()) + 1,
    )
    model.fit(X, batch_idx, labels=label_idx, device="cuda")
    adata.obsm["X_minimal"] = model.get_latent_embedding(X, device="cuda")

    # 定性：两套嵌入各出 UMAP（按 cell type 上色）
    for rep, tag in [("X_scAtlasVAE", "official"), ("X_minimal", "mine")]:
        sc.pp.neighbors(adata, use_rep=rep, key_added=tag)
        sc.tl.umap(adata, neighbors_key=tag)
        adata.obsm[f"X_umap_{tag}"] = adata.obsm["X_umap"]
        sc.pl.embedding(adata, basis=f"X_umap_{tag}", color=LABEL_KEY,
                        save=f"_{tag}.png", show=False)

    # 定量：邻域重叠
    ov = knn_overlap(adata.obsm["X_scAtlasVAE"], adata.obsm["X_minimal"])
    print(f"官方 vs 手写 的 kNN 邻域平均 Jaccard = {ov:.3f} （越高说明结构越一致）")
    adata.write_h5ad(PROC_PATH)


if __name__ == "__main__":
    main()
