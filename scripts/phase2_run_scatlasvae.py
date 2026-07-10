"""阶段二 · 步骤 4 & 6：训练 scAtlasVAE 得到整合嵌入，并出 UMAP/Leiden。

用法（在环境 A `scatlasvae` 中）
    python phase2_run_scatlasvae.py --stage train   # 训练 -> obsm['X_scAtlasVAE']
    python phase2_run_scatlasvae.py --stage umap    # 近邻图/UMAP/Leiden + 出图

对应报告
    reports/phase2_integration_and_benchmark.md 步骤 4、6。
"""
import argparse
import numpy as np
import scanpy as sc
import scatlasvae

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"        # 与 phase2_data_download_and_qc.py 保持一致
LABEL_KEY = "cell_type"
LOSS_PATH = "phase2_scatlasvae_loss.npz"   # 训练动态（loss 曲线 + λ_KL 预热真值）


def train():
    adata = sc.read_h5ad(PROC_PATH)
    # scAtlasVAE 的 ZINB 需要原始整数计数作为输入。preprocess 已把 X 存为 counts；
    # 若你的 X 是 log 值，请改成 adata.X = adata.layers['counts'].copy()
    model = scatlasvae.model.scAtlasVAE(
        adata=adata,               # 注意：keyword-only，必须写 adata=adata
        batch_key=BATCH_KEY,
        label_key=LABEL_KEY,       # 提供标签即启用半监督分类头（见 01 文档 §1.4h）
        batch_embedding="embedding",
        batch_hidden_dim=10,
        device="cuda:0",
    )
    # fit() 返回逐 epoch 的各项 loss；接住它以画训练曲线。
    history = model.fit()          # epoch 数按 min(round(20000/N*400),400) 自动决定

    # 记录 λ_KL 的真实预热轨迹：源码 fit() 里 n_epochs_kl_warmup=min(max_epoch,400)，
    # 权重每个 epoch 末 +1/warmup，故第 e 个 epoch（0-indexed）实际用的权重 = e/warmup。
    # 对 4 万细胞 max_epoch≈73<400 → warmup=73 → λ_KL 全程 0→~1（**证伪旧文档"只到0.18"的说法**）。
    n_epoch = len(history["epoch_total_loss_list"])
    warmup = min(n_epoch, 400)
    kl_weight = np.minimum(1.0, np.arange(n_epoch) / warmup)
    np.savez(
        LOSS_PATH,
        kl_weight=kl_weight,
        **{k: np.asarray(v, dtype=float) for k, v in history.items()},
    )
    print(f"训练 {n_epoch} epoch；λ_KL 末值 ≈ {kl_weight[-1]:.3f}"
          f"（若是旧文档说的 0.18 才对；实际应接近 1）-> 已存 {LOSS_PATH}")

    adata.obsm["X_scAtlasVAE"] = model.get_latent_embedding()
    adata.write_h5ad(PROC_PATH)
    model.save_to_disk("scatlasvae_tcell.pt")
    print("训练完成，已写入 obsm['X_scAtlasVAE']")


def umap():
    adata = sc.read_h5ad(PROC_PATH)
    # 对"未校正基线"和"scAtlasVAE 嵌入"各走一遍 近邻图 -> UMAP -> Leiden，便于对比
    for rep, tag in [("X_pca", "pca"), ("X_scAtlasVAE", "scatlasvae")]:
        sc.pp.neighbors(adata, use_rep=rep, n_neighbors=15, key_added=tag)
        sc.tl.umap(adata, neighbors_key=tag)
        adata.obsm[f"X_umap_{tag}"] = adata.obsm["X_umap"]
        sc.tl.leiden(adata, neighbors_key=tag, key_added=f"leiden_{tag}", resolution=1.0)
        # 按 batch 和按 cell type 两种上色，直观看整合前后差别
        sc.pl.embedding(adata, basis=f"X_umap_{tag}", color=[BATCH_KEY, LABEL_KEY],
                        save=f"_{tag}.png", show=False)
    adata.write_h5ad(PROC_PATH)
    print("UMAP/Leiden 完成，图见 figures/ 目录")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "umap"], default="train")
    args = ap.parse_args()
    train() if args.stage == "train" else umap()
