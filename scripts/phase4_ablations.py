"""阶段四 · 消融实验：控制单一变量，验证作者的设计选择是否必要。

做两个消融（每次只改一个旋钮，其余照默认）：
    1) 潜维度 n_latent ∈ {2, 10, 50}
    2) KL 预热 开 / 关（fit 的 n_epochs_kl_warmup=默认 vs 0）

为省事，消融直接用官方 scAtlasVAE（它的构造/fit 参数正好能改这两个旋钮）。
训练在环境 A 产出各嵌入；评测在环境 B 用 scib-metrics 打分。

用法
    conda activate scatlasvae && python phase4_ablations.py --stage train
    conda activate scib       && python phase4_ablations.py --stage benchmark

对应报告：reports/phase4_ablation_studies.md
"""
import argparse
import scanpy as sc

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "study_name"
LABEL_KEY = "cell_type"


def train():
    import scatlasvae
    adata = sc.read_h5ad(PROC_PATH)

    # 消融 1：潜维度。其余全用默认（含 KL 预热）
    for k in (2, 10, 50):
        m = scatlasvae.model.scAtlasVAE(
            adata=adata, batch_key=BATCH_KEY, label_key=LABEL_KEY,
            n_latent=k, device="cuda:0",
        )
        m.fit()
        adata.obsm[f"X_nlat{k}"] = m.get_latent_embedding()

    # 消融 2：关掉 KL 预热（n_epochs_kl_warmup=0 → 第一轮就给满权重）
    m = scatlasvae.model.scAtlasVAE(
        adata=adata, batch_key=BATCH_KEY, label_key=LABEL_KEY,
        n_latent=10, device="cuda:0",
    )
    m.fit(n_epochs_kl_warmup=0)
    adata.obsm["X_nowarmup"] = m.get_latent_embedding()
    # 有预热的基线就是上面的 X_nlat10

    adata.write_h5ad(PROC_PATH)
    print("消融训练完成：obsm 里新增 X_nlat2 / X_nlat10 / X_nlat50 / X_nowarmup")


def benchmark():
    from scib_metrics.benchmark import Benchmarker
    adata = sc.read_h5ad(PROC_PATH)
    keys = ["X_nlat2", "X_nlat10", "X_nlat50", "X_nowarmup"]
    bm = Benchmarker(adata, batch_key=BATCH_KEY, label_key=LABEL_KEY,
                     embedding_obsm_keys=keys, n_jobs=-1)
    bm.benchmark()
    res = bm.get_results(min_max_scale=False)
    print(res)
    res.to_csv("phase4_ablation_results.csv")
    print("完成：见 phase4_ablation_results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "benchmark"], required=True)
    args = ap.parse_args()
    train() if args.stage == "train" else benchmark()
