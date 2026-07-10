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
BATCH_KEY = "patient"
LABEL_KEY = "cell_type"
# 消融关心的是"改一个旋钮、相对排序变不变"，不需要每个配置都训满 200 epoch。
# 4 个配置若各训 200 epoch 约需 ~2 小时；在 4060 笔记本上把每个配置截到 100 epoch，
# 足够看出相对差异，总时长 ~1 小时。注意：预热长度 = min(max_epoch,400)=100，
# 所以"默认预热"这一支 λ_KL 在这 100 epoch 内 0→1 爬满（与 nowarmup 支恒 1.0 形成对照）。
MAX_EPOCH_ABL = 100


def train():
    import scatlasvae
    adata = sc.read_h5ad(PROC_PATH)

    # 消融 1：潜维度。其余全用默认（含 KL 预热）
    for k in (2, 10, 50):
        m = scatlasvae.model.scAtlasVAE(
            adata=adata, batch_key=BATCH_KEY, label_key=LABEL_KEY,
            n_latent=k, device="cuda:0",
        )
        m.fit(max_epoch=MAX_EPOCH_ABL)
        adata.obsm[f"X_nlat{k}"] = m.get_latent_embedding()

    # 消融 2：关掉 KL 预热（n_epochs_kl_warmup=0 → 第一轮就给满权重 1.0）
    # 对照支 = 上面 n_latent=10 那份（有预热，λ_KL 在 100 epoch 内 0→1 爬满）
    m = scatlasvae.model.scAtlasVAE(
        adata=adata, batch_key=BATCH_KEY, label_key=LABEL_KEY,
        n_latent=10, device="cuda:0",
    )
    m.fit(max_epoch=MAX_EPOCH_ABL, n_epochs_kl_warmup=0)
    adata.obsm["X_nowarmup"] = m.get_latent_embedding()

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
