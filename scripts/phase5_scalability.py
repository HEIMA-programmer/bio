"""阶段五 · 可扩展性曲线：训练时间 / 显存 随细胞数如何增长（对标论文 Ext. Data Fig. 4e,f）。

动机
    论文 Ext. Data Fig. 4e,f 显示 scAtlasVAE 的时间与显存随细胞数**近似线性**增长，
    且优于 scPoli/SCALEX，是"可扩展到图谱级"的证据。我们此前没做这条。这里在本机 4060 上，
    对递增的细胞数子集**各训练固定 epoch 数**，测每次的墙钟训练时间与峰值显存，画出曲线。

设计
    - 从 tcell_processed.h5ad 里按 patient 分层子采样出 n ∈ {10k, 30k, 60k, 100k} 细胞。
    - 每个规模都用**相同的固定 epoch 数**（默认 20）训练官方 scAtlasVAE（监督），
      这样"每细胞成本"才可比（否则 fit() 会按规模自动改 epoch 数、混淆变量）。
    - 记录：fit() 墙钟秒数、torch.cuda.max_memory_allocated 峰值显存(MB)。
    - 只读主 h5ad、只在子集上临时训练，**不写回任何 obsm**，与主流水线无冲突。

用法（环境 A `scatlasvae`）
    python phase5_scalability.py                       # 默认 10k/30k/60k/100k，各 20 epoch
    python phase5_scalability.py --sizes 10000 50000   # 自定义规模
    python phase5_scalability.py --epochs 15

产出
    phase5_scalability.csv：n_cells, fit_seconds, peak_gpu_mb, sec_per_epoch, sec_per_10k_cells

对应报告
    reports/phase5_deeper_validation.md（可扩展性一节，对标 Ext. Data Fig. 4e,f）。
"""
import argparse
import time

import numpy as np
import scanpy as sc
import torch

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"
LABEL_KEY = "cell_type"
OUT = "phase5_scalability.csv"


def stratified_subsample(adata, n_target, seed=0):
    """按 patient 分层子采样到 ~n_target 细胞，保住 batch 多样性。"""
    if adata.n_obs <= n_target:
        return adata.copy()
    rng = np.random.default_rng(seed)
    frac = n_target / adata.n_obs
    idx = []
    for _, grp in adata.obs.groupby(BATCH_KEY, observed=True):
        k = max(1, int(round(len(grp) * frac)))
        idx.extend(rng.choice(grp.index, size=min(k, len(grp)), replace=False))
    return adata[np.array(idx)].copy()


def measure(adata_sub, epochs):
    """在子集上训练固定 epoch 的官方 scAtlasVAE，返回 (墙钟秒, 峰值显存MB)。"""
    import scatlasvae
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = scatlasvae.model.scAtlasVAE(
        adata=adata_sub, batch_key=BATCH_KEY, label_key=LABEL_KEY, device="cuda:0",
    )
    t0 = time.time()
    m.fit(max_epoch=epochs)
    dt = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1048576
    del m
    torch.cuda.empty_cache()
    return dt, peak_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 30000, 60000, 100000])
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    adata = sc.read_h5ad(PROC_PATH)
    rows = []
    for n in args.sizes:
        sub = stratified_subsample(adata, n)
        dt, peak = measure(sub, args.epochs)
        n_real = sub.n_obs
        rows.append((n_real, dt, peak, dt / args.epochs, dt / (n_real / 10000)))
        print(f"n={n_real:>6}  fit={dt:7.1f}s  peak={peak:7.0f}MB  "
              f"{dt/args.epochs:5.2f}s/epoch  {dt/(n_real/10000):5.1f}s/10k细胞", flush=True)

    import csv
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_cells", "fit_seconds", "peak_gpu_mb", "sec_per_epoch", "sec_per_10k_cells"])
        w.writerows(rows)
    print(f"完成 -> {OUT}（对标论文 Ext. Data Fig. 4e,f：看时间/显存是否随细胞数近似线性）")


if __name__ == "__main__":
    main()
