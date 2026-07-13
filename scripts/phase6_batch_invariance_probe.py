"""阶段六 · E3：批不变编码器的"实证探针"。

论点
    scAtlasVAE 的招牌是**编码器 batch-invariant**——`_gex_model.py:969-970` 把
    "batch 拼进编码器输入"的那行**注释掉了**，所以潜向量 z 只由基因表达 X 决定、与 batch 无关。
    这正是它能"查询数据不重训直接映射进参考图谱(zero-shot)"的根因，也是与 scVI
    （编码器 F(X,B,S) 显式吃 batch）的本质区别。

做法（低层直接给编码器喂不同的 batch 索引，最干净的证明）
    同一批细胞 X，分别用 **真实 batch / 打乱 batch / 全 None** 过编码器，比较潜均值：
      - scAtlasVAE：q_mu(real) 与 q_mu(perm)、q_mu(None) 应**逐元素完全相同**（max|Δ|≈0）。
      - scVI     ：z(real) 与 z(perm) 应**明显漂移**（编码器吃 batch，改 batch 就改 z）。
    "我读到那行被注释" -> 升级成 "我测出来了"。

用法
    conda activate scatlasvae && python phase6_batch_invariance_probe.py --model scatlasvae
    conda activate scvi       && python phase6_batch_invariance_probe.py --model scvi
产出
    phase6_invariance_scatlasvae.csv / phase6_invariance_scvi.csv（各写一行指标）
    phase6_invariance_z.npz（各模型 real/perm 的一小段 z，供画图）
对应报告
    reports/phase6_deeper_validation.md（E3）。
"""
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"
LABEL_KEY = "cell_type"
N_PROBE = 8000          # 探针用的细胞数（取子集够看趋势、更快）
SEED = 0


def _save_z(tag, z_real, z_perm):
    """把一小段 z 存进共享 npz（供画图），保留已有其它模型的键。"""
    path = "phase6_invariance_z.npz"
    data = {}
    try:
        with np.load(path) as f:
            data = {k: f[k] for k in f.files}
    except FileNotFoundError:
        pass
    k = min(2000, z_real.shape[0])
    data[f"{tag}_real"] = z_real[:k]
    data[f"{tag}_perm"] = z_perm[:k]
    np.savez(path, **data)


def probe_scatlasvae():
    import scatlasvae
    adata = sc.read_h5ad(PROC_PATH)
    # 必须在**全量** adata 上建模型才能装载预训练权重（batch 类别数需匹配 n_batch=45）。
    model = scatlasvae.model.scAtlasVAE(
        adata=adata, batch_key=BATCH_KEY, label_key=LABEL_KEY,
        batch_embedding="embedding", batch_hidden_dim=10, device="cuda:0",
        pretrained_state_dict="scatlasvae_tcell.pt",
    )
    model.eval()
    n = min(N_PROBE, len(model._dataset))
    loader = model.as_dataloader(subset_indices=list(range(n)), shuffle=False, n_per_batch=128)

    d_perm, d_none = [], []
    zr_all, zp_all = [], []
    g = torch.Generator(device="cpu").manual_seed(SEED)
    with torch.no_grad():
        for x in loader:
            X, P, bi, li, ali, abi, lib = model._prepare_batch(x)
            qmu_real = model.encode(X, bi)["q_mu"]
            if bi is not None:
                perm = bi[torch.randperm(bi.shape[0], generator=g).to(bi.device)]
            else:
                perm = None
            qmu_perm = model.encode(X, perm)["q_mu"]
            qmu_none = model.encode(X, None)["q_mu"]
            d_perm.append((qmu_real - qmu_perm).abs().max().item())
            d_none.append((qmu_real - qmu_none).abs().max().item())
            zr_all.append(qmu_real.cpu().numpy())
            zp_all.append(qmu_perm.cpu().numpy())

    zr = np.concatenate(zr_all); zp = np.concatenate(zp_all)
    max_abs_perm = float(np.max(d_perm))
    max_abs_none = float(np.max(d_none))
    mean_l2_perm = float(np.mean(np.linalg.norm(zr - zp, axis=1)))
    row = pd.DataFrame([{
        "model": "scAtlasVAE",
        "encoder": "batch-invariant F(X)",
        "n_cells_probed": zr.shape[0],
        "max_abs_dz_perm_batch": max_abs_perm,   # 打乱 batch 后 q_mu 的最大逐元素改变
        "max_abs_dz_none_batch": max_abs_none,   # 用 None batch 后的最大改变
        "mean_l2_drift_perm": mean_l2_perm,      # 打乱 batch 后 z 的平均 L2 漂移
    }])
    row.to_csv("phase6_invariance_scatlasvae.csv", index=False)
    _save_z("scAtlasVAE", zr, zp)
    print(row.to_string(index=False))
    print(f"\n[结论] scAtlasVAE 编码器无视 batch：打乱 batch 后 max|Δq_mu| = {max_abs_perm:.3e}"
          f"（≈0 即坐实 batch-invariant）。")


def _probe_scvi_model(model, n, tag, g):
    """给一个已训练的 scVI 模型做打乱-batch 探针，返回(row_dict, zr, zp)。"""
    from scvi import REGISTRY_KEYS
    scdl = model._make_data_loader(adata=model.adata, indices=list(range(n)), batch_size=128)
    mod = model.module
    mod.eval()

    def _z_of(inf):
        if "qz" in inf and hasattr(inf["qz"], "loc"):
            return inf["qz"].loc
        return inf.get("qz_m", inf["z"])

    zr_all, zp_all = [], []
    with torch.no_grad():
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            b = tensors[REGISTRY_KEYS.BATCH_KEY]
            zr = _z_of(mod.inference(x, b))
            bperm = b[torch.randperm(b.shape[0], generator=g)]
            zp = _z_of(mod.inference(x, bperm))
            zr_all.append(zr.cpu().numpy()); zp_all.append(zp.cpu().numpy())
    zr = np.concatenate(zr_all); zp = np.concatenate(zp_all)
    row = {
        "model": tag,
        "n_cells_probed": zr.shape[0],
        "max_abs_dz_perm_batch": float(np.max(np.abs(zr - zp))),
        "max_abs_dz_none_batch": np.nan,
        "mean_l2_drift_perm": float(np.mean(np.linalg.norm(zr - zp, axis=1))),
    }
    return row, zr, zp


def probe_scvi():
    import scvi
    adata = sc.read_h5ad(PROC_PATH)
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=BATCH_KEY)
    n = min(N_PROBE, adata.n_obs)
    g = torch.Generator(device="cpu").manual_seed(SEED)

    rows = []
    # (a) 默认 scVI：encode_covariates=False -> 编码器**不吃 batch**（重要细节：
    #     论文表把 scVI 编码器记作 F(X,B,S) 是"一般形式"，scvi-tools 默认并不把 batch 编码进去）。
    m0 = scvi.model.SCVI(adata)
    m0.train(max_epochs=10)
    m0.encoder_desc = "default(encode_covariates=False)"
    r0, zr0, zp0 = _probe_scvi_model(m0, n, "scVI (默认,不编码batch)", g)
    r0["encoder"] = "F(X)  编码器不吃 batch"
    rows.append(r0); _save_z("scVI_default", zr0, zp0)

    # (b) encode_covariates=True -> 编码器**显式吃 batch** = 真正 batch-variant F(X,B)。
    #     这是 scAtlasVAE 刻意避免的架构；此时打乱 batch 才会让 z 漂移。
    m1 = scvi.model.SCVI(adata, encode_covariates=True)
    m1.train(max_epochs=10)
    r1, zr1, zp1 = _probe_scvi_model(m1, n, "scVI (编码batch)", g)
    r1["encoder"] = "F(X,B)  编码器吃 batch"
    rows.append(r1); _save_z("scVI_enccov", zr1, zp1)

    df = pd.DataFrame(rows)[["model", "encoder", "n_cells_probed",
                             "max_abs_dz_perm_batch", "max_abs_dz_none_batch", "mean_l2_drift_perm"]]
    df.to_csv("phase6_invariance_scvi.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n[结论] scVI 默认编码器也不吃 batch(漂移≈{rows[0]['mean_l2_drift_perm']:.3f})；"
          f"仅当 encode_covariates=True 显式编码 batch 时 z 才漂移"
          f"({rows[1]['mean_l2_drift_perm']:.3f})。scAtlasVAE 则是**结构上永不**编码 batch。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["scatlasvae", "scvi"], required=True)
    args = ap.parse_args()
    probe_scatlasvae() if args.model == "scatlasvae" else probe_scvi()
