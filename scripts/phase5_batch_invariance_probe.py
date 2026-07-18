"""阶段五 · E3：批不变编码器的"实证探针"。

论点
    scAtlasVAE 的招牌是**编码器 batch-invariant**——`_gex_model.py` 的 `encode()` 把
    "batch 拼进编码器输入"的那行**注释掉了**（行号会随本地修复变化），所以在固定 X 时
    batch 元数据不会直接改变 encoder 输出；这不等于 X 中不存在批次信号。
    这为"查询数据不重训直接映射进参考图谱(zero-shot)"提供结构基础。scVI 的 encoder
    是否接收 batch 可配置，因此本探针同时测默认 ``encode_covariates=False`` 与显式开启版本。

做法（低层直接给编码器喂不同的 batch 索引，最干净的证明）
    从全对象可复现地抽取细胞并覆盖全部 patient；随后在全部探针细胞上构造保持 batch
    边际计数、且每个细胞都换 batch 的全局置换。分别用 **真实 batch / 全局置换 batch /
    全 None** 过编码器，比较潜均值：
    scAtlasVAE 探针显式关闭其内部 dataset shuffle，确保保存的 AnnData obs 索引与实际输入行一致。
      - scAtlasVAE：q_mu(real) 与 q_mu(perm)、q_mu(None) 应**逐元素完全相同**（max|Δ|≈0）。
      - scVI 默认：z(real) 与 z(perm) 应相同；显式 ``encode_covariates=True`` 时应明显漂移。
    "我读到那行被注释" -> 升级成 "我测出来了"。

用法
    conda activate scatlasvae && python phase5_batch_invariance_probe.py --model scatlasvae
    conda activate scvi       && python phase5_batch_invariance_probe.py --model scvi
产出
    phase5_invariance_scatlasvae.csv（1 行）/ phase5_invariance_scvi.csv（默认与编码 batch 共 2 行）
    phase5_invariance_z.npz（各模型全部探针细胞的 real/perm z、索引与 batch，供复核）
对应报告
    reports/phase5_deeper_validation.md（E3）。
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"
LABEL_KEY = "cell_type"
N_PROBE = 8000          # 探针用的细胞数（取子集够看趋势、更快）
SEED = 0


def _probe_indices(adata, n_probe=N_PROBE, seed=SEED):
    """从全对象做近似均衡分层抽样，覆盖每个 patient。"""
    n = min(n_probe, adata.n_obs)
    batches = np.asarray(adata.obs[BATCH_KEY].astype(str))
    categories = np.unique(batches)
    if n < len(categories):
        raise ValueError(f"n_probe={n} 小于 batch 数 {len(categories)}，无法覆盖全部 batch")
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(batches == category) for category in categories]
    counts = np.asarray([len(pool) for pool in pools], dtype=np.int64)
    allocation = np.minimum(counts, n // len(categories))
    remaining = int(n - allocation.sum())
    # 逐轮把剩余额度均匀发给仍有容量的 patient，避免大 patient 支配均值。
    while remaining:
        eligible = np.flatnonzero(allocation < counts)
        if len(eligible) == 0:
            raise AssertionError("分层抽样容量不足")
        chosen = rng.permutation(eligible)[:min(remaining, len(eligible))]
        allocation[chosen] += 1
        remaining -= len(chosen)
    indices = np.concatenate([
        rng.choice(pool, size=int(take), replace=False)
        for pool, take in zip(pools, allocation)
    ]).astype(np.int64, copy=False)
    rng.shuffle(indices)
    return indices


def _indices_digest(indices):
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _derange_batches(batch_codes):
    """保持 batch 多重集不变，同时令每个探针细胞都换到另一个 batch。"""
    real = np.asarray(batch_codes).reshape(-1)
    _, counts = np.unique(real, return_counts=True)
    max_count = int(counts.max())
    if max_count * 2 > len(real):
        raise ValueError(
            "最大 batch 超过探针细胞的一半，无法在保持边际计数时构造完全错排"
        )

    order = np.argsort(real, kind="stable")
    sorted_codes = real[order]
    permuted_sorted = np.roll(sorted_codes, max_count)
    permuted = np.empty_like(real)
    permuted[order] = permuted_sorted
    if not np.array_equal(np.sort(permuted), np.sort(real)):
        raise AssertionError("全局 batch 置换没有保持边际计数")
    if np.any(permuted == real):
        raise AssertionError("全局 batch 置换仍含未改变的细胞")
    return permuted


def _canonical_probe_batches(adata, probe_indices):
    """按患者名称生成跨模型共享的规范 code 与全局错排。"""
    labels = np.asarray(
        adata.obs.iloc[np.asarray(probe_indices)][BATCH_KEY].astype(str)
    )
    categories = np.unique(labels)
    real = np.searchsorted(categories, labels).astype(np.int64, copy=False)
    return real, _derange_batches(real)


def _translate_canonical_permutation(model_real, canonical_real, canonical_perm):
    """把共享的患者级错排翻译成当前模型注册表使用的内部整数 code。"""
    model_real = np.asarray(model_real).reshape(-1)
    canonical_real = np.asarray(canonical_real).reshape(-1)
    canonical_perm = np.asarray(canonical_perm).reshape(-1)
    if not (model_real.shape == canonical_real.shape == canonical_perm.shape):
        raise AssertionError("模型 batch code 与规范错排行数不一致")

    mapping = {}
    reverse = {}
    for canonical_code, model_code in zip(canonical_real, model_real):
        canonical_code = int(canonical_code)
        model_code = int(model_code)
        if canonical_code in mapping and mapping[canonical_code] != model_code:
            raise AssertionError("同一患者在模型注册表中对应多个 batch code")
        if model_code in reverse and reverse[model_code] != canonical_code:
            raise AssertionError("模型 batch code 没有一一对应到患者")
        mapping[canonical_code] = model_code
        reverse[model_code] = canonical_code
    if len(mapping) != np.unique(canonical_real).size:
        raise AssertionError("探针没有建立完整的患者到模型 batch code 映射")

    translated = np.asarray(
        [mapping[int(code)] for code in canonical_perm], dtype=model_real.dtype
    )
    if not np.array_equal(np.sort(translated), np.sort(model_real)):
        raise AssertionError("翻译后的模型 batch code 没有保持边际计数")
    if np.any(translated == model_real):
        raise AssertionError("翻译后的模型 batch code 仍含未改变细胞")
    return translated


def _save_z_groups(groups):
    """一次原子更新一个或多个完整模型组，避免留下新旧 latent 混合组。"""
    path = Path("phase5_invariance_z.npz")
    data = {}
    try:
        with np.load(path) as f:
            data = {k: f[k] for k in f.files}
    except FileNotFoundError:
        pass
    # 清除旧脚本留下的含糊键；当前规范统一使用明确的模型 tag。
    for legacy_key in ("scVI_real", "scVI_perm"):
        data.pop(legacy_key, None)
    for tag, payload in groups.items():
        z_real, z_perm, obs_indices, batch_real, batch_perm = payload
        for suffix in ("real", "perm", "obs_indices", "batch_real", "batch_perm"):
            data.pop(f"{tag}_{suffix}", None)
        data[f"{tag}_real"] = np.asarray(z_real, dtype=np.float32)
        data[f"{tag}_perm"] = np.asarray(z_perm, dtype=np.float32)
        data[f"{tag}_obs_indices"] = np.asarray(obs_indices, dtype=np.int64)
        data[f"{tag}_batch_real"] = np.asarray(batch_real, dtype=np.int64)
        data[f"{tag}_batch_perm"] = np.asarray(batch_perm, dtype=np.int64)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    np.savez_compressed(tmp_path, **data)
    os.replace(tmp_path, path)


def _atomic_to_csv(frame, path):
    path = Path(path)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _scatlas_batch_codes(model, probe_indices):
    """按实际 dataloader 顺序读取 scAtlasVAE 的 batch code。"""
    loader = model.as_dataloader(
        subset_indices=probe_indices.tolist(), shuffle=False, n_per_batch=128
    )
    codes = []
    for x in loader:
        _, _, batch_index, _, _, _, _ = model._prepare_batch(x)
        if batch_index is None:
            raise ValueError("scAtlasVAE 探针没有 batch_index")
        codes.append(batch_index.detach().cpu().numpy().reshape(-1))
    return np.concatenate(codes)


def probe_scatlasvae():
    import scatlasvae
    adata = sc.read_h5ad(PROC_PATH)
    # 必须在**全量** adata 上建模型才能装载预训练权重（batch 类别数需匹配 n_batch=45）。
    model = scatlasvae.model.scAtlasVAE(
        adata=adata, batch_key=BATCH_KEY, label_key=LABEL_KEY,
        batch_embedding="embedding", batch_hidden_dim=10, device="cuda:0",
        pretrained_state_dict="scatlasvae_tcell.pt",
        _shuffle_dataset=False,
    )
    model.eval()
    probe_indices = _probe_indices(adata)
    canonical_real, canonical_perm = _canonical_probe_batches(adata, probe_indices)
    model_real_batch = _scatlas_batch_codes(model, probe_indices)
    model_perm_batch = _translate_canonical_permutation(
        model_real_batch, canonical_real, canonical_perm
    )
    loader = model.as_dataloader(
        subset_indices=probe_indices.tolist(), shuffle=False, n_per_batch=128
    )

    d_perm, d_none = [], []
    zr_all, zp_all = [], []
    offset = 0
    with torch.no_grad():
        for x in loader:
            X, P, bi, li, ali, abi, lib = model._prepare_batch(x)
            qmu_real = model.encode(X, bi)["q_mu"]
            size = bi.shape[0]
            perm = torch.as_tensor(
                model_perm_batch[offset:offset + size], dtype=bi.dtype, device=bi.device
            ).reshape_as(bi)
            offset += size
            qmu_perm = model.encode(X, perm)["q_mu"]
            qmu_none = model.encode(X, None)["q_mu"]
            d_perm.append((qmu_real - qmu_perm).abs().max().item())
            d_none.append((qmu_real - qmu_none).abs().max().item())
            zr_all.append(qmu_real.cpu().numpy())
            zp_all.append(qmu_perm.cpu().numpy())

    if offset != len(probe_indices):
        raise AssertionError(f"探针行数错位：读取 {offset}，预期 {len(probe_indices)}")
    zr = np.concatenate(zr_all); zp = np.concatenate(zp_all)
    max_abs_perm = float(np.max(d_perm))
    max_abs_none = float(np.max(d_none))
    mean_l2_perm = float(np.mean(np.linalg.norm(zr - zp, axis=1)))
    row = pd.DataFrame([{
        "model": "scAtlasVAE",
        "encoder": "batch-invariant F(X)",
        "n_cells_probed": zr.shape[0],
        "n_batches_probed": int(np.unique(canonical_real).size),
        "n_batch_changed": int(np.sum(canonical_real != canonical_perm)),
        "batch_changed_fraction": float(np.mean(canonical_real != canonical_perm)),
        "probe_seed": SEED,
        "probe_indices_sha256": _indices_digest(probe_indices),
        "max_abs_dz_perm_batch": max_abs_perm,   # 打乱 batch 后 q_mu 的最大逐元素改变
        "max_abs_dz_none_batch": max_abs_none,   # 用 None batch 后的最大改变
        "mean_l2_drift_perm": mean_l2_perm,      # 打乱 batch 后 z 的平均 L2 漂移
    }])
    _save_z_groups({
        "scAtlasVAE": (
            zr, zp, probe_indices, canonical_real, canonical_perm
        )
    })
    _atomic_to_csv(row, "phase5_invariance_scatlasvae.csv")
    print(row.to_string(index=False))
    print(f"\n[结论] scAtlasVAE 编码器无视 batch：打乱 batch 后 max|Δq_mu| = {max_abs_perm:.3e}"
          f"（≈0 即坐实 batch-invariant）。")


def _probe_scvi_model(model, probe_indices, canonical_real, canonical_perm, tag):
    """给一个已训练的 scVI 模型做打乱-batch 探针，返回(row_dict, zr, zp)。"""
    from scvi import REGISTRY_KEYS
    indices = probe_indices.tolist()
    scdl = model._make_data_loader(adata=model.adata, indices=indices, batch_size=128)
    mod = model.module
    mod.eval()

    def _z_of(inf):
        if "qz" in inf and hasattr(inf["qz"], "loc"):
            return inf["qz"].loc
        return inf.get("qz_m", inf["z"])

    real_batches = []
    for tensors in scdl:
        real_batches.append(
            tensors[REGISTRY_KEYS.BATCH_KEY].detach().cpu().numpy().reshape(-1)
        )
    model_real_batch = np.concatenate(real_batches)
    model_perm_batch = _translate_canonical_permutation(
        model_real_batch, canonical_real, canonical_perm
    )

    scdl = model._make_data_loader(adata=model.adata, indices=indices, batch_size=128)
    zr_all, zp_all = [], []
    offset = 0
    with torch.no_grad():
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            b = tensors[REGISTRY_KEYS.BATCH_KEY]
            zr = _z_of(mod.inference(x, b))
            size = b.shape[0]
            bperm = torch.as_tensor(
                model_perm_batch[offset:offset + size], dtype=b.dtype, device=b.device
            ).reshape_as(b)
            offset += size
            zp = _z_of(mod.inference(x, bperm))
            zr_all.append(zr.cpu().numpy()); zp_all.append(zp.cpu().numpy())
    if offset != len(probe_indices):
        raise AssertionError(f"探针行数错位：读取 {offset}，预期 {len(probe_indices)}")
    zr = np.concatenate(zr_all); zp = np.concatenate(zp_all)
    row = {
        "model": tag,
        "n_cells_probed": zr.shape[0],
        "n_batches_probed": int(np.unique(canonical_real).size),
        "n_batch_changed": int(np.sum(canonical_real != canonical_perm)),
        "batch_changed_fraction": float(np.mean(canonical_real != canonical_perm)),
        "probe_seed": SEED,
        "probe_indices_sha256": _indices_digest(probe_indices),
        "max_abs_dz_perm_batch": float(np.max(np.abs(zr - zp))),
        "max_abs_dz_none_batch": np.nan,
        "mean_l2_drift_perm": float(np.mean(np.linalg.norm(zr - zp, axis=1))),
    }
    return row, zr, zp, canonical_real, canonical_perm


def probe_scvi():
    import scvi
    adata = sc.read_h5ad(PROC_PATH)
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=BATCH_KEY)
    probe_indices = _probe_indices(adata)
    canonical_real, canonical_perm = _canonical_probe_batches(adata, probe_indices)

    rows = []
    # (a) 默认 scVI：encode_covariates=False -> 编码器**不吃 batch**（重要细节：
    #     论文表把 scVI 编码器记作 F(X,B,S) 是"一般形式"，scvi-tools 默认并不把 batch 编码进去）。
    scvi.settings.seed = SEED
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    m0 = scvi.model.SCVI(adata)
    m0.train(max_epochs=10)
    m0.encoder_desc = "default(encode_covariates=False)"
    r0, zr0, zp0, br0, bp0 = _probe_scvi_model(
        m0, probe_indices, canonical_real, canonical_perm,
        "scVI (默认,不编码batch)"
    )
    r0["encoder"] = "F(X)  编码器不吃 batch"
    rows.append(r0)

    # (b) encode_covariates=True -> 编码器**显式吃 batch** = 真正 batch-variant F(X,B)。
    #     这是 scAtlasVAE 刻意避免的架构；此时打乱 batch 才会让 z 漂移。
    scvi.settings.seed = SEED
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    m1 = scvi.model.SCVI(adata, encode_covariates=True)
    m1.train(max_epochs=10)
    r1, zr1, zp1, br1, bp1 = _probe_scvi_model(
        m1, probe_indices, canonical_real, canonical_perm, "scVI (编码batch)"
    )
    r1["encoder"] = "F(X,B)  编码器吃 batch"
    rows.append(r1)

    df = pd.DataFrame(rows)[["model", "encoder", "n_cells_probed", "n_batches_probed",
                             "n_batch_changed", "batch_changed_fraction", "probe_seed",
                             "probe_indices_sha256",
                             "max_abs_dz_perm_batch", "max_abs_dz_none_batch", "mean_l2_drift_perm"]]
    _save_z_groups({
        "scVI_default": (zr0, zp0, probe_indices, br0, bp0),
        "scVI_enccov": (zr1, zp1, probe_indices, br1, bp1),
    })
    _atomic_to_csv(df, "phase5_invariance_scvi.csv")
    print(df.to_string(index=False))
    print(f"\n[结论] scVI 默认编码器也不吃 batch(漂移≈{rows[0]['mean_l2_drift_perm']:.3f})；"
          f"仅当 encode_covariates=True 显式编码 batch 时 z 才漂移"
          f"({rows[1]['mean_l2_drift_perm']:.3f})。scAtlasVAE 则是**结构上永不**编码 batch。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["scatlasvae", "scvi"], required=True)
    args = ap.parse_args()
    probe_scatlasvae() if args.model == "scatlasvae" else probe_scvi()
