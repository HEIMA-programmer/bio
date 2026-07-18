"""公平 kNN 对照（复查 E1 里 'kNN on scVI latent' 是否因 transductive 而占便宜）。

背景：transfer 里的 kNN 基线用的 `obsm['X_scVI']` 是**在全量 104,805 细胞上训练**的
（`phase2_baseline_scvi.py`），即 query 细胞被 scVI 见过 —— transductive；而 scAtlasVAE
zero-shot 只在 reference 上训模型、query 从没被见过 —— inductive。二者不同起跑线。

本脚本对每个设计各算两版 kNN，隔离这个差异：
  - transductive：直接用全量 X_scVI（复算，应与 transfer CSV 的 kNN 行一致，作自检）。
  - fair(inductive)：scVI **只在 reference 上训练**，query counts 直接过**同一个
    batch-invariant 编码器**投影（encode_covariates=False，不训练 query、不看 query 标签）
    —— 与 scAtlasVAE zero-shot 同为 inductive、真正同起跑线。

切分与 SEED 与 phase5_annotation_transfer.py 完全一致（query 集相同）。
在 `scvi` 环境跑：python phase5_fair_knn.py
产出：phase5_fair_knn_results.csv
"""
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from scipy import sparse
from scvi import REGISTRY_KEYS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROC = "tcell_processed.h5ad"
SEED = 0
K = 13
LABEL = "cell_type"
BATCH = "patient"
DROP_CANCER = "UCEC"
INFERENCE_BATCH_SIZE = 128


def select_patient_holdout(adata):
    """Match transfer design P: broadest label coverage, then most cells."""
    table = (
        adata.obs.groupby(BATCH, observed=True)
        .agg(n_cells=(LABEL, "size"), n_labels=(LABEL, "nunique"))
        .reset_index()
    )
    table[BATCH] = table[BATCH].astype(str)
    table = table.sort_values(
        ["n_labels", "n_cells", BATCH], ascending=[False, False, True]
    ).reset_index(drop=True)
    if table.empty:
        raise ValueError("No patient is available for design P")
    return str(table.loc[0, BATCH])


def macro_ovr_auc(yt_idx, probs):
    aucs = []
    for c in range(probs.shape[1]):
        yc = (yt_idx == c).astype(int)
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        aucs.append(roc_auc_score(yc, probs[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def knn_eval(zref, yref, zq, yq):
    knn = KNeighborsClassifier(n_neighbors=K).fit(zref, yref)
    yp = knn.predict(zq)
    probs = knn.predict_proba(zq)
    classes = list(knn.classes_)
    acc = accuracy_score(yq, yp)
    f1 = f1_score(yq, yp, average="macro", zero_division=0)
    cidx = {c: i for i, c in enumerate(classes)}
    yti = np.array([cidx.get(y, -1) for y in yq])
    keep = yti >= 0
    auc = macro_ovr_auc(yti[keep], probs[keep]) if keep.sum() > 0 else float("nan")
    return acc, f1, auc


def encoder_dataloader(adata, layer="counts", batch_size=INFERENCE_BATCH_SIZE):
    """为 scVI encoder 构造只含 counts 和占位 batch 的确定性数据流。

    scvi-tools 1.3.3 的 ``get_latent_representation`` 接受自定义 dataloader。
    本实验显式使用 ``encode_covariates=False``，因此 batch tensor 不进入 encoder；
    这里的全零 batch 仅用于满足 VAE inference 接口，不表示把 query 病人拟合为
    reference 中的某位病人。
    """
    x = adata.layers[layer] if layer is not None else adata.X
    for start in range(0, adata.n_obs, batch_size):
        xb = x[start : start + batch_size]
        if sparse.issparse(xb):
            xb = xb.toarray()
        xb = np.asarray(xb, dtype=np.float32)
        yield {
            REGISTRY_KEYS.X_KEY: torch.from_numpy(xb),
            REGISTRY_KEYS.BATCH_KEY: torch.zeros((xb.shape[0], 1), dtype=torch.int64),
        }


def project_query_inductive(ref, q):
    """训练 reference-only scVI，并用冻结的 reference encoder 直接投影 query。"""
    if not ref.var_names.equals(q.var_names):
        raise ValueError("reference/query 的基因及顺序必须完全一致，不能直接 encoder 投影")

    scvi.model.SCVI.setup_anndata(ref, layer="counts", batch_key=BATCH)
    rm = scvi.model.SCVI(ref, encode_covariates=False)
    rm.train(max_epochs=10)
    zref = rm.get_latent_representation()

    if rm.module.encode_covariates:
        raise RuntimeError("该基线要求 encode_covariates=False；否则 query batch 会影响 encoder")

    # 直接调用训练完成的 reference encoder。没有 query model、parameter surgery 或 query 训练。
    zq = rm.get_latent_representation(
        dataloader=encoder_dataloader(q, layer="counts")
    )
    mode = "reference-encoder-direct(no-query-training)"
    return zref, zq, mode


def main():
    adata = sc.read_h5ad(PROC)
    yall = adata.obs[LABEL].astype(str).values
    rows = []
    for design in ["A", "B", "P"]:
        rng = np.random.default_rng(SEED)
        n = adata.n_obs
        if design == "A":
            qmask = np.zeros(n, dtype=bool)
            qi = rng.choice(n, size=int(round(0.05 * n)), replace=False)
            qmask[qi] = True
            query_unit = "random 5% cells"
        elif design == "B":
            qmask = (adata.obs["cancerType"].astype(str).values == DROP_CANCER)
            query_unit = f"cancer:{DROP_CANCER}"
        else:
            patient = select_patient_holdout(adata)
            qmask = (adata.obs[BATCH].astype(str).values == patient)
            query_unit = f"patient:{patient}"
        ref_mask = ~qmask
        print(f"\n=== 设计 {design}: ref n={int(ref_mask.sum())} / query n={int(qmask.sum())} ===")

        # (1) transductive（全量 X_scVI 复算；应与 transfer CSV 的 kNN 行一致）
        Z = adata.obsm["X_scVI"]
        acc_t, f1_t, auc_t = knn_eval(Z[ref_mask], yall[ref_mask], Z[qmask], yall[qmask])
        print(f"  transductive kNN (full-data X_scVI): acc={acc_t:.3f} f1={f1_t:.3f} AUROC={auc_t:.3f}")
        rows.append(dict(design=design, query_unit=query_unit, kind="transductive(full-data scVI)",
                         accuracy=round(acc_t, 4), macro_f1=round(f1_t, 4), macro_ovr_auc=round(auc_t, 4)))
        pd.DataFrame(rows).to_csv("phase5_fair_knn_results.csv", index=False)

        # (2) fair inductive（reference-only scVI + 归纳投影 query）
        ref = adata[ref_mask].copy()
        q = adata[qmask].copy()
        zref, zq, mode = project_query_inductive(ref, q)
        acc_f, f1_f, auc_f = knn_eval(zref, yall[ref_mask], zq, yall[qmask])
        print(f"  FAIR inductive kNN ({mode}): acc={acc_f:.3f} f1={f1_f:.3f} AUROC={auc_f:.3f}")
        rows.append(dict(design=design, query_unit=query_unit, kind=f"fair-inductive({mode})",
                         accuracy=round(acc_f, 4), macro_f1=round(f1_f, 4), macro_ovr_auc=round(auc_f, 4)))
        pd.DataFrame(rows).to_csv("phase5_fair_knn_results.csv", index=False)

    print("\n===== 汇总 =====")
    print(pd.DataFrame(rows).to_string(index=False))
    print("完成：phase5_fair_knn_results.csv")


if __name__ == "__main__":
    main()
