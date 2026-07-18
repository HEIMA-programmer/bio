"""阶段五 · E1：注释迁移 / 自动标注（复现论文 Task 3 / Ext. Data Fig. 2g,h）。

论文招牌能力
    训练好带分类头的参考模型后，query 数据可**不重训直接映射进参考图谱并自动标注**
    （zero-shot），也可与参考共训（full-shot）。论文在 TCellLandscape 上给的对标数：
    drop 5% cells / drop one study 的 ROC-AUC ≈ 0.905 / 0.859（zero-shot；Supp Table 3）。

我们的复现（数据 = 我们那份 104,805 细胞的 GSE156728 全量 CD8 10X）
    三种 query 切法：
      设计 A：随机留出 5% 细胞为 query（对标 "drop 5% cells"）。
      设计 B：留出一个整癌种（默认 UCEC），测试生物 OOD；它不是 batch/study 留出。
      设计 P：留出整位患者，作为当前字段下的样本/批次域类比；仍不是 leave-one-study。

    **分类头训练协议（--protocol，2026-07 加）**：源码 `_gex_model.py:1430`
    `if epoch > max_epoch - pred_last_n_epoch:` 才把分类头损失加进总损失。
      - `paper`（默认，推荐）：`pred_last_n_epoch=10`、`max_epoch` 按 reference 大小自动计算，
        分类头只训末 10 轮。训练日程与论文 Task 3 默认一致，但 B/P 的留出单位不同，不能与
        论文 leave-one-study 数值直接对齐。
      - `fulltime`：`pred_last_n_epoch=max_epoch`、`max_epoch=150`，分类头**全程训练**。这是早期
        ~3.8 万小参考集时期的补丁（那时末 10 轮不够、zero-shot acc 仅 0.26）；全量 ~10 万后
        论文默认已够（论文自己在 110k 上用默认拿到 0.905），此模式仅保留作对照。

    每种设计：
      - reference（其余细胞）上**监督训练**新模型（不能复用见过全量的 scatlasvae_tcell.pt）。
      - zero-shot：setup_anndata(query, ref.pt) -> 预训练权重建 query 模型 -> predict_labels。
      - full-shot（仅设计 A）：query 标签置 undefined，与 reference concat 共训后 predict。
      - 主表 kNN 使用既有 full-data `X_scVI`，属于 transductive 诊断；真正 reference-only、
        query 不训练的公平版本见 `phase5_fair_knn.py`。
    指标：accuracy、macro-F1、macro one-vs-rest ROC-AUC（论文指标）+ 混淆矩阵。

用法（环境 A `scatlasvae`；sklearn 由 scanpy 依赖自带）
    python phase5_annotation_transfer.py                          # 论文协议、设计 A+B
    python phase5_annotation_transfer.py --protocol fulltime      # 旧全程训练（对照）
产出（按协议分文件，互不覆盖）
    phase5_transfer_results[_paper].csv（每行 = 设计×方法 的 acc/F1/AUROC）
    phase5_transfer_cm[_paper].npz（各配置的 y_true/y_pred，供画混淆矩阵）
对应报告
    reports/phase5_deeper_validation.md（E1）。
"""
import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import softmax

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"
LABEL_KEY = "cell_type"
DROP_CANCER = "UCEC"        # 设计 B 留作 query 的癌种
KNN_K = 13                  # 我们自设的 kNN 对照（"无专用预测头"控制；论文 Bench3 基线是 scPoli/scANVI/CellTypist，非 kNN）
SEED = 0


def auto_max_epoch(n):
    """论文默认 epoch 公式 min(round(20000/N*400),400)。"""
    return int(min(round(20000 / n * 400), 400))


def select_patient_holdout(adata):
    """Deterministically choose one whole patient for a batch-domain holdout.

    This dataset has no study identifier.  The chosen patient maximizes cell-type
    coverage, then cell count; this is an analogue, not paper-exact leave-study-out.
    """
    table = (
        adata.obs.groupby(BATCH_KEY, observed=True)
        .agg(n_cells=(LABEL_KEY, "size"), n_labels=(LABEL_KEY, "nunique"))
        .reset_index()
    )
    table[BATCH_KEY] = table[BATCH_KEY].astype(str)
    table = table.sort_values(
        ["n_labels", "n_cells", BATCH_KEY], ascending=[False, False, True]
    ).reset_index(drop=True)
    if table.empty:
        raise ValueError("No patient is available for design P")
    return str(table.loc[0, BATCH_KEY]), table


# ---------- 指标 ----------
def macro_ovr_auc(y_true_idx, probs):
    """对 query 中真实存在(有正负样本)的类别算 one-vs-rest AUROC，再取宏平均。"""
    aucs = []
    for c in range(probs.shape[1]):
        yc = (y_true_idx == c).astype(int)
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        aucs.append(roc_auc_score(yc, probs[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def eval_pred(tag, design, method, y_true_str, y_pred_str, classes, probs, cm_store):
    """统一算 acc / macro-F1 / macro-OVR-AUROC，并存混淆矩阵原料。"""
    acc = accuracy_score(y_true_str, y_pred_str)
    f1 = f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)
    auc = float("nan")
    if probs is not None:
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        yt_idx = np.array([cls_to_idx.get(y, -1) for y in y_true_str])
        keep = yt_idx >= 0
        if keep.sum() > 0:
            auc = macro_ovr_auc(yt_idx[keep], probs[keep])
    cm_store[f"{tag}_true"] = np.asarray(y_true_str, dtype=object)
    cm_store[f"{tag}_pred"] = np.asarray(y_pred_str, dtype=object)
    print(f"  [{design} / {method}] acc={acc:.3f}  macroF1={f1:.3f}  macroOVR-AUC={auc:.3f}  n={len(y_true_str)}")
    return {"design": design, "method": method, "n_query": len(y_true_str),
            "accuracy": acc, "macro_f1": f1, "macro_ovr_auc": auc}


def predictions_from_logits(model, logits):
    """由同一次前向的 logits 同时派生类别、标签和概率。"""
    logits = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    classes = [
        str(category)
        for category in model.label_category.categories
        if category != model.unlabel_key
    ]
    if logits.shape[1] != len(classes):
        raise ValueError(
            f"分类头输出 {logits.shape[1]} 列，但有效标签类别有 {len(classes)} 个"
        )
    probs = softmax(logits, axis=1)
    y_pred = np.asarray(classes, dtype=object)[np.argmax(probs, axis=1)]
    return y_pred, classes, probs


# ---------- scAtlasVAE 迁移 ----------
def train_reference(adata, ref_mask, ref_pt, max_epoch, pred_last):
    import scatlasvae
    ref = adata[ref_mask].copy()
    m = scatlasvae.model.scAtlasVAE(
        adata=ref, batch_key=BATCH_KEY, label_key=LABEL_KEY,
        batch_embedding="embedding", batch_hidden_dim=10, device="cuda:0",
    )
    # 分类头训练调度：源码 fit `if epoch > max_epoch - pred_last_n_epoch` 才加分类损失。
    #   paper 协议: pred_last=10、max_epoch 按 reference 大小自动计算 → 末 10 轮。
    #   fulltime  : pred_last=max_epoch → 全程（小参考集时期补丁，全量下非必需）。
    m.fit(max_epoch=max_epoch, pred_last_n_epoch=pred_last)
    m.save_to_disk(ref_pt)
    print(f"  参考模型已训练并保存 -> {ref_pt}"
          f"（reference n={int(ref_mask.sum())}, max_epoch={max_epoch}, pred_last_n_epoch={pred_last}）")


def zeroshot_predict(adata, query_mask, ref_pt):
    # 完全照官方 pipeline.run_transfer 的范式：setup_anndata + 用保存的 model_config 重建，
    # 保证架构与预训练权重严格对齐（分类头维度等）。
    import torch
    import scatlasvae
    q = adata[query_mask].copy()
    y_true = q.obs[LABEL_KEY].astype(str).values     # 先取真值，setup_anndata 会改写 obs[label]
    # 关键：官方 setup_anndata 对 query 的 batch/label 做 add_categories 时**假设与参考不相交**，
    # 而我们留出式 query 的 batch(病人)与 label(亚型)都是参考的子集 -> 会报 "must not include old"。
    # 解法：删掉 query 的 batch 与 label 两列，让官方走"不在 obs -> 全设 undefined"的分支：
    #   - batch：编码器 batch-invariant（E3 实测 Δz=0），取值对潜向量/预测毫无影响；
    #   - label：setup 会把 categories 设成"参考 17 类 + undefined"，n_label 由 categories 推出仍=17，
    #            与预训练分类头对齐；真值已在上面存进 y_true 供评测。
    # 这也正是诚实的 zero-shot 语义：假装不知道 query 的批次与标签。
    for col in (BATCH_KEY, LABEL_KEY):
        if col in q.obs.columns:
            del q.obs[col]
    state_dict = torch.load(ref_pt, map_location="cuda:0")
    cfg = dict(state_dict["model_config"])
    cfg.pop("device", None)
    if "new_adata_key" in cfg:                        # 兼容旧版本 config
        cfg["unlabel_key"] = cfg.pop("new_adata_key")
    scatlasvae.model.scAtlasVAE.setup_anndata(q, path_to_state_dict=ref_pt)
    qm = scatlasvae.model.scAtlasVAE(
        adata=q, pretrained_state_dict=state_dict["model_state_dict"],
        device="cuda:0", **cfg,
    )
    # 只做一次随机 latent 前向；标签、概率和 AUROC 必须来自同一份 logits。
    logits = qm.predict_labels(return_pandas=False)
    y_pred, classes, probs = predictions_from_logits(qm, logits)
    return y_true, y_pred, classes, probs


def fullshot_predict(adata, ref_mask, query_mask, max_epoch, pred_last):
    import scatlasvae
    ref = adata[ref_mask].copy()
    q = adata[query_mask].copy()
    y_true = q.obs[LABEL_KEY].astype(str).values
    q_index = list(q.obs.index)
    q.obs[LABEL_KEY] = "undefined"          # 共训时隐藏 query 标签
    merged = sc.concat([ref, q])
    m = scatlasvae.model.scAtlasVAE(
        adata=merged, batch_key=BATCH_KEY, label_key=LABEL_KEY,
        batch_embedding="embedding", batch_hidden_dim=10, device="cuda:0",
    )
    m.fit(max_epoch=max_epoch, pred_last_n_epoch=pred_last)   # 与参考同协议
    # 只做一次随机 latent 前向；随后从同一份 logits 中截取 query 并派生全部指标。
    logits_all = m.predict_labels(return_pandas=False)
    # 取 query 行的 logits（predict_labels 的行序 = self.adata.obs 行序 = merged 行序）
    pos = {idx: i for i, idx in enumerate(list(merged.obs.index))}
    qrows = [pos[i] for i in q_index]
    logits_query = logits_all[qrows]
    y_pred, classes, probs = predictions_from_logits(m, logits_query)
    return y_true, y_pred, classes, probs


def knn_baseline(adata, ref_mask, query_mask, rep="X_scVI"):
    """Transductive 诊断：在既有 full-data X_scVI 上做 reference→query kNN。"""
    Z = adata.obsm[rep]
    yt = adata.obs[LABEL_KEY].astype(str).values
    knn = KNeighborsClassifier(n_neighbors=KNN_K)
    knn.fit(Z[ref_mask], yt[ref_mask])
    y_pred = knn.predict(Z[query_mask])
    classes = list(knn.classes_)
    probs = knn.predict_proba(Z[query_mask])   # 列序 = knn.classes_
    return yt[query_mask], y_pred, classes, probs


# ---------- 主流程 ----------
def run_design(adata, design, protocol, fulltime_max_epoch, out_csv, out_npz, rows, cm_store):
    rng = np.random.default_rng(SEED)
    n = adata.n_obs
    query_definition = ""
    if design == "A":
        query_mask = np.zeros(n, dtype=bool)
        q_idx = rng.choice(n, size=int(round(0.05 * n)), replace=False)
        query_mask[q_idx] = True
        query_definition = "random 5% cells"
    elif design == "B":  # whole-cancer biological OOD extension
        query_mask = (adata.obs["cancerType"].astype(str).values == DROP_CANCER)
        if query_mask.sum() == 0:
            print(f"  [跳过] 设计 B：数据里没有癌种 {DROP_CANCER}")
            return
        query_definition = f"leave one cancer: {DROP_CANCER}"
    else:  # P: closest available analogue to leaving one batch/domain out
        patient, patient_table = select_patient_holdout(adata)
        query_mask = (adata.obs[BATCH_KEY].astype(str).values == patient)
        selected = patient_table.loc[patient_table[BATCH_KEY] == patient].copy()
        selected.insert(0, "design", "P")
        selected.insert(1, "interpretation", "leave-one-patient batch-domain analogue")
        selected.to_csv("phase5_patient_holdout_selection.csv", index=False)
        query_definition = f"leave one patient: {patient}"
    ref_mask = ~query_mask
    n_ref = int(ref_mask.sum())

    # 协议决定 max_epoch 与 pred_last_n_epoch
    if protocol == "paper":
        max_epoch = auto_max_epoch(n_ref)
        pred_last = 10                         # 论文默认
    else:  # fulltime
        max_epoch = fulltime_max_epoch         # 150
        pred_last = fulltime_max_epoch         # 全程
    print(f"\n=== 设计 {design} ({query_definition}; protocol={protocol})：reference n={n_ref} / query n={int(query_mask.sum())}"
          f" | max_epoch={max_epoch}, pred_last_n_epoch={pred_last} ===")

    ref_pt = f"ref_model_design{design}_{protocol}.pt"
    if not os.path.exists(ref_pt):
        train_reference(adata, ref_mask, ref_pt, max_epoch, pred_last)
    else:
        print(f"  参考模型已存在，跳过训练：{ref_pt}")

    # zero-shot
    yt, yp, cls, pr = zeroshot_predict(adata, query_mask, ref_pt)
    rows.append(eval_pred(f"{design}_zeroshot", design, "scAtlasVAE (zero-shot)", yt, yp, cls, pr, cm_store))

    # full-shot（仅设计 A，控时长）
    if design == "A":
        yt, yp, cls, pr = fullshot_predict(adata, ref_mask, query_mask, max_epoch, pred_last)
        rows.append(eval_pred(f"{design}_fullshot", design, "scAtlasVAE (full-shot)", yt, yp, cls, pr, cm_store))

    # 旧 full-data X_scVI 上的 transductive 诊断；公平 reference-only 版本另见 phase5_fair_knn.py。
    if "X_scVI" in adata.obsm:
        yt, yp, cls, pr = knn_baseline(adata, ref_mask, query_mask, rep="X_scVI")
        rows.append(eval_pred(f"{design}_knn_scvi", design, "kNN on scVI latent", yt, yp, cls, pr, cm_store))

    # 每个设计结束落一次盘，便于中断续跑
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    np.savez(out_npz, **cm_store)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", default=["A", "B"], choices=["A", "B", "P"],
                    help="A=random 5%%; B=leave UCEC cancer (biological OOD); "
                         "P=leave one whole patient (batch-domain analogue)")
    ap.add_argument("--protocol", choices=["paper", "fulltime"], default="paper",
                    help="paper=论文协议(pred_last_n_epoch=10、max_epoch自动，与论文benchmark同起跑线，推荐)；"
                         "fulltime=分类头全程训练(旧小参考集补丁，仅作对照)")
    ap.add_argument("--max-epoch", type=int, default=150,
                    help="仅 fulltime 协议用：全程训练的 epoch 数（控 4060 时长；默认 150）")
    args = ap.parse_args()

    # 按协议分文件，避免覆盖对照结果
    if args.designs == ["P"]:
        suffix = "_patient_fulltime" if args.protocol == "fulltime" else "_patient_paper"
    else:
        suffix = "" if args.protocol == "fulltime" else "_paper"
    out_csv = f"phase5_transfer_results{suffix}.csv"
    out_npz = f"phase5_transfer_cm{suffix}.npz"

    adata = sc.read_h5ad(PROC_PATH)
    rows, cm_store = [], {}
    for d in args.designs:
        run_design(adata, d, args.protocol, args.max_epoch, out_csv, out_npz, rows, cm_store)

    print("\n===== 汇总 =====")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n完成：见 {out_csv} 与 {out_npz}")


if __name__ == "__main__":
    main()
