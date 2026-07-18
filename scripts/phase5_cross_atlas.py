"""阶段五 · Task 2：跨图谱整合 + 标签对齐（对标论文 Ext. Data Fig. 3）。

论文的 Task 2
    把两个**各自独立注释**的 CD8 图谱联合整合，并借 scAtlasVAE 独有的**多分类头**
    把两套标签体系**并行对齐**。这是 scVI/scPoli 等做不到的（它们只整合、不对齐标签）。

我们的复现
    图谱 1 = 我们的 Zheng/GSE156728 重建对象（~10.5 万 CD8，meta.cluster 17 亚型；
             不是带 28 个 study_name 的论文成品 TCellLandscape）。
    图谱 2 = Yost 2019 BCC（GSE123813，10X，CD8 亚型 CD8_act/eff/ex/ex_act/mem）——
             真实、独立、且本身就是论文 28 studies 之一，是货真价实的跨研究/跨癌种挑战。
    做法：合并两图谱，batch=[patient, atlas]（atlas 作附加批次），
          label=[ct_zheng, ct_yost]（两个分类头，各只在本图谱有标签的细胞上训练；
          另一图谱的细胞该列置 'undefined'）。训练后：
      (1) 跨图谱整合质量：两图谱在潜空间是否混合（atlas silhouette，越低越混）。
      (2) 论文式标签对齐：主分类头和附加分类头在同一次推理中预测所有细胞，统计两套预测标签的
          共现比例，得到 Yost×Zheng 对齐矩阵（预期 CD8_ex↔Tex.*、CD8_mem↔Tm.* 等）。
      (3) latent-kNN 标签对应：保留原最近邻分析作为共享潜空间的附加诊断，并与 PCA 比较；
          它不再冒充论文的多分类头标签对齐。

用法（环境 A `scatlasvae`）
    python phase5_cross_atlas.py                 # 默认全量 Zheng + Yost CD8
    python phase5_cross_atlas.py --zheng-n 40000 # 想更快就下采样 Zheng
产出
    phase5_cross_atlas_mixing.csv     ：各嵌入的 atlas 混合度（silhouette）
    phase5_cross_atlas_head_alignment.csv
                                      ：论文式多分类头预测共现矩阵（行归一化占比）
    phase5_cross_atlas_head_alignment_counts.csv
                                      ：多分类头预测的原始共现计数
    phase5_cross_atlas_head_alignment_links.csv
                                      ：行占比至少 10% 的标签对应边
    phase5_cross_atlas_head_predictions.csv.gz
                                      ：同一次前向产生的逐细胞双头预测
    phase5_cross_atlas_latent_knn_alignment.csv
                                      ：原 latent-kNN 标签对应矩阵（仅附加诊断）
    phase5_cross_atlas_latent_knn_alignment_pca.csv
                                      ：同一 latent-kNN 诊断的未校正 PCA 对照
    phase5_cross_atlas.npz            ：X_cross / X_pca_cross / atlas / 两套标签（供画 UMAP）
对应报告
    reports/phase5_deeper_validation.md（Task 2 一节）。
"""
import argparse
import gzip
import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

RAW_ZHENG = "TCellLandscape_raw.h5ad"          # 历史兼容名；Zheng/GSE156728 8 癌种全量 raw（105k × 24148）
YOST_META = "GSE123813_bcc_tcell_metadata.txt.gz"
YOST_COUNTS = "GSE123813_bcc_scRNA_counts.txt.gz"
YOST_CD8 = ["CD8_act", "CD8_eff", "CD8_ex", "CD8_ex_act", "CD8_mem"]
N_HVG = 4000
UNDEF = "undefined"
SEED = 0


# ---------- 载入 Yost BCC CD8（流式，内存安全）----------
def load_yost_cd8():
    cache = "yost_cd8.h5ad"
    if os.path.exists(cache):                          # 缓存命中：秒载（避免每次重解析 100MB counts）
        ad = sc.read_h5ad(cache)
        print(f"  Yost CD8 (缓存): {ad.n_obs} 细胞 × {ad.n_vars} 基因；亚型 {sorted(set(ad.obs['ct_yost']))}")
        return ad
    meta = pd.read_csv(YOST_META, sep="\t").set_index("cell.id")
    meta = meta[meta["cluster"].isin(YOST_CD8)]
    keep_set = set(meta.index)
    # 该 counts 是 R write.table(rownames) 格式：表头只有 53030 个 cell 名（无基因列表头），
    # 数据行 = 基因名 + 53030 个值。故 header 全是 cell；用 index_col=0 让 pandas 把首列(基因)作索引，
    # 其余列对齐表头(cells)。分块读、每块按名取 CD8 列（保序）。
    with gzip.open(YOST_COUNTS, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    # 数据行 = 基因名 + 53030 个值：字段位置 0=基因，位置 i+1 对应 header[i](cell)。
    # 只读 基因列(0) + CD8 细胞列（positional usecols），避免读全部 5.3 万列（快 ~4 倍、省内存）。
    keep_positions = [0] + [i + 1 for i, c in enumerate(header) if c in keep_set]
    cd8_cells = [c for c in header if c in keep_set]        # 与所读列同序(file order)
    gene_names, blocks = [], []
    for chunk in pd.read_csv(YOST_COUNTS, sep="\t", header=None, skiprows=1,
                             usecols=keep_positions, chunksize=2000):
        chunk = chunk.set_index(chunk.columns[0])          # 首列=基因
        gene_names.extend(chunk.index.astype(str).tolist())
        blocks.append(csr_matrix(chunk.values.astype(np.float32)))
    mat = vstack(blocks).tocsr()                 # 基因 × CD8细胞
    gn = pd.Index(gene_names)
    dup = gn.duplicated(keep="first")
    if dup.any():
        mat = mat[np.where(~dup)[0]]; gn = gn[~dup]
    ad = sc.AnnData(X=csr_matrix(mat.T))          # 细胞 × 基因
    ad.obs_names = cd8_cells
    ad.var_names = gn.astype(str)
    m = meta.reindex(cd8_cells)
    ad.obs["patient"] = ("yost_" + m["patient"].astype(str)).values
    ad.obs["ct_yost"] = m["cluster"].astype(str).values
    ad.obs["ct_zheng"] = UNDEF                     # 另一图谱的标签列置 undefined（多头训练用）
    ad.obs["atlas"] = "Yost"
    ad.write_h5ad(cache)                            # 缓存，供下次秒载
    print(f"  Yost CD8: {ad.n_obs} 细胞 × {ad.n_vars} 基因；亚型 {sorted(set(ad.obs['ct_yost']))}")
    return ad


def load_zheng(zheng_n):
    ad = sc.read_h5ad(RAW_ZHENG)
    if zheng_n and ad.n_obs > zheng_n:                # 可选下采样（分层 patient）
        rng = np.random.default_rng(SEED); frac = zheng_n / ad.n_obs; idx = []
        for _, g in ad.obs.groupby("patient", observed=True):
            k = max(1, int(round(len(g) * frac))); idx.extend(rng.choice(g.index, min(k, len(g)), replace=False))
        ad = ad[np.array(idx)].copy()
    ad.obs["patient"] = ("zheng_" + ad.obs["patient"].astype(str)).values
    ad.obs["ct_zheng"] = ad.obs["cell_type"].astype(str).values
    ad.obs["ct_yost"] = UNDEF                       # 另一图谱标签列置 undefined
    ad.obs["atlas"] = "Zheng"
    ad.obs = ad.obs[["patient", "ct_zheng", "ct_yost", "atlas"]].copy()   # 只留需要的列，concat 干净
    print(f"  Zheng: {ad.n_obs} 细胞 × {ad.n_vars} 基因")
    return ad.copy()


def latent_knn_alignment_matrix(Z, atlas, ct_zheng, ct_yost, k=30):
    """附加诊断：在共享潜空间用 Yost→Zheng kNN 投出标签对应矩阵。

    这项分析衡量 latent 邻域中的生物标签对应，不是论文 Task 2 的多分类头
    标签对齐；后者由 :func:`classifier_head_alignment` 实现。
    """
    zmask = atlas == "Zheng"; ymask = atlas == "Yost"
    nn = NearestNeighbors(n_neighbors=k).fit(Z[zmask])
    _, idx = nn.kneighbors(Z[ymask])
    zlab = ct_zheng[zmask]
    ylab = ct_yost[ymask]
    yost_types = sorted(set(ylab)); zheng_types = sorted(set(zlab))
    M = pd.DataFrame(0.0, index=yost_types, columns=zheng_types)
    for i, yt in enumerate(ylab):
        neigh = zlab[idx[i]]
        for zt in neigh:
            M.loc[yt, zt] += 1
    M = M.div(M.sum(1), axis=0)                        # 行归一化=每个 Yost 亚型的 Zheng 邻居分布
    return M


def _to_numpy(x):
    """把 torch.Tensor/array-like 安全转成 NumPy，不保留计算图。"""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def predict_all_label_heads_once(model):
    """只前向一次，返回与 ``model.adata.obs`` 对齐的所有分类头 logits。

    scAtlasVAE 1.0.6a3 的 ``predict_labels(return_pandas=False)`` 返回
    ``(main_logits, additional_predictions)``。其中 ``main_logits`` 已恢复为
    AnnData 原顺序，但 ``additional_predictions`` 仍是
    ``list[minibatch][additional_head]``，需要逐头拼接并用
    ``model._shuffle_indices`` 恢复顺序。不能再调用一次 ``predict_labels``，
    因为每次调用都会重新采样随机 latent ``z``。
    """
    result = model.predict_labels(return_pandas=False, show_progress=True)
    if not (isinstance(result, tuple) and len(result) == 2):
        raise RuntimeError(
            "当前模型没有返回 (main_logits, additional_predictions)；"
            "请确认训练时 label_key 至少包含两个标签列。"
        )

    main_logits, additional_raw = result
    main_logits = _to_numpy(main_logits)
    n_cells = model.adata.n_obs
    n_heads = len(model.n_additional_label or [])
    if n_heads == 0:
        raise RuntimeError("没有 additional classifier head，无法执行 Task 2 标签对齐。")

    # 当前官方源码：list[minibatch][head]，尚未拼接/恢复 AnnData 顺序。
    if len(additional_raw) and isinstance(additional_raw[0], (list, tuple)):
        inverse_shuffle = np.asarray(model._shuffle_indices)
        additional_logits = [
            np.vstack([_to_numpy(batch[head]) for batch in additional_raw])[inverse_shuffle]
            for head in range(n_heads)
        ]
    # 兼容未来可能直接返回 list[head] 的实现；这种结构应已与 main 一样恢复顺序。
    elif len(additional_raw) == n_heads:
        additional_logits = [_to_numpy(x) for x in additional_raw]
    else:
        raise RuntimeError("无法识别 predict_labels 返回的 additional_predictions 结构。")

    if main_logits.shape[0] != n_cells or any(x.shape[0] != n_cells for x in additional_logits):
        raise RuntimeError("分类头 logits 行数与 AnnData 细胞数不一致。")
    return main_logits, additional_logits


def classifier_head_alignment(model, threshold=0.10):
    """按论文 Task 2，用两个分类头对所有细胞的预测共现对齐标签体系。

    返回原始共现计数、按 Yost 预测标签行归一化的 P(Zheng|Yost)、达到
    ``threshold`` 的对应边，以及每个细胞的两头预测。当前实验只有一个
    additional head（Yost）。
    """
    main_logits, additional_logits = predict_all_label_heads_once(model)
    if len(additional_logits) != 1:
        raise RuntimeError(
            f"本脚本预期恰好一个 additional head，实际得到 {len(additional_logits)} 个。"
        )

    undefined = str(getattr(model, "unlabel_key", UNDEF))
    zheng_types = [str(x) for x in model.label_category.categories if str(x) != undefined]
    yost_types = [str(x) for x in model.additional_label_category[0].categories if str(x) != undefined]
    yost_logits = additional_logits[0]
    if main_logits.shape[1] != len(zheng_types):
        raise RuntimeError(
            f"主头 logits 有 {main_logits.shape[1]} 列，但有效 Zheng 标签有 {len(zheng_types)} 个。"
        )
    if yost_logits.shape[1] != len(yost_types):
        raise RuntimeError(
            f"附加头 logits 有 {yost_logits.shape[1]} 列，但有效 Yost 标签有 {len(yost_types)} 个。"
        )

    pred_zheng = np.asarray(zheng_types, dtype=object)[main_logits.argmax(axis=1)]
    pred_yost = np.asarray(yost_types, dtype=object)[yost_logits.argmax(axis=1)]
    counts = pd.crosstab(
        pd.Categorical(pred_yost, categories=yost_types),
        pd.Categorical(pred_zheng, categories=zheng_types),
        dropna=False,
    )
    counts.index.name = "predicted_yost_label"
    counts.columns.name = "predicted_zheng_label"
    row_fraction = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    links = row_fraction.stack().rename("p_zheng_given_yost").reset_index()
    links["cooccurrence_count"] = [
        int(counts.loc[y, z])
        for y, z in zip(links["predicted_yost_label"], links["predicted_zheng_label"])
    ]
    links = links[links["p_zheng_given_yost"] >= threshold].copy()
    links["threshold"] = threshold
    links = links.sort_values(
        ["predicted_yost_label", "p_zheng_given_yost"], ascending=[True, False]
    )
    predictions = pd.DataFrame(
        {"predicted_zheng_label": pred_zheng, "predicted_yost_label": pred_yost},
        index=model.adata.obs_names,
    )
    return counts, row_fraction, links, predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zheng-n", type=int, default=0, help="Zheng 下采样(0=全量)")
    ap.add_argument("--max-epoch", type=int, default=0, help="0=自动 min(round(20000/N*400),400)")
    ap.add_argument("--batch-hidden-dim", type=int, default=10,
                    help="批嵌入维度（论文 Ext.Fig.4b 默认 64；跨图谱硬 batch 可能需要更大容量）")
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="学习率（默认 5e-5；全量多头在高 KL 权重末期会梯度爆炸致 q_mu=NaN，"
                         "调到 3e-5 可稳住——见报告 Task 2 踩坑）")
    ap.add_argument("--pred-last", type=int, default=0,
                    help="分类头训练轮数 pred_last_n_epoch：0=全程(=max_epoch，旧行为)；"
                         "10=官方源码默认(末10轮；论文Methods未明确披露)。用于检验结果对分类头日程的敏感性。")
    args = ap.parse_args()

    print("载入两个图谱...")
    yost = load_yost_cd8()
    zheng = load_zheng(args.zheng_n)

    # 共享基因
    common = sorted(set(zheng.var_names) & set(yost.var_names))
    print(f"共享基因 {len(common)} 个")
    zheng = zheng[:, common].copy(); yost = yost[:, common].copy()

    # 合并 + 两套标签列（另一图谱置 undefined）
    merged = sc.concat([zheng, yost], join="inner", index_unique="-")
    # 两个 adata 都已带 ct_zheng/ct_yost（本图谱真值 + 另一图谱 undefined），concat 后直接转 categorical
    for col in ("ct_zheng", "ct_yost", "atlas", "patient"):
        merged.obs[col] = pd.Categorical(merged.obs[col].astype(str).values)
    print(f"合并：{merged.shape}  atlas 分布={dict(pd.Series(merged.obs['atlas']).value_counts())}")

    # 兜底非零 + int32 raw
    merged = merged[np.asarray(merged.X.sum(1)).ravel() > 0].copy()
    merged.layers["counts"] = csr_matrix(merged.X, dtype=np.int32)

    # HVG（在合并数据上，供 PCA 基线；scAtlasVAE 直接吃 counts 全基因也行，这里裁到 HVG 提速）
    tmp = merged.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=N_HVG, batch_key="atlas")
    hvg = tmp.var_names[tmp.var["highly_variable"]].tolist()
    merged = merged[:, hvg].copy()                    # 子集 layers['counts'] 也随之裁到 HVG
    merged.X = csr_matrix(merged.layers["counts"], dtype=np.int32)   # scAtlasVAE 吃原始计数

    # PCA 基线（未校正）
    tmp2 = merged.copy(); sc.pp.normalize_total(tmp2, target_sum=1e4); sc.pp.log1p(tmp2)
    sc.pp.scale(tmp2, max_value=10); sc.tl.pca(tmp2, n_comps=50)
    merged.obsm["X_pca_cross"] = tmp2.obsm["X_pca"][:, :50]
    del tmp, tmp2

    # 训练 scAtlasVAE 多头（batch=[patient,atlas]，label=[ct_zheng,ct_yost]）
    import scatlasvae
    N = merged.n_obs
    max_epoch = args.max_epoch or int(min(round(20000 / N * 400), 400))
    pred_last = args.pred_last if args.pred_last > 0 else max_epoch
    tag = "" if args.pred_last <= 0 else f"_pl{pred_last}"   # 输出文件后缀，避免覆盖全程版结果
    print(f"训练 scAtlasVAE 跨图谱多头：N={N}, max_epoch={max_epoch}, pred_last_n_epoch={pred_last}")
    m = scatlasvae.model.scAtlasVAE(
        adata=merged,
        batch_key=["patient", "atlas"],
        label_key=["ct_zheng", "ct_yost"],
        batch_embedding="embedding", batch_hidden_dim=args.batch_hidden_dim, device="cuda:0",
    )
    m.fit(max_epoch=max_epoch, pred_last_n_epoch=pred_last, lr=args.lr)
    merged.obsm["X_cross"] = m.get_latent_embedding()

    # ---- 评估 ----
    atlas = merged.obs["atlas"].astype(str).values
    ctz = merged.obs["ct_zheng"].astype(str).values
    cty = merged.obs["ct_yost"].astype(str).values

    # (1) atlas 混合度：用两个**对类别不平衡稳健**的指标。
    #   ⚠️ 直接对全体做 silhouette 会被多数图谱主导而误导：Yost 仅占 ~10%，
    #      原始 silhouette 甚至会让 scAtlasVAE 看着比 PCA 差（其实是不平衡假象）。故改用：
    #   a. **平衡子采样**的 atlas silhouette（等量 Yost/Zheng，越低=越混）
    #   b. **Yost 细胞 30-NN 中 Zheng 占比**（越高=两图谱越混；理想≈Zheng 全局占比）——最稳健
    rng = np.random.default_rng(SEED)
    zi = np.where(atlas == "Zheng")[0]; yi = np.where(atlas == "Yost")[0]
    kbal = min(len(yi), 3000)
    bal = np.concatenate([rng.choice(zi, kbal, replace=False), rng.choice(yi, kbal, replace=False)])
    ideal = float((atlas == "Zheng").mean())
    rows = []
    for name, key in [("scAtlasVAE 跨图谱", "X_cross"), ("PCA(未校正)", "X_pca_cross")]:
        X = merged.obsm[key]
        sil_bal = silhouette_score(X[bal], atlas[bal])
        nn = NearestNeighbors(n_neighbors=31).fit(X); _, idx = nn.kneighbors(X[yi])
        frac_zheng = float((atlas[idx[:, 1:]] == "Zheng").mean())
        rows.append({"embedding": name,
                     "atlas_silhouette_balanced_lower_is_more_mixed": round(float(sil_bal), 4),
                     "yost_nn_frac_zheng_higher_is_more_mixed": round(frac_zheng, 4)})
        print(f"  [{name}] balanced silhouette={sil_bal:.4f}（越低越混） | "
              f"Yost 30-NN 中 Zheng={frac_zheng:.3f}（理想≈{ideal:.2f}）")
    pd.DataFrame(rows).to_csv(f"phase5_cross_atlas_mixing{tag}.csv", index=False)

    # (2) 论文 Task 2：两个分类头在同一次随机 z 前向中预测所有细胞，再统计标签共现。
    # 不可分别调用 predict_labels 获取两头结果，否则会使用两次不同的随机 latent。
    head_counts, head_alignment, head_links, head_predictions = classifier_head_alignment(
        m, threshold=0.10
    )
    head_counts.to_csv(f"phase5_cross_atlas_head_alignment_counts{tag}.csv")
    head_alignment.to_csv(f"phase5_cross_atlas_head_alignment{tag}.csv")
    head_links.to_csv(f"phase5_cross_atlas_head_alignment_links{tag}.csv", index=False)
    head_predictions.to_csv(f"phase5_cross_atlas_head_predictions{tag}.csv.gz", compression="gzip")
    print("\n论文式多分类头 Yost×Zheng 对齐（P(Zheng头标签 | Yost头标签) top3）：")
    for yt in head_alignment.index:
        top = head_alignment.loc[yt].sort_values(ascending=False).head(3)
        print(f"  {yt:12s} -> " + ", ".join(f"{z}({p:.2f})" for z, p in top.items()))

    # (3) 原 latent-kNN 分析保留为附加诊断，并用明确文件名避免冒充多头标签对齐。
    knn_alignment = latent_knn_alignment_matrix(
        merged.obsm["X_cross"], atlas, ctz, cty, k=30
    )
    knn_alignment.to_csv(f"phase5_cross_atlas_latent_knn_alignment{tag}.csv")
    pca_knn_alignment = latent_knn_alignment_matrix(
        merged.obsm["X_pca_cross"], atlas, ctz, cty, k=30
    )
    pca_knn_alignment.to_csv(
        f"phase5_cross_atlas_latent_knn_alignment_pca{tag}.csv"
    )
    print("\n附加诊断：latent-kNN Yost真值×Zheng真值标签对应 top3：")
    for yt in knn_alignment.index:
        top = knn_alignment.loc[yt].sort_values(ascending=False).head(3)
        print(f"  {yt:12s} -> " + ", ".join(f"{z}({p:.2f})" for z, p in top.items()))

    np.savez(f"phase5_cross_atlas{tag}.npz",
             X_cross=merged.obsm["X_cross"], X_pca_cross=merged.obsm["X_pca_cross"],
             atlas=atlas, ct_zheng=ctz, ct_yost=cty)
    print(
        f"\n完成：phase5_cross_atlas_mixing{tag}.csv / "
        f"_head_alignment{tag}.csv / _latent_knn_alignment{tag}.csv / "
        f"_latent_knn_alignment_pca{tag}.csv / "
        f"phase5_cross_atlas{tag}.npz"
    )


if __name__ == "__main__":
    main()
