# 阶段 2 · 端到端跑通与整合评测

> **阶段** 2 / 6　·　**前置**：[阶段 1 · 环境搭建](phase1_environment_setup.md)　·　**产出**：整合前后 UMAP + 指标对比表 + 三份脚本　·　**预计** 3–4 天
> **导航**：[← 阶段 1](phase1_environment_setup.md)　·　[总纲](00_overview_and_learning_map.md)　·　[知识框架](01_concepts_and_toolbox.md)　·　[阶段 3 →](phase3_reimplement_vae.md)
>
> **本阶段结果均为本机真实实跑**：数据 = GSE156728 的 10X CD8 **全量 ~10.5 万细胞**（与论文 benchmark 同量级）；scAtlasVAE 使用 RTX 4060，baseline **scVI** 与 scib-metrics 在各自独立环境中运行（本机该 scVI 环境为 CPU）。指标表、UMAP、loss 曲线均为真实数据（含已修的 scib PCR 基线与官方训练损失问题，见 §7）。

---

## 1. 阶段概览

阶段 1 把环境搭好了。阶段 2 要用**作者的代码 + 一份真实数据**，走完单细胞整合的完整链路，并**学会两件真正可迁移的能力**：

- **能力 A — 拿到一份陌生单细胞数据，怎么自己"验货"**：格式对不对、能不能直接喂给模型。
- **能力 B — 怎么判断"整合到底好不好"**：这需要一套**量化指标**，而不是肉眼看 UMAP 图觉得"挺好"。

这一阶段是**近似 L1**：使用作者代码与同源真实数据，但不是带 28 个 `study_name` 的论文成品对象。证据来自当前 `patient` batch + `scib-metrics` 口径下预先定义的定量指标和同口径基线；它只能支持方向性的内部比较，不能仅凭 UMAP/“趋势对上”，也不能把绝对值与论文旧口径逐点对齐。

**全流程一图**（本阶段就是把它跑出来）：

![scAtlasVAE 整合效果（真实）](figures/fig_phase2_integration_umap.png)

*图 2-1 — **本机真实结果**（全量 ~10.5 万 CD8 细胞）。上排未校正 X_pca、下排 scAtlasVAE；左列按模型与指标实际使用的患者(batch)、右列按 17 个 CD8 亚型上色。可见整合后**患者批次更混合**、而 Tn/Tem/Temra/Tex 等**细胞类型仍分得开**——定量见 §7。*

---

## 2. 学习目标

完成本阶段后你应能：

- 自己**找到并验货**一个单细胞数据集是否满足模型要求（一套可复用的检查清单）；
- 说清标准预处理 **QC / HVG / 归一化**各是什么、为什么做；
- 理解整合评测的**两类指标**（生物保留 vs 批次校正），以及为什么不能只看一个；
- 会用 `scib-metrics` 把多种方法放一起比，并知道**看相对排序、不看绝对值**。

---

## 3. 侦查：数据去哪找、怎么"验货"

> 沿用[阶段 1](phase1_environment_setup.md) 的侦查法——**为什么找 → 去哪找 → 怎么动手 → 看到什么 → 结论**。

### 3.1 该用哪份数据——三处交叉确认

- **为什么找**：复现的是论文的 benchmark 实验，就该用论文 benchmark 用的数据，而不是随便找一份。
- **去哪找 / 怎么动手**：三处互相印证——
  1. **论文 Methods 的 "Benchmarking" 段**（PDF 里搜 `GSE156728`）明写："pan-cancer CD8⁺ T cell landscape containing **110,218 cells from 28 studies** (data available at **GSE156728**)"。
  2. **[总纲](00_overview_and_learning_map.md) 的复现路线**指向同一个 GEO 号。
  3. **官方文档** `gex_integration` 教程给了 CD8 数据的用法示例。
- **结论（本轮审计更正）**：当前可从 GEO **GSE156728** 重建的是 Zheng *et al.* 2021 的 10X CD8 主体；本项目实际使用其中 8 个癌种的 **104,805** 个细胞。它与论文 110,218-cell benchmark 同量级、也是其占绝对多数的主体，但**不是论文已经拼好并带 28 个 `study_name` 的成品 TCellLandscape 对象**。因此这是同源真实数据上的近似复现，不是 28-study 设置的逐项复刻。

> **为什么不用全 115 万 atlas**：那是受控数据 + 巨大算力，学习增益却很低（见[总纲 §3–4](00_overview_and_learning_map.md)）。11 万的 benchmark 足以复现"方法相对排序"这一核心结论。

### 3.2 拿到后必做的"验货清单"

scAtlasVAE 的 ZINB 重构对输入有**硬要求**（回顾 [知识框架 §1.4f](01_concepts_and_toolbox.md)：ZINB 建模的是**原始整数计数**）。数据一到手，先在 Python 里逐条查，别急着训练：

```python
import scanpy as sc, numpy as np
adata = sc.read_h5ad("tcell_landscape.h5ad")
print(adata)                       # 先看整体：多少细胞×基因、有哪些 obs/var/layers/obsm
print(adata.obs.columns.tolist())  # 列名五花八门，得亲眼确认 batch/label 列叫什么
print(adata.X[:3, :8].toarray() if hasattr(adata.X, "toarray") else adata.X[:3, :8])
print("每细胞总计数>0 :", bool((np.asarray(adata.X.sum(1)) > 0).all()))
```

| 要查什么 | 为什么 | 怎么判断 |
|---|---|---|
| `adata.X` 是**原始整数计数**？ | ZINB 需要 count；若已被 log 归一化就不能直接用 | 打印出的值是否为**非负整数**；或看有没有 `adata.layers['counts']` 备份 |
| **每细胞总计数 > 0** | 否则训练出 `NaN`（README "Common Issues" 第 2 条，也见[阶段 1](phase1_environment_setup.md)） | `(np.asarray(adata.X.sum(1))>0).all()` 为 `True` |
| **batch 键**叫什么 | 要传给 `batch_key`（可能是 `study_name`/`patient`/`cancerType`） | 看 `adata.obs.columns`，逐列 `adata.obs['列'].value_counts()` |
| **cell type 列**叫什么 | 半监督/评测要用（可能是 `meta.cluster`/`cell_type`） | 同上 |

- **结论（本机实测）**：本次用 GSE156728 的 10X CD8 子集（8 个癌种），组装成 **104,805 细胞 × 24,148 基因**（**全量、未下采样**，与论文 TCellLandscape 的 11 万同量级；组装脚本 `phase2_data_fetch_gse156728.py` 用分块流式读取，在 16GB 机器上也不爆内存）；`adata.X` 为原始整数计数、每细胞总计数均 > 0；**batch 列取 `patient`（45 个样本）、类型列取 `cell_type`（=meta.cluster，17 个 CD8 亚型）**。这些列名已填进各脚本顶部的 `CONFIG`。
  > **关于"11 万 vs 4 万"**：早期脚本默认 `--target 40000` 把数据随机下采样到 4 万（只为在 4060 上跑得快），文档一度把"源数据 11 万"和"实际用 4 万"混着说、造成困惑。**现已改为默认取全量 ~10.5 万**（epoch 数随细胞数自动变少，训练时间与 4 万时基本一个量级），与论文规模对齐。想要小规模先跑通仍可 `--target 40000`。

> **为什么这么做**：不同来源的 AnnData 列名千差万别。**先打印 `adata` 与 `adata.obs.columns` 把"这份数据长什么样"搞清楚，再动手**——这是所有单细胞分析的第一步，也是最常被跳过、然后在后面莫名报错的一步。

> **常见坑**：若打印出的 `X` 是小数（如 2.71、0.69），说明它已被 `log1p` 归一化过——**不能直接喂 ZINB**。去 `adata.layers` 找 `counts`；找不到就得回到数据源重新拿原始计数。

---

## 4. 会遇到的工具与术语

> **包速览 — scvi-tools**：scVI/scANVI 等方法的官方现代实现。本阶段用它跑 **scVI baseline**（别自己手写 baseline）。文档：docs.scvi-tools.org。

> **包速览 — scib-metrics**：单细胞整合评测指标库（YosefLab，JAX 加速）。核心是 `Benchmarker`：喂它一个 `adata`、若干个嵌入（`obsm` 里的 key）、batch 键、label 键，它一次算出全套指标并排名。文档：scib-metrics.readthedocs.io。

**术语速览**（第一次出现）：**QC**（质量控制，过滤低质量细胞/基因）· **HVG**（高变基因，细胞间变化最大的一批基因，本项目取 4000）· **PCA**（主成分分析，线性降维；这里用它得到一个**未做批次校正的基线嵌入** `X_pca`）· **近邻图 kNN graph**（每个细胞连到最近的 k 个邻居，Leiden 和 UMAP 都基于它）· **Leiden**（在近邻图上做社区发现得到聚类）· **UMAP**（把嵌入压到 2D 画图，是可视化手段、不是分析本身）。

---

## 5. 原理：这条流程为什么这样走

**预处理三步的动机：**

- **QC**：去掉将死细胞（线粒体基因占比过高）、空液滴（检测到的基因数过少）等，留下可信细胞。
- **HVG**：只留最有信息的约 4000 个基因——降噪、加速，且论文正是用 4000 HVG（Methods 原文）。
- **归一化**（编码器输入）：`normalize_total` 消除测序深浅差异 + `log1p` 压缩动态范围（原理见 [知识框架 §1.4f](01_concepts_and_toolbox.md)）。**注意**：这只作用于**编码器输入**；ZINB 重构的**目标仍是原始计数**——两者别混。

**怎么量化"整合好不好"——两类指标缺一不可：**

好的整合要同时满足两个**互相拉扯**的目标，所以指标分两类、最后取平均：

$$\text{Overall} = 0.6\cdot\underbrace{S_{\text{bio}}}_{\text{生物保留}} + 0.4\cdot\underbrace{S_{\text{batch}}}_{\text{批次校正}}$$

> **注**：这是 scib-metrics/scIB 论文的**默认加权（生物保留 0.6、批次校正 0.4）**，不是等权 ½。生物保留权重更大，所以一个"生物结构保得好、批次没校正"的方法（如 scaled PCA）总分也能不低——读总分时要记着这一点（详见 §7）。

- **批次校正 $S_{\text{batch}}$**：不同批次的同类细胞混得好不好。常用 batch ASW、graph connectivity、PCR 等。
- **生物保留 $S_{\text{bio}}$**：不同细胞类型分得开不开。常用 cell-type ASW、isolated-label F1/ASW 等。

> **为什么必须两类一起看**：只看批次校正，一个"把所有细胞搅成一团"的烂模型也能拿满分（批次当然混得均匀，但生物学也被抹平了）；只看生物保留，一个"完全不校正"的 PCA 也能把类型分开，但批次照样各成一团。**两股力互相制衡，平均分才有意义**——这也正是[知识框架 §1.4e](01_concepts_and_toolbox.md) 讲的"重构 vs KL 拔河"在评测层面的回声。

其中 **ASW（average silhouette width，平均轮廓宽度）** 最直观：对每个细胞看"它离**同类**多近、离**异类**多远"。对细胞 $i$：

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}} \in [-1, 1]$$

$a(i)$ 是它到同簇其他点的平均距离、$b(i)$ 是到最近异簇的平均距离；$s$ 越接近 1 越好。ASW 就是所有细胞 $s(i)$ 的平均。

> **常见坑（务必写进报告）**：论文用的是**旧版 `scib`(1.1.4)**，我们用现代 `scib-metrics`，官方明确说**两者数值不可直接比**。所以你算出的绝对分数不必和论文表格对齐——**只看方法之间的相对排序**（scAtlasVAE vs scVI vs 未校正 PCA）是否符合论文结论。

**三个对照对象**：`X_pca`（未校正基线）、`X_scVI`（batch-conditioned 生成模型；本项目默认 `encode_covariates=False`，encoder 不显式接收 batch）、`X_scAtlasVAE`（源码结构固定为 encoder 只接收 X）。两种结构都不保证 X 中的批次信号自动消失，整合效果仍由 patient-based 指标判断。

> **scvi-tools 的 Windows 安装坑（记一笔）**：`scvi-tools` 依赖 JAX 生态的 `orbax-checkpoint`，包内有超长路径测试文件，在**未开长路径**的 Windows 上会触发 260 字符上限而装不上。解决：以管理员执行 `Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1`、重开终端即可。本机据此单独建了 `scvi`(py3.10) 环境跑 scVI。（另附 [`phase2_baseline_harmony.py`](../scripts/phase2_baseline_harmony.py)：Harmony 作为可选的第二基线，不需要 scvi-tools。）

---

## 6. 操作步骤

> 训练在**环境 A（`scatlasvae`，py3.8）**；评测在**环境 B（`scib`，py3.10）**。为什么拆两个环境见 [阶段 1 附录](phase1_environment_setup.md)。

### 步骤 1 · 建好评测环境 B（若阶段 1 没建）

```powershell
conda create -n scib python=3.10 -y
conda activate scib
pip install scib-metrics scanpy scvi-tools
```

### 步骤 2 · 下载并验货（环境 A）

**目的**：从 GSE156728 重建并验收 104,805-cell Zheng CD8 对象、按 §3.2 清单验货、确定 batch/label 列名；该对象不是论文带 28 个 `study_name` 的成品 TCellLandscape。

```powershell
conda activate scatlasvae
python phase2_data_download_and_qc.py --stage check
```

见 [`phase2_data_download_and_qc.py`](../scripts/phase2_data_download_and_qc.py)：它打印 `adata`、`obs.columns`、`X` 是否整数、每细胞总计数是否 > 0。**把查到的 batch/label 列名填回脚本顶部 `CONFIG`。**

### 步骤 3 · 预处理 + 未校正基线（环境 A）

**目的**：QC/HVG/归一化，并算一个未做批次校正的 `X_pca` 作对照。

```powershell
python phase2_data_download_and_qc.py --stage preprocess
```

产出：`tcell_processed.h5ad`（含 `layers['counts']` 原始计数备份、4000 HVG、`obsm['X_pca']`）。

### 步骤 4 · 训练 scAtlasVAE → `X_scAtlasVAE`（环境 A）

```powershell
python phase2_run_scatlasvae.py
```

见 [`phase2_run_scatlasvae.py`](../scripts/phase2_run_scatlasvae.py)：`scAtlasVAE(adata=adata, batch_key=..., label_key=...)` → `fit()` → `adata.obsm['X_scAtlasVAE']=get_latent_embedding()`，并存回 h5ad。

**实跑（本机 ~10.5 万细胞）**：`max_epoch=min(round(20000/104805·400),400)=76` 个 epoch（细胞越多、epoch 越少，故 10.5 万训练时间与早期 4 万时相当），4060 上约 **20 分钟**；loss 稳定下降、**无 NaN**；末尾 10 个 epoch（`pred_last_n_epoch`）开始训练分类头，曲线上能看到一个小台阶。训练曲线（真实）：

![训练损失曲线（真实）](figures/fig_phase2_loss_curve.png)

*图 2-2 — 总损失/重构损失随 epoch 下降；右轴是 KL 权重的预热曲线。注意 λ_KL 在整个训练里从 0 线性爬到 ~1（因 `n_epochs_kl_warmup=min(max_epoch,400)` 被截断到 max_epoch）——[阶段 3 §8](phase3_reimplement_vae.md) 细讲，并纠正了旧稿"只到 0.18"的错。*

### 步骤 5 · scVI baseline → `X_scVI`（`scvi` 环境，需 scvi-tools）

```powershell
conda activate scvi
python phase2_baseline_scvi.py
```

见 [`phase2_baseline_scvi.py`](../scripts/phase2_baseline_scvi.py)：用 `scvi-tools` 默认参数、`max_epochs=10` 跑 scVI，得到 `obsm['X_scVI']`。CPU 上 ~10.5 万细胞 10 epoch 约十几分钟（本机实测）。

### 步骤 6 · UMAP + Leiden（可视化整合效果）

在处理好的 h5ad 上，对 `X_pca` 与 `X_scAtlasVAE` 各算一次近邻图 → UMAP → Leiden，按 **batch** 和按 **cell type** 两种上色出图。代码在 `phase2_run_scatlasvae.py` 的 `--stage umap`。**这一步产出的就是图 2-1 那种对照图。**

### 步骤 7 · scib-metrics 定量对比（环境 B）

```powershell
conda activate scib
python phase2_benchmark_scib.py
```

见 [`phase2_benchmark_scib.py`](../scripts/phase2_benchmark_scib.py)。核心就三行——把三个嵌入一起丢给 `Benchmarker`：

```python
from scib_metrics.benchmark import Benchmarker
bm = Benchmarker(adata, batch_key="patient", label_key="cell_type",
                 embedding_obsm_keys=["X_pca", "X_scVI", "X_scAtlasVAE"])
bm.benchmark()                 # 一次算全套指标
df = bm.get_results(min_max_scale=False)   # 拿到指标表
```

> **为什么这样比**：`Benchmarker` 会对每个嵌入分别算"批次校正"和"生物保留"两组指标、再综合排名。你要读的不是某个绝对分，而是**三个嵌入的名次**。

---

## 7. 结果（本机实测）

**指标对比（真实，scib-metrics）**：

![四方整合评测对比（真实）](figures/fig_phase2_benchmark_bars.png)

*图 2-3 — 四种嵌入的批次校正/生物保留/总分（**本机 scib-metrics 实测，~10.5 万细胞、已修 PCR 基线**）。scAtlasVAE 分"无监督/监督"两根柱。*

| 嵌入 | 批次校正 $S_{\text{batch}}$ | 生物保留 $S_{\text{bio}}$ | 总分 Overall |
|---|---|---|---|
| `X_pca`（未校正） | 0.271 | 0.486 | 0.400 |
| `X_scVI` | 0.312 | 0.485 | 0.416 |
| `X_scAtlasVAE_unsup`（无监督） | 0.309 | 0.478 | 0.411 |
| **`X_scAtlasVAE_sup`（监督）** | **0.336** | **0.515** | **0.444** |

> **两处更正（都值得记，是本项目最实的"诚实"）**：
> 1. **监督/无监督**：早先这里只有一根 `X_scAtlasVAE` 柱，其实它**传了 `label_key`、是监督版**。按修复后的官方训练代码重训并补上无监督版后：**监督(0.444) 明显最高、且批次校正与生物保留两项都最高**；无监督(0.411)接近 scVI(0.416)——scAtlasVAE 相对 scVI 的主要增益来自**半监督分类头**。
> 2. **一个 scib-metrics 配置 bug（修复后结论更保守也更真实）**：早先 `PCR comparison` 这一列对所有方法**恒为 0**，且 `X_pca` 基线被 Benchmarker 用**原始计数**现算的 PCA 覆盖（生物保留被压到 0.37）。根因：没给 `pre_integrated_embedding_obsm_key`、而我们 `adata.X` 是原始计数（详见脚本注释与 [阶段 5 · E5](phase5_deeper_validation.md)）。**修复后重跑**：PCR 恢复区分度（监督 0.130 > scVI 0.082 > 无监督 0.059 > PCA 0），而**正确的 scaled-log PCA 基线生物保留高达 0.486**——所以旧稿"VAE ≫ PCA"里那道大差距，**一部分是 bug 把 PCA 基线算错造成的假象**。

**怎么读这张表（相对排序 = 复现判据）**：

- **总分排序**：**监督 scAtlasVAE(0.444) > scVI(0.416) ≈ 无监督(0.411) > 未校正 PCA(0.400)**。注意 scib-metrics 的总分 = **0.4·批次校正 + 0.6·生物保留**（生物保留权重更大，见 [_core.py](../../scAtlasVAE) 里的 scIB 默认加权），而 scaled PCA 的生物保留本就很高，**所以"总分"上 PCA 是个比想象中强得多的基线**，"≫ PCA"在总分口径下**不再成立**。
- **VAE 相对 PCA 的真实优势在"批次校正"这一列**：PCA 0.271 < 无监督 0.309 ≈ scVI 0.312 < **监督 0.336**——这才是 §5"两类指标缺一不可"的正确体现（PCA 生物保留高但患者批次混不开）。
- **与论文同向的哪一条**：论文 Ext. Data Fig. 2a 说"无监督与 scVI 相当、监督明显胜出"；我们的内部相对排序复现了这个方向。但 batch 键与指标实现不同，不能称逐点复现。
- **为何绝对分/差距和论文不同**：① 我们 batch=**patient**（同一研究内、批次效应弱），论文 batch=**study**（跨 28 研究、批次效应强），差距被压扁；② scib-metrics ≠ 论文旧 scib，指标口径不同（对照见 [阶段 5 · E5](phase5_deeper_validation.md)）。**判据是相对排序，不是绝对值对齐。**（batch 键这条很关键，单列一段说明见下方 ⬇️）
- **（附）第二基线 Harmony**：另用 [`phase2_baseline_harmony.py`](../scripts/phase2_baseline_harmony.py) 跑过 Harmony（线性迭代式校正）作互补对照，想加进对比把 `X_harmony` 塞进 `phase2_benchmark_scib.py` 的 `EMBEDDINGS` 即可。

> **⬇️ 为什么我们用 `batch=patient` 而不是论文的 `study`（重要的诚实设置说明）**
>
> **这不是随手一填，也不是完全的自由选择，而是"在数据里真实存在的字段中主动挑了 patient"：**
> - **GSE156728 的公开 metadata 里根本没有 `study` 这一列**。原始 `GSE156728_metadata.txt.gz` 只有 7 列：`cellID / cancerType / patient / libraryID / loc / meta.cluster / platform`——**没有**论文所说的那个 28 值 `study_name`。技术/样本批次候选主要是 `patient`(45) 与 `libraryID`；`cancerType`(8) 是生物变量、不能拿来顶替 batch。这里选 `patient`（最清楚的患者/样本域）。
> - **我们下的数据对不对？——用细胞数核实：对，拿到了 95%。** 数 `GSE156728_metadata.txt.gz` 里全部 CD8 细胞（meta.cluster 以 CD8 开头）：**Zheng 自产 CD8 共 109,389**（10X 109,089 + SS2 300），论文 **TCellLandscape CD8 = 110,218**，差 829（0.75%）。我们的 104,805 精确对上：`104,805（8 癌种 10X）+ OV(3,517) + FTC(767) + CHOL的SS2(300) = 109,389`——即**拿了 GSE156728 里 CD8 的 95%**，只漏 OV/FTC/CHOL 3 个小队列（~4,584）。补齐就再下 `GSE156728_OV_10X.CD8.counts` 与 `GSE156728_FTC_10X.CD8.counts`。
> - **那"28 studies"到底是啥？（已查 Supp Table 1）** 论文 **Supplementary Table 1** 把 TCellLandscape 标注为 **28 个源研究的合集**（`Zheng_2021` + `van Galen_2019 / Yost_2019 / Guo_2018 / Savas_2018 / Zilionis_2019 / Zheng_2017 / Zhang_2018/2019/2020 / Ma_2019 / Li_2019 / Jerby-Arnon_2018 / …` 等 ~17 个外部数据集），`study_name` 就是每个细胞的**源研究名**。**但**其 CD8 总数（110,218）≈ GSE156728 里 Zheng 自产 CD8（109,389）——说明 **Zheng 自产数据占绝大多数、外部源贡献很小**；每个源的确切细胞数拆分不在公开文件里（在 Zenodo 成品对象里）。**关键点不变**：这套 28 路 `study_name` 是作者组装图谱时**贴的来源标签**，**不在 GSE156728 的基础 metadata**（只有 `cancerType`/`patient`/`libraryID`/`platform`），所以我们从原始计数出发拿不到它。〔更正记录：这条我先后读错过两次——先从多面板图注误合并、又据细胞数误判成"纯 Zheng、非合集"；现以 Supp Table 1「28 源合集」+ 细胞数「Zheng 占绝大多数」二者并存为准。〕
> - **所以为何用 `patient`**：我们下到的是 Zheng 自产、**10X 平台、8 个癌种**这一块，内部同源，**没有 28 路 study 标签可用**，能当 batch 的真实字段只有 `patient`(45)/`cancerType`(8)/`libraryID`——我们主动选了最标准的 `patient`。
> - **对结果的影响（诚实、且偏保守）**：`patient`（同研究/同平台内）要校正的批次效应，比论文那种带内部多队列 `study_name` 的设定弱 → 各方法（PCA/scVI/scAtlasVAE）能拉开的差距被压扁，但**方向不变**（监督最高、VAE>PCA）。**batch 越弱，越难显出"scAtlasVAE 校正得更好"，所以这个设置是让方法更不容易赢、而非往好看里修**——是保守的复现，不是取巧。真正的强跨研究场景，我们用 **[Task 2](phase5_deeper_validation.md)（引入 Yost 2019 作第二图谱）** 专门补回。
> - **不能把 `cancerType` 改当 batch 来补 study**：癌种是需要保留的**生物变量**，不是技术批次。按癌种上色的 UMAP 可以描述癌种结构，却不能证明 `patient` batch 已被移除；把癌种作为校正目标还可能主动抹掉真实生物差异。若要精确复现 leave-one-study，只能取得带 `study_name` 的论文成品对象或重新可靠映射每个细胞的来源研究。

**记录区（本机实测）**：
```
数据：细胞数=104805（GSE156728 全量 CD8 10X，8 癌种）  batch列=patient(45)  label列=cell_type(17 个 CD8 亚型)  基因=4000 HVG
训练：scAtlasVAE epoch=76（自动=min(round(20000/N·400),400)）  有无NaN=无  λ_KL 末值≈1
指标（真实，总分，PCR与训练损失均已修）：X_pca=0.400 / X_scVI=0.416 / scAtlasVAE(无监督)=0.411 / scAtlasVAE(监督)=0.444
批次校正一列（VAE 真实优势）：PCA 0.271 < 无监督 0.309 ≈ scVI 0.312 < 监督 0.336
结论与论文趋势：监督 scAtlasVAE 两项皆最高（复现 Ext.Data Fig.2a 的"监督胜出"）；"≫PCA"仅在批次校正列成立、总分口径下 PCA 是强基线
```

---

## 7.1 与论文 Supplementary Table 3 的交叉验证（同一份 Zheng 数据，强背书）

论文 Supplementary Table 3 给出了 "Benchmark 1: single-atlas integration of CD8+ T cells from Zheng et al., 2021" 的分数。我们的 GSE156728 重建对象来自同一 Zheng 数据来源、规模接近，但过滤后为 104,805 细胞且缺少成品对象的 `study_name`，因此这些论文数字只作背景参照，不能逐点直接对标。

**① 生物保留核心指标（label silhouette，与 batch 层级无关）——方向和我们完全一致：**

| 方法 | 论文 label silhouette（Supp Table 3） | 与我们的结论 |
|---|---|---|
| **scAtlasVAE_supervised** | **0.503（最高）** | ✅ 监督最高 |
| scVI | 0.492 | ✅ 无监督 ≈ scVI |
| scAtlasVAE（无监督） | 0.487 | ✅ 略低于 scVI |

> 论文同时按 `sample_name` 与 `study_name` 两个层级计算批次指标，方法排序方向一致。我们的 `patient` 是当前最接近 sample/domain 的代理，但没有可靠字段映射证明它等同于论文 `sample_name`；因此这里能说的是内部排序与论文方向一致，不能说 batch 层级完全复刻。

**② 注释迁移（Benchmark 3，Zheng 来源数据）——A 的留出单位可对应，数据对象仍非完全相同：**

| | 论文 scAtlasVAE zero-shot | 我们（论文协议·末10轮） |
|---|---|---|
| 随机留 5% | 0.905 | **0.891** |
| 留一个 study | 0.859 | **当前数据无 `study_name`，不能做精确对应** |

> 注：随机留 5% 可以和论文对应；本项目另做的整癌种 UCEC（设计 B）是**生物域外泛化**，整位 patient（设计 P）是当前字段下最接近的**批次/样本域留出类比**，两者都不能冒充论文的 leave-one-study。若要比较 B/P，只能在本项目内部比较方法，不能把它们与论文 0.859 的差值解释成复现误差。

**③ 超参网格（Supp Table 3 后半段）证明我们的参数没问题**：论文对 `batch_hidden ∈ {16,32,64}`、`n_latent ∈ {5,10,20}` 做了完整网格——**16/32/64 几乎无差（overall 波动 <0.02）、`n_latent` 10 和 20 都好、5 系统性偏低约 0.03**（Zheng 网格 overall：5≈0.78 < 10≈0.81 ≈ 20≈0.81）。这直接背书：我们 `n_latent=10` 的选择站得住；`batch_hidden_dim=10` 虽**略低于论文网格下界 16**，但该维度 16/32/64 近乎无差、外推到 10 风险很低。我们消融"n_latent=2 差、10/50 好"也与论文"5 偏低、10/20 好"同向。

**结论（回答"要不要按 study 重训"）**：当前文件里没有 `study_name`，所以**不是选择不重训，而是现有数据无法做精确的 study 级复现**。我们的 `patient` 设置可回答样本/患者批次整合，设计 P 可回答整位患者留出；它们足以支撑当前范围内的相对比较，但不能替代论文 28-study 设置。若未来取得带可靠 `study_name` 的成品对象，应把它作为一项新的、独立实验完整重训，而不是把 `cancerType` 改名顶替。

---

## 8. 检查点与完成标准（DoD）

- [x] 数据通过 §3.2 验货（整数 count、总计数 > 0、batch/label 列已确定）
- [x] `X_scAtlasVAE`、`X_scVI`、`X_pca` 三个嵌入都已存入 `obsm`
- [x] 整合前后 UMAP 出图；`scib-metrics` 指标表出炉
- [x] **相对排序符合论文趋势**：监督 scAtlasVAE 两项(批次校正+生物保留)皆最高、复现"监督胜出"；VAE 相对 PCA 的优势在批次校正一列（总分 0.444/0.416/0.411/0.400，PCR 与训练损失均已修，详见 §7）

---

## 9. 自测题

1. 拿到一份陌生的单细胞数据，你会先查哪几件事？为什么每细胞总计数必须 > 0？
2. QC、HVG、归一化分别解决什么问题？为什么 scAtlasVAE 只用 4000 个基因？
3. 整合评测为什么要分"批次校正"和"生物保留"两类？只看其中一个会怎样？（用图 2-3 里 PCA 的例子说明）
4. 你的 `scib-metrics` 绝对分数和论文对不上，这说明复现失败了吗？正确的判断标准是什么？

---

## 10. 延伸阅读

- 单细胞预处理与整合权威教程：《Single-cell best practices》[数据整合章](https://www.sc-best-practices.org/cellular_structure/integration.html)
- scVI 教程：https://docs.scvi-tools.org/en/stable/tutorials/index.html
- scib-metrics：https://scib-metrics.readthedocs.io/
- 官方整合教程：https://scatlasvae.readthedocs.io/en/latest/gex_integration.html

---

> **导航**：[← 阶段 1](phase1_environment_setup.md)　·　[总纲](00_overview_and_learning_map.md)　·　[阶段 3 · 核心 VAE 从零重写 →](phase3_reimplement_vae.md)
