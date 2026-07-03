# 知识框架与工具箱

> 本文目标：在你动手之前，用**直觉**把整个项目的知识框架搭起来——你在处理什么生物问题、它如何变成一个计算问题、scAtlasVAE 这个模型到底在做什么、以及你会用到的每个工具是干什么的。
> 阅读方式：**直觉优先**。正文只讲"是什么、为什么"；需要一点数学的地方，放在"深入（可选）"框里，第一遍可以跳过。

---

## 1. 心智模型：四步看懂这个项目

整个项目可以浓缩成一条链：**生物问题 → 计算问题（数据整合）→ 用 VAE 解决 → scAtlasVAE 的独特设计**。下面逐段建立直觉。

### 1.1 生物问题：给 CD8⁺ T 细胞画一张"全景地图"

- **CD8⁺ T 细胞**是免疫系统的"杀手细胞"，通过 T 细胞受体（TCR）识别并清除被感染或癌变的细胞。它们有很多**状态（亚型）**：初始（naive, Tn）、效应（effector）、记忆（memory）、以及在肿瘤/慢性炎症中因长期抗原刺激而进入的**耗竭（exhausted, Tex）**状态。
- **单细胞 RNA 测序（scRNA-seq）**能测出**每一个细胞**里每个基因的表达量，得到一张巨大的表格：**行是细胞、列是基因、值是计数（count）**（这个基因的 mRNA 被捕获到多少条）。这张表**非常稀疏**（大部分格子是 0）且噪声大。
- **图谱（atlas）**：把很多研究、很多样本的细胞**合并成一张统一的参考地图**，覆盖 CD8⁺ T 细胞的所有状态。本论文构建的图谱有 115 万细胞、来自 68 个研究。

### 1.2 计算问题：数据整合（integration）

把很多来源的数据拼在一起，会遇到一个核心障碍：**批次效应（batch effect）**。

> **什么是批次效应**：每个研究/样本是在不同时间、不同实验室、不同实验方案、不同测序仪上做出来的，这些**技术差异**会系统性地改变数据。结果是：**来自两个研究的同一种细胞，可能仅仅因为"批次不同"就看起来不一样**——这不是生物学差异，是技术噪声。

**数据整合的目标**：把所有细胞放进一个统一的空间，让它们**按生物学（细胞类型/状态）聚在一起，而不是按批次（来自哪个研究）聚在一起**。一句话——**去掉技术批次、保留生物学信号**。

> **为什么这么做**：这里有一个内在张力。校正得太狠，会把真实的生物学差异也一起抹掉；校正得不够，批次又混不开。好的整合方法要在两者间取得平衡——这也是阶段 2 用指标去量化"整合到底好不好"的原因。

几个反复出现的概念：

- **潜空间 / 嵌入 (latent space / embedding)**：与其用约 4000 个基因描述一个细胞，不如把它压缩成一个短向量（比如 10 个数）来抓住它的"本质状态"。整合方法产出的就是这个嵌入；后续的聚类和可视化都在它上面做。
- **聚类 (clustering)**：用 Leiden/Louvain 算法把相似的细胞分组。
- **UMAP**：把高维嵌入压到 2D **画成图给人看**——它是**可视化**手段，不是分析本身。你的 UMAP 和论文长得不完全一样很正常（见 `00` 总纲）。
- **注释 (annotation)**：给每个聚类贴上细胞类型标签（依据标志基因 marker genes）。

### 1.3 用 VAE 解决：变分自编码器直觉

scAtlasVAE 的核心是一个 **VAE（变分自编码器, Variational Autoencoder）**。分三层理解：

**(a) 自编码器 (autoencoder)**：一个神经网络，先用**编码器 (encoder)** 把输入压缩成一个很小的"瓶颈"向量，再用**解码器 (decoder)** 从这个向量重建出原输入。瓶颈很窄，逼着网络学会数据的"精华"——这个精华就是我们要的**潜空间嵌入**。

**(b) "变分" (variational)**：普通自编码器的瓶颈是一个确定的点；VAE 的编码器输出的是一个**分布**（均值 μ、方差 σ²），我们从中**采样**得到潜向量 z，再加一个**正则项（KL 散度）**把这个分布往标准正态 N(0, I) 拉。这样潜空间会变得平滑、连续、且有"生成能力"。

> **深入（可选）——重参数化技巧 (reparameterization trick)**：采样这一步本身不可导，没法反向传播。技巧是把采样写成 `z = μ + σ · ε`，其中 `ε ~ N(0, 1)`。这样随机性被挪到与参数无关的 ε 上，μ 和 σ 就可导了。scAtlasVAE 里对应 `q_var = exp(z_var_fc(x)) + eps`、再 `Normal(q_mu, q_var.sqrt()).rsample()`。

**(c) 为什么要"生成式/概率式"来建模 scRNA 计数**：因为计数数据有三个特点，普通的"预测一个数"处理不好：

- 是**计数**、不是连续值 → 用**负二项分布 (negative binomial, NB)**。它像泊松分布，但允许"方差大于均值"（真实数据是**过离散 overdispersed** 的）。
- **零特别多**（dropout：一个基因其实表达了，但没被捕获到）→ 在 NB 上再加一块"零膨胀"，得到 **ZINB（零膨胀负二项, Zero-Inflated Negative Binomial）**。
- 于是解码器为每个基因输出**三组参数**来刻画 ZINB：
  1. **均值比例 (scale)**：对所有基因做 softmax 得到"占比"，再乘以该细胞的**文库大小 (library size)**（总计数）得到计数尺度的均值 μ；
  2. **离散度 (dispersion, θ)**：控制方差比均值大多少（代码里存成 logits，用时取 `.exp()`）；
  3. **零膨胀门控 (dropout/gate, π)**：额外多大概率直接吐一个 0。

> **常见坑**：编码器**输入**要先做 `log(1+x)` 归一化（`log_variational=True`），但**重构目标是原始计数**。别把"输入归一化"和"重构对象"搞混——这是阶段 3 手写时最容易错的地方之一。另外，凡是走 ZINB，`adata.X` 必须是**整数计数**，且每个细胞总计数 > 0，否则训练会出 NaN。

**(d) 损失函数 (loss)** = **重构损失**（ZINB 对观测计数的拟合有多好）+ **KL 项**（把潜空间正则化）。有标签时再加一个**分类损失**（见下）。

> **深入（可选）——KL 预热 (warmup) 与后验坍缩 (posterior collapse)**：如果一开始就给 KL 项很大的权重，模型会图省事让潜空间直接坍缩成 N(0,1)、根本不携带信息（后验坍缩）。解决办法是让 KL 的权重 λ_KL 从 0 慢慢线性升到目标值——先学会用潜空间重构，再逐步正则化。论文里这个"预热"贯穿整个训练。**关掉预热**是阶段 4 一个很好的消融实验。

### 1.4 scAtlasVAE 的独特设计：与 scVI 的关键区别（全项目最重要的一点）

市面上已有 scVI、scANVI、SCALEX、scPoli 等 VAE 方法。scAtlasVAE 的关键创新在于**编码器不看批次**。论文里给了这张对比表：

| 方法 | 编码器 | 解码器 | 重构 |
|---|---|---|---|
| **scAtlasVAE** | `F(X) → z`（**只吃基因表达**） | `F(z, B) → (r_mean, r_var, r_gate)` | ZINB |
| scVI / scANVI | `F(X, B, S) → (z, z_l)`（吃了批次 B 和文库 S） | `F(z, z_l, B) → ...` | ZINB |
| SCALEX | `F(X) → z` | `F(z, B) → X̃` | BCE |
| scPoli | `F(X, B) → z`（吃了批次 B） | `F(z, B) → ...` | ZINB |

- **批不变编码器 (batch-invariant encoder)**：scAtlasVAE 的编码器输入**只有基因表达 X**，不含批次。批次信息只在**解码端**注入（`decode()` 里把 batch 先过一个 embedding 层，再和 z 拼接送进解码 MLP）。
- **这带来什么能力（zero-shot 迁移）**：因为编码器从不看批次，一个**全新的查询数据集**可以**不重新训练**、直接过同一个编码器映射进参考图谱——这就是"零样本迁移 (zero-shot transfer)"。而 scVI 的编码器依赖批次，来了新批次就得做"架构手术"或重训。
- **多个细胞类型预测器**：当有标签时，在潜空间 z 上接**多个独立的分类头**做交叉熵。这让**跨图谱的标注对齐**成为可能。**单个 atlas 只需要一个分类头**——多头只在跨图谱时才有意义（这是阶段 3 可以合理简化的地方）。

> 回到 `00` 总纲的"北极星"问题 1、2、5——现在你应该能开始回答它们了。

---

## 2. 工具箱总表：你会遇到的每个包干什么

下面每个工具在后续阶段都会用到。现在**不用记**，混个脸熟，用到时回来查。

| 工具 | 是什么 | 在本项目里干什么 | 官方文档 |
|---|---|---|---|
| **NumPy** | Python 数值计算基础库（多维数组） | 一切数据的底层数组表示 | numpy.org/doc |
| **pandas** | 表格数据处理库 | 存放细胞的元信息（`obs`：批次、细胞类型等） | pandas.pydata.org/docs |
| **PyTorch** | 深度学习框架（张量 + 自动求导 + GPU） | 定义、训练神经网络；阶段 3 手写 VAE 的语言 | pytorch.org/docs |
| **AnnData** | 单细胞数据的标准容器 | 把"表达矩阵 + 细胞信息 + 基因信息 + 嵌入"打包在一个对象里 | anndata.readthedocs.io |
| **Scanpy** | 单细胞分析工具箱（基于 AnnData） | 预处理（QC/HVG/归一化）、聚类（Leiden）、UMAP、画图 | scanpy.readthedocs.io |
| **scvi-tools** | VAE 类单细胞方法的官方现代实现 | 阶段 2 跑 baseline（scVI）；也是理解 scAtlasVAE 的最佳对照 | docs.scvi-tools.org |
| **scib-metrics** | 整合评测指标库（GPU 加速） | 量化"整合好不好"（batch 校正 vs 生物保留） | scib-metrics.readthedocs.io |
| **umap-learn** | UMAP 降维算法 | 把嵌入压到 2D 可视化 | umap-learn.readthedocs.io |
| **leidenalg** | Leiden 图聚类算法 | 在嵌入上把细胞分成簇 | 随 scanpy 调用 |
| **scAtlasVAE** | 本论文的方法本体 | 阶段 1–2 调它跑通；阶段 3 对照它手写 | scatlasvae.readthedocs.io |

### 几个主力工具的"包速览"

> **包速览 — PyTorch**：深度学习框架。三件事：① **张量 (tensor)**，像 NumPy 数组但能放到 GPU 上；② **自动求导 (autograd)**，你只写前向计算，它自动算梯度；③ **`torch.nn`**，搭网络的积木（线性层、激活函数等）。阶段 3 你会直接用 `torch.nn` 和 `torch.distributions` 手写模型。

> **包速览 — AnnData**：单细胞数据的"标准集装箱"。一个 `adata` 对象里：`adata.X` 是表达矩阵（细胞×基因）；`adata.obs` 是细胞的元信息表（批次、细胞类型…）；`adata.var` 是基因信息表；`adata.obsm` 放每个细胞的低维嵌入（如 `X_pca`、`X_scAtlasVAE`）；`adata.layers` 放同尺寸的其他矩阵（如原始计数 `counts`）。**几乎所有单细胞工具都以 AnnData 为通用货币。**

> **包速览 — Scanpy**：基于 AnnData 的分析工具箱，函数按模块分：`sc.pp.*` 预处理（`normalize_total`、`log1p`、`highly_variable_genes`）、`sc.tl.*` 工具（`leiden`、`umap`、`rank_genes_groups`）、`sc.pl.*` 画图。阶段 2 的外围流程基本都靠它。

> **包速览 — scvi-tools / scib-metrics**：前者是 scVI/scANVI 等方法的**官方现代实现**——baseline 直接调它，别自己手写；后者是整合评测指标的现代库。**注意**：论文用的是**旧版 `scib`(1.1.4)**，我们用现代 `scib-metrics`，两者**数值不可直接比**，只看方法间相对排序（这点阶段 2 会反复强调）。

---

## 3. 试一试：亲手感受 AnnData（把"知道"变成"会用"）

下面两个小练习可以在**阶段 1 建好的 `scatlasvae` 环境**里运行（它已装好 `scanpy`/`anndata`）。目的是在正式上真实数据前，先摸清数据结构长什么样。

> **试一试 1 — AnnData 的解剖**：新建一个文件 `try_anndata.py` 粘贴运行，观察打印出来的每一部分对应上面"包速览"里的哪个属性。

```python
"""练习：亲手创建并解剖一个 AnnData 对象，理解它的四大组成部分。"""
import numpy as np
import pandas as pd
import anndata as ad

rng = np.random.default_rng(0)
n_cells, n_genes = 6, 4

# X：表达矩阵（行=细胞，列=基因），这里用整数模拟原始计数
X = rng.poisson(1.0, size=(n_cells, n_genes)).astype("int32")

adata = ad.AnnData(
    X=X,
    obs=pd.DataFrame(                       # obs：每个细胞的元信息
        {"batch": ["s1", "s1", "s2", "s2", "s3", "s3"]},
        index=[f"cell_{i}" for i in range(n_cells)],
    ),
    var=pd.DataFrame(                        # var：每个基因的信息
        index=[f"gene_{j}" for j in range(n_genes)]
    ),
)

print(adata)                                # 一览：维度 + 各字段
print("形状 (细胞, 基因):", adata.shape)
print("表达矩阵 X:\n", adata.X)
print("细胞元信息 obs:\n", adata.obs)
print("每个细胞的总计数(文库大小):", np.asarray(adata.X).sum(axis=1))
```

> **试一试 2 — 归一化前后对比**：这正是 scAtlasVAE 编码器输入所做的 `normalize_total(1e4) + log1p`。运行后对比同一个细胞归一化前后的数值，理解"为什么要按文库大小归一化"。

```python
"""练习：观察 normalize_total + log1p 对表达值的影响。"""
import scanpy as sc
import numpy as np

adata = sc.datasets.pbmc3k()                # scanpy 自带的小型真实数据集
adata.layers["counts"] = adata.X.copy()     # 先把原始计数备份到 layers（好习惯）

print("归一化前，第 0 个细胞前 5 个基因:", np.asarray(adata.X[0, :5].todense()).ravel())
print("第 0 个细胞总计数:", np.asarray(adata.X[0].sum()))

sc.pp.normalize_total(adata, target_sum=1e4)  # 把每个细胞缩放到总计数=1e4，消除测序深度差异
sc.pp.log1p(adata)                            # log(1+x)，压缩动态范围、稳定方差

print("归一化后，第 0 个细胞前 5 个基因:", np.asarray(adata.X[0, :5].todense()).ravel())
```

> **延伸阅读**：AnnData 与 scanpy 的权威入门见《Single-cell best practices》的"数据结构"章：https://www.sc-best-practices.org/introduction/fundamental_data_structures_and_frameworks.html ；数据整合的原理与方法综述见"数据整合"章：https://www.sc-best-practices.org/cellular_structure/integration.html

---

## 4. 一页速查（阶段 3 手写时回看）

- 编码器 `F(X) → (μ, σ²)`，**只吃 X**；重参数化得 z（默认 10 维）。
- 解码器 `F(z, B) → (scale, dispersion, gate)`，批次在**解码端**注入；`均值 μ = scale × 文库大小`。
- 损失 = ZINB 重构 + λ_KL·KL（KL 权重预热）+（可选）λ_ct·交叉熵分类。
- 默认超参：`n_latent=10`、`hidden=[128]`、`batch_hidden_dim=8`、`lr=5e-5`、AdamW、`batch_size=128`、`seed=12`；`max_epoch=min(round(20000/N×400), 400)`。
- 与 scVI 的唯一"题眼"：**编码器批不变 → zero-shot 迁移**。
