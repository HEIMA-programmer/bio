# scAtlasVAE 复现规划

> 论文：Xue, Wu, Tian *et al.* (2024) *Integrative mapping of human CD8⁺ T cells in inflammation and cancer.* **Nature Methods**. DOI: 10.1038/s41592-024-02530-0
> 代码：https://github.com/WanluLiuLab/scAtlasVAE　文档：https://scatlasvae.readthedocs.io/en/latest/
>
> **你的约束**：RTX 4060（8GB，Ada Lovelace / sm_89）· 2 周（每天可高投入）· 产出为"详细复现报告 / 组会汇报" · 已有 Claude Code
>
> 本文件是一份**执行路线图**，不是最终交给导师的报告。建议把它放进你 fork 的仓库根目录，让 Claude Code 能读到它、你也能逐条打勾。

---

## 0. 先明确"复现到什么程度算学到东西"

学术界（NeurIPS 复现性报告等）把"复现"分成一条由浅入深的谱系，你要清楚自己停在哪层、为什么：

| 层级 | 含义 | 学习价值 | 本次是否做 |
|---|---|---|---|
| Repeatability | 原代码 + 原数据，跑一遍得到一样的数 | 几乎为零 | 只作为热身（L0） |
| Reproducibility | 用作者代码/数据，重生成论文结果 | 中 | 做（L1） |
| Replicability | **自己重写核心方法**，用（可不同的）数据得到相似结论 | 高——真正的训练在这里 | **做（L2，重点）** |
| 扰动 / 迁移 | 消融设计选择、迁移到新数据、挑战某个结论 | 最高，已是研究 | 做一点（L3），象征性碰 L4 |

**核心心态**：你复现对没对，判断标准是**结论和趋势**（batch 被校正、Tex 分成三亚型、指标量级接近），**不是像素级重合、也不是 MSE 归零**。版本、随机种子、GPU 浮点都会让你的 UMAP 与论文的图不一致，这在真实科学复现里本就正常。报告里要诚实写清做了什么规模、没做什么、为什么。

**本次目标定位**：以 **L2（核心 VAE 从零重写）为必达底线**，配 1–2 个 L3 消融，产出一份"理解透彻 + 有独立发现 + 有消融"的复现报告。这远胜于"把 115 万细胞跑通但说不清为什么"。

---

## 1. 环境搭建（第一天最容易翻车的地方）

### 1.1 关键坑：4060 与仓库锁定的 PyTorch 不兼容

仓库 `requirements.txt` 锁死 `torch==1.13.1`（默认 cu117 构建，2022 年）。你的 4060 是 **Ada Lovelace，算力 sm_89**，而 **CUDA 11.7 / cu117 里没有为 sm_89 预编译的核**。后果是：`torch.cuda.is_available()` 可能返回 True，但一做矩阵乘就抛 `CUBLAS_STATUS_INVALID_VALUE`（正是 README "Common Issues" 里那条），或报 `sm_89 ... not compatible`。**支持 Ada 的最低 CUDA 是 11.8（cu118）。**

### 1.2 推荐策略：拆成两个环境（训练环境 + 评测环境）

原因：scAtlasVAE 的旧依赖栈（`python=3.8`、`scanpy==1.8.1`、`numpy==1.21.6`、`numba==0.57.1`）与现代评测工具 `scib-metrics`（要求 Python ≥ 3.10）冲突。硬凑一个环境会陷入依赖地狱。干净做法是解耦：**在训练环境里训练、导出 latent embedding 到 `adata.obsm`，再在独立的现代环境里算指标。**

**环境 A —— 训练环境（跑 scAtlasVAE）**

保留其余旧依赖，只把 PyTorch 换成支持 Ada 的构建。torch 2.0–2.2 的 cu118 构建同时支持 Python 3.8 和 sm_89：

```bash
# 用 uv（见 §8 工具清单，比 conda 快很多）或 conda 建 py3.8 环境
conda create -n scatlasvae python=3.8 -y && conda activate scatlasvae

# 关键：装支持 Ada 的 PyTorch（cu118），不要用仓库锁的 1.13.1+cu117
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 其余按仓库来（若某个包与 torch 2.x 冲突，逐个放宽版本）
pip install scanpy==1.8.1 scirpy==0.10.1 anndata==0.8.0 \
            numpy==1.21.6 numba==0.57.1 scikit-learn==0.24.1 \
            umap-learn==0.5.1 einops==0.4.1 seaborn==0.12.2

# 安装 scAtlasVAE 本体（从你 clone 的仓库，可编辑安装，便于改源码对照）
pip install -e /path/to/scAtlasVAE
```

**装完立刻做冒烟测试**（别急着上数据）：

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
x = torch.randn(1024, 1024, device="cuda")
y = (x @ x).sum(); y.backward if False else None   # 触发 cublas
print("matmul OK:", float(y.mean()))
```

能打印出结果、不抛 cublas 错，才算环境真的通了。

**若 torch 2.x 下 scAtlasVAE 报 API 错**：通常是小改（如某个已改名的 torch 函数），让 Claude Code 帮你逐处 patch。**实在搞不定的备选**：① 用 Docker/conda 装一个 cu118 的完整旧栈；② 本地 4060 只做开发调试，把需要跑的那一两次训练放到云端免费卡（Colab/Kaggle 的 T4/L4）。

**环境 B —— 评测环境（算 scib-metrics 指标）**

```bash
conda create -n scib python=3.10 -y && conda activate scib
pip install scib-metrics scanpy   # GPU 加速需另装对应版本的 JAX
```

⚠️ **重要提醒**：`scib-metrics` 的指标实现与论文用的旧 `scib` 有差异，官方明确说**两者数值不可直接对比**。所以你算出的绝对分数不必和论文表格对齐；你要看的是**方法之间的相对排序**（scAtlasVAE vs scVI vs 未校正 PCA）是否符合论文结论。这一点在报告里要写明。

---

## 2. 数据方案

### 2.1 主力数据集：TCellLandscape（不用全 atlas）

| 数据 | 规模 | 来源 | 本次角色 |
|---|---|---|---|
| **TCellLandscape**（Zheng et al. 2021） | ~11 万 CD8⁺ 细胞 / 28 studies | GEO **GSE156728** | **主力**：公开、可下、4060 跑得动、本就是论文 benchmark 数据 |
| TCellMap（Chu et al. 2023） | ~20 万 / 21 studies | https://singlecell.mdanderson.org/TCM/ | 有余力做第二数据集 / cross-atlas |
| 全 atlas（本文构建） | 115 万 / 68 studies | Zenodo 10.5281/zenodo.12542577（排除受控数据） | **不自己跑**，报告里引用论文数字即可 |

选 TCellLandscape 的理由：公开无需申请；11 万在 4060 上舒适；它是论文真实做过的 benchmark 实验之一（Extended Data Fig 1–2），你复现的是真实验而非玩具。

> 拿到数据后先确认格式（是否已是 h5ad、`adata.X` 是原始 count 还是已归一化、有无 `layers['counts']`、`obs` 里的 batch 键叫什么、有无 cell type 注释列）。scAtlasVAE 的 ZINB 重构**需要原始 count**——若 `adata.X` 已被 log 归一化，要从 `layers` 里取回 count 或重新获取。

### 2.2 4060 上的规模现实

- 11 万细胞 + 4000 HVG，小批量（128）训练，**显存占用远低于 8GB**，无压力。
- 瓶颈只在系统内存（加载 AnnData，几 GB，一般够）和墙钟时间（约 73 epoch，几十分钟量级）。
- 若内存吃紧，**下采样到 3–5 万细胞完全不影响科学结论**——benchmark 的相对排序在这个规模上照样成立，报告里注明即可。

---

## 3. 复现范围切分：哪些手写，哪些调包，哪些引用不做

这是省时间的关键。方法论文的价值全在"方法本身"，外围一律调包。

| 模块 | 处理方式 | 说明 |
|---|---|---|
| **scAtlasVAE 核心 VAE**（编码器/解码器/ZINB/KL/分类头） | ✅ **从零手写**（L2） | 学习真正发生的地方，见 §5 |
| 三个 benchmark 任务里的**单 atlas 整合**、**标注迁移** | ✅ 重点复现 | 最能体现方法核心，数据可控 |
| scVI / scANVI 等 **baseline** | 🔧 调 `scvi-tools` | 官方现代实现，别手写 |
| Harmony / Scanorama / Seurat 等 baseline | 🔧 调各自包 | 同上 |
| **评测指标**（ASW / graph connectivity 等） | 🔧 调 `scib-metrics` | 注意与旧 scib 不可直接比 |
| 预处理（QC、HVG、归一化） | 🔧 调 `scanpy` | 外围 |
| UMAP / Leiden 聚类 | 🔧 调 `scanpy` / `umap-learn` | 外围 |
| TCR 克隆分析、STARTRAC、克隆分享 | 📖 读懂 + 复跑一小块 | 需 TCR 数据，非重点，理解即可 |
| cross-atlas 整合（多 label 对齐） | ⏳ 有余力再做 | 需两个 atlas 同时，更吃资源 |
| Tex 三亚型、GRN、生物学下游 | 📖 读懂，报告里定性讨论 | 不重写 |
| 全 atlas 训练 | ❌ 不做 | 引用论文数字 |

**一个合理的简化**：scAtlasVAE 的核心新意之一是"多个独立 cell-type predictor 实现跨 atlas 标注对齐"（代码里 `n_additional_label` / `additional_fc`）。但**在单 atlas 上只需要一个分类头**——多头只在 cross-atlas 时才有意义。所以你手写版可以只实现单个 predictor，这是合法的范围削减，不影响你理解核心机制。

---

## 4. 两周逐阶段计划

> 每个阶段都标了"产出物"，因为你的最终交付是报告——每一步都在攒报告素材。

### 阶段一｜环境 + 端到端跑通（第 1–2 天，L0→L1）
- 按 §1 建好环境 A、B，过冒烟测试。
- 下 TCellLandscape，确认格式（§2.1 的检查清单）。
- 跑通官方 `tutorial_cd8` 或 `gex_integration` 教程，确认能出 latent + UMAP。
- **产出**：环境说明 + 一张"官方流程跑通"的 UMAP 截图。别在这层超过两天。

### 阶段二｜完整数据流走通（第 3–4 天，L1）
- 不套教程，自己从 h5ad 手动走：预处理 → `scAtlasVAE(adata, batch_key="study_name")` → `fit()` → `get_latent_embedding()` → UMAP → Leiden。
- 用 `scib-metrics` 的 `Benchmarker` 对比三种 embedding：`X_pca`（未校正）、`X_scAtlasVAE`、`X_scVI`（scvi-tools 跑的 baseline）。
- **产出（报告核心图之一）**：整合前 vs 整合后 UMAP（batch 混不开 → 被校正、细胞类型分得开）+ 一张指标对比表，对照论文 Extended Data Fig 1i,j 与 Fig 2。目标是数据每次形状变化你都能解释。

### 阶段三｜核心 VAE 从零重写（第 5–9 天，L2，最硬的一块）
- **不看** `_gex_model.py`，只对着 §5 的公式↔结构，用纯 PyTorch 写最小可用版。
- 在 TCellLandscape 上训练，latent 与官方实现做**定性**对比。
- 然后**逐行对照原实现找差异**，列出差异清单。
- **产出（报告最有价值的一节）**：你的手写模型代码 + "我的实现 vs 原实现差异清单及原因"。

### 阶段四｜消融实验（第 10–11 天，L3）
- 见 §6，做 1–2 个。
- **产出**：消融结果图/表 + "作者的设计选择是否必要"的结论。

### 阶段五｜写报告 / 做 slides（第 12–14 天）
- 按 §7 模板整理。**留足三天，别压缩。**

---

## 5. 核心模块重写指南（阶段三的地图）

先建立"论文声明 → 代码实现"的对应，再动手。以下是你重写时的骨架和必须盯住的点。

### 5.1 架构对应（论文 Fig 1b ↔ 代码）

- **批不变编码器**（batch-invariant encoder）：输入**只有基因表达**，不含 batch。代码里 `encode()` 只吃 `X`（`SAE` 类，一个 MLP）。这是它与 scVI 的**本质区别**——scVI 编码器是 `F_encoder(X, B, S)`（batch-variant）。正因编码器不看 batch，query 数据才能不重训直接映射进来（zero-shot transfer）。
- **批条件解码器**（batch-variant decoder）：把 batch 信息在**解码端**注入。代码里 `decode()` 做 `torch.hstack([z, batch_embedding, ...])` 再送进解码 MLP。batch 先经一个 embedding 层（`batch_hidden_dim=8`）。
- **ZINB 重构**：解码器输出三组量参数化零膨胀负二项分布：
  - `px_rna_scale`：基因均值比例（softmax over genes）× library size → 均值 μ
  - `px_rna_rate`：离散度 θ（代码里存 logits，用时 `.exp()`）
  - `px_rna_dropout`：零膨胀门控（dropout）logits
- **分类头**（cell-type predictor）：对 latent `z` 接全连接层做交叉熵。单 atlas 只需一个头。

### 5.2 损失函数（论文 Methods 公式 ↔ 代码 `forward()`）

总损失 = 重构 + KL + 分类：

```
L = L_recon(ZINB)  +  λ_KL · KL(q(z|X) ‖ N(0,I))  +  λ_pred · L_celltype(交叉熵)
```

- KL：`kld(Normal(q_mu, q_var.sqrt()), Normal(0, 1)).sum(dim=1)`。
- **KL warmup（确定性预热）**：λ_KL 从 0 线性升到目标值，防止潜空间过早坍缩。论文里预热贯穿整个训练；你手写时让权重随 epoch 从 0 爬到 1。
- 分类损失：`CrossEntropyLoss`，且**最后 N 个 epoch 才主要训练分类头**（代码 `pred_last_n_epoch=10`）——先学好表示，再学分类。

### 5.3 训练配置（代码默认值，抄这些）

```
优化器      AdamW
学习率      5e-5
weight_decay 1e-6
n_latent    10
hidden      [128]
批大小      128（n_per_batch）
KL warmup   贯穿训练
max_epoch   min(round(20000 / N * 400), 400)
            → N≈110218 时约 73 个 epoch
random_seed 12
```

### 5.4 重写时必须盯住的"坑点"（对照时逐一核验）

这些差异全是知识点，也是报告素材：

1. **ZINB 的 dropout 是 logits 还是概率**——代码里是 logits，loss 函数内部再处理。
2. **library size 怎么进 decoder**——`px_rna_scale * lib_size.unsqueeze(1)`，即比例 × 文库大小才得到 count 尺度的均值。
3. **归一化在哪做**——ZINB 路径下，编码器输入先 `log(1+X)`（`log_variational=True`），但**重构目标是原始 count**。别把这两处搞混。
4. **reparameterize 的数值稳定**——`q_var = exp(z_var_fc(x)) + eps`（`eps=1e-4`），采样用 `Normal(q_mu, q_var.sqrt()).rsample()`。
5. **NaN 防护**——README 提到若某些细胞 total-count=0 会出 NaN；预处理时确保 `adata.X.sum(1) > 0`。

### 5.5 "代码 > 论文" 的发现（读代码才看得到，报告里单列一节）

论文正文没展开、但代码里实现了的东西——发现并理解它们本身就是复现的价值：

- **MMD loss**（`mmd_loss` / `hierarchical_mmd_loss`）：另一种 batch correction 手段，可选开启（`mmd_key`）。
- **Latent constraint**（`constrain_latent_embedding`）：可用预先算好的 PCA embedding 去约束潜空间。
- **可选 TabNet 编码器**（`EncoderType.TABNET`），而不只是普通 MLP。
- **多 batch 层级**（`n_additional_batch`）与**多 label 头**（`n_additional_label`）——对应论文的分层 batch 与跨 atlas 标注对齐。

---

## 6. 消融实验设计（阶段四，做 1–2 个即可）

目的：从"我复现了"升级到"我验证了作者的设计选择是否必要"。在你的手写版或官方版上，控制单一变量：

| 消融 | 怎么改 | 预期观察 | 揭示什么 |
|---|---|---|---|
| **潜维度** | `n_latent` 由 10 改为 2 / 50 | 2 太小信息压没了；50 未必更好 | 为何论文选 10（Ext Fig 4d 说 10/20 稳定） |
| **KL warmup** | 关掉预热（λ_KL 直接=1） | 潜空间可能坍缩、聚类变差 | 预热的作用 |
| **batch 注入位置** | 把 batch 从 decoder 挪到 encoder | 退化成 scVI 风格，迁移能力可能变差 | 编码器 batch-invariant 的意义 |

用 `scib-metrics` 量化每次消融的 batch correction / bio conservation 分数,画成对比条形图。

---

## 7. 报告结构模板（你的最终交付）

组会/复现报告，导师最看重的是**你懂不懂**，不是数字齐不齐。建议章节：

1. **背景与目标**（1 段）：论文解决什么问题、你复现的范围与约束（4060 / 2 周 / TCellLandscape 子集）。
2. **方法拆解**：scAtlasVAE 架构；**重点讲编码器 batch-invariant 与 scVI 的关键区别**；损失组成。用一张自画的结构图。
3. **复现设置**：数据、环境（含 4060 的 CUDA 处理）、超参数、baseline（scvi-tools）。
4. **结果与对照**：整合前后 UMAP、指标对比表、与论文对应图的**定性**比较（趋势对上即成功）。
5. **核心重写与发现**：你手写 VAE 的结果；"我的实现 vs 原实现"差异清单;§5.5 的"代码>论文"发现。
6. **消融结论**：设计选择是否必要。
7. **局限与诚实声明**：做了什么规模、没做什么、为什么（算力约束是完全正当的理由）；哪些结论只做了定性验证。
8. **收获与后续**：你对 VAE-based 单细胞整合方法建立的理解；若继续能做什么（cross-atlas、迁移到新数据、挑战某个生物学结论）。

---

## 8. 推荐的现代工具

| 工具 | 用途 | 为什么值得用 |
|---|---|---|
| **uv**（Astral） | Python 环境/依赖管理 | 极快，重建这套 2022 年脆弱依赖栈、隔离多环境比 conda 省心得多 |
| **scvi-tools** | scVI/scANVI 官方现代实现 | baseline 直接调，别手写；也是理解 scAtlasVAE 对照对象的最佳参考 |
| **scib-metrics**（YosefLab） | 整合评测指标（JAX/GPU） | 旧 `scib` 的现代继任者。⚠️ 数值与旧 scib 不可直接比，看相对排序 |
| **Weights & Biases / TensorBoard** | 记录 loss 曲线、跨消融对比 | 消融多次时可视化训练动态，直接产报告图；W&B 学术免费 |
| **Claude Code**（你已有） | 读仓库、对照重写、调试 | 见下方用法建议 |
| rapids-singlecell（可选） | GPU 版 scanpy | 仅当你之后想上更大规模时才需要，本次 11 万用不上 |

**Claude Code 用法建议**（针对这个任务）：
- 把 clone 的 `scAtlasVAE` 仓库和本规划文件一起打开，让它建立全局理解。
- 阶段三让它帮你**逐行 diff** 你的手写实现与 `_gex_model.py`，快速定位你想岔的地方。
- 环境阶段让它帮你排 CUDA / cublas / NaN 报错（把完整报错贴给它）。
- **但核心 VAE 的第一版一定自己写**——让它 review、不要让它代写,否则 L2 的学习价值就没了。

---

## 9. 避坑清单

- ❌ 追求数字/像素与论文完全一致 → ✅ 看结论与趋势是否一致。
- ❌ 直接用仓库锁定的 `torch==1.13.1+cu117` → ✅ 4060 上换 cu118+ 构建，先冒烟测试。
- ❌ 训练与评测挤一个环境 → ✅ 拆开（py3.8 训练 / py3.10+ 评测）。
- ❌ 把 `scib-metrics` 的绝对分数和论文表格对齐 → ✅ 只比方法间相对排序。
- ❌ 用已归一化的 `adata.X` 喂 ZINB → ✅ 确保是原始 count。
- ❌ 手写 scVI/Harmony 等 baseline → ✅ 调 scvi-tools/各自包。
- ❌ 想复现全 115 万 atlas → ✅ 用 11 万的 TCellLandscape，必要时再下采样。
- ❌ 环境阶段耗超过两天 → ✅ 装不上就上 Docker 或云卡,把时间留给 L2。
- ❌ 让 Claude Code 代写核心模型 → ✅ 自己写、让它 review。

---

## 10. 自测：怎么知道自己真学到了

做完后能不看资料回答这些，就说明到位了：

1. scAtlasVAE 的编码器为什么**不接收 batch**？这带来什么能力（相比 scVI）？
2. batch 信息具体在代码的哪个函数、以什么方式注入？
3. ZINB 的三个输出各是什么、`library size` 在哪一步乘进去？
4. KL warmup 关掉会发生什么？为什么？
5. "多个 cell-type predictor" 解决的是什么问题？单 atlas 为何用不到?
6. 你的复现 UMAP 和论文的不一样,这能说明复现失败吗?判断成功的正确标准是什么?
7. 论文正文没写、你在代码里发现的两个东西是什么?

---

*建议：先把 §1 环境和 §2 数据在第一天搞定并打勾，遇到具体报错随时找 Claude Code。整个两周里 §5（核心重写）是重心,值得多花时间。*
