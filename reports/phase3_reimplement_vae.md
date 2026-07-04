# 阶段 3 · 核心 VAE 从零重写（L2 ★ 全项目重点）

> **阶段** 3 / 5　·　**前置**：[阶段 2 · 整合与评测](phase2_integration_and_benchmark.md)、[知识框架](01_concepts_and_toolbox.md)　·　**产出**：手写模型 `minimal_scatlasvae.py` + 差异清单　·　**预计** 5 天
> **导航**：[← 阶段 2](phase2_integration_and_benchmark.md)　·　[总纲](00_overview_and_learning_map.md)　·　[知识框架](01_concepts_and_toolbox.md)　·　[阶段 4 →](phase4_ablation_studies.md)
>
> **结果为「预期（示意）」占位**：先按手写版复现成功写完，你实跑后替换记录区。

---

## 1. 阶段概览

这是整个复现的**核心**。前两阶段是"用作者的代码"，这一阶段是**自己把核心方法重写一遍**（复现谱系里的 **L2**）——真正长本事的地方。做法：**对着论文公式和 [知识框架](01_concepts_and_toolbox.md)，用最少的纯 PyTorch 代码复刻 scAtlasVAE 的核心机制**，把"论文公式 → 代码"的每一步都打通，再与官方实现逐行对照、列出差异。

> **关于"自己写 vs 我给你写"**：最好的学习方式是**你先自己写一版**、我只做 review（复现指南 §8 反复强调这点）。但你要求我直接给出代码，所以我提供一份**参考实现** [`minimal_scatlasvae.py`](../scripts/minimal_scatlasvae.py) 供你**逐行读懂、对照、改写**——请务必把它当"精读对象"而非"复制粘贴对象"，边读边问，才有 L2 的价值。

---

## 2. 学习目标

- 把论文 Methods 的**每一个公式**对应到 PyTorch 代码；
- 理解 VAE 全套（编码器 / 重参数化 / 解码器 / ZINB / KL 预热 / 分类头）在代码里长什么样；
- 掌握复现硬功夫：**逐行对照官方实现、找出并解释每一处差异**。

---

## 3. 侦查：论文公式在源码哪里落地（公式 ↔ 代码 映射）

> 沿用侦查法：先在论文 Methods 找公式，再去官方源码（`scatlasvae/model/_gex_model.py`、`scatlasvae/utils/_loss.py`）找它落在哪个函数，最后对到我的最小实现。

![scAtlasVAE 架构图](fig_scatlasvae_architecture.svg)

*图 1 — 模型架构：批不变编码器 → 潜向量 → 批条件解码器（batch 在此注入）→ ZINB 三头；z 另接分类头。*

| 论文公式 | 官方源码位置 | 我的最小实现 |
|---|---|---|
| 编码器 $q_\phi(z\mid X)=\mathcal N(\mu,\sigma^2)$ | `_gex_model.py` `encode()`：`z_mean_fc`、`q_var=exp(z_var_fc)+1e-4` | `encode()` |
| 重参数化 $z=\mu+\sigma\,\epsilon,\ \epsilon\sim\mathcal N(0,1)$ | `Normal(q_mu,q_var.sqrt()).rsample()` | `reparameterize()` |
| 解码器 $\mu=\mathrm{softmax}\!\big(f(z,B)\big)\cdot \ell$ | `decode()`：`px_rna_scale_decoder`(含 softmax) `× lib_size` | `decode()` |
| 离散度 $\theta$、门控 $\pi$ | `px_rna_rate_decoder`、`px_rna_dropout_decoder` | `decode()` 的 `theta`,`pi` |
| 重构 $-\log p_{\text{ZINB}}(X)$ | `_loss.py` `zinb_reconstruction_loss`：`-ZINB.log_prob(X)` | `log_zinb()` |
| KL $D_{KL}\!\big(q\,\|\,\mathcal N(0,I)\big)$ | `kld(Normal(q_mu,·),Normal(0,1)).sum` | `elbo()` 里解析式 |

**总损失**（论文 Methods）：

$$\mathcal L \;=\; \underbrace{-\,\mathbb E_{q_\phi(z\mid X)}\big[\log p_\theta(X\mid z,B)\big]}_{\text{ZINB 重构}} \;+\; \lambda_{KL}\, D_{KL}\!\big(q_\phi(z\mid X)\,\big\|\,\mathcal N(0,I)\big) \;+\; \lambda_{ct}\,\mathcal L_{\text{celltype}}$$

其中 KL 权重 $\lambda_{KL}$ 在训练中**从 0 缓升**（预热）。KL 有解析式：

$$D_{KL}\!\big(\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0,1)\big)=\tfrac{1}{2}\sum_{d}\big(\sigma_d^2+\mu_d^2-1-\log\sigma_d^2\big)$$

---

## 4. 手写实现逐模块讲解

下面按 [`minimal_scatlasvae.py`](../scripts/minimal_scatlasvae.py) 的结构讲，每段都标"这对应哪条公式/哪个官方部件"。完整代码在脚本里，这里只摘关键片段。

### 4.1 批不变编码器 $F(X)\to(\mu,\sigma^2)$

**题眼**：编码器**只吃基因表达 X**，不接收 batch——这是 scAtlasVAE 区别于 scVI 的根本（见 [知识框架 §1.5](01_concepts_and_toolbox.md)）。输入先 `log1p`（`log_variational=True`），但重构目标仍是原始计数。

```python
def encode(self, x):
    h = self.encoder(torch.log1p(x))      # 输入 log1p；只有 x，没有 batch
    mu = self.z_mean(h)
    var = torch.exp(self.z_logvar(h)) + 1e-4   # 方差取 exp 保证正，+eps 数值稳定
    return mu, var
```

### 4.2 重参数化 $z=\mu+\sigma\epsilon$

把随机采样改写成对 $\mu,\sigma$ 可导的形式，梯度才能传回编码器（原理见 [知识框架 §1.4d](01_concepts_and_toolbox.md)）。

```python
@staticmethod
def reparameterize(mu, var):
    return mu + var.sqrt() * torch.randn_like(mu)   # z = μ + σ·ε
```

### 4.3 批条件解码器 $F(z,B)$ 与 ZINB 三头

batch 只在**这里**注入：把批次索引经 `nn.Embedding` 变成向量，与 z 拼接。解码器输出三头，恰好参数化 ZINB（见 [知识框架 §1.4f](01_concepts_and_toolbox.md)）。

```python
def decode(self, z, batch_index, libsize):
    h = self.decoder(torch.cat([z, self.batch_emb(batch_index)], dim=-1))
    scale = F.softmax(self.px_scale(h), dim=-1)   # 各基因占比，和为 1
    mu = scale * libsize                          # 占比 × 文库大小 = 均值 μ
    theta = torch.exp(self.px_rate(h))            # 离散度 > 0
    pi = self.px_dropout(h)                       # 零膨胀门控 logits
    return mu, theta, pi
```

### 4.4 ZINB 负对数似然（数值稳定，遵循 scVI 公式）

重构损失就是"真实计数在 ZINB 下的负对数似然"。为数值稳定，用 `softplus`/`logits` 形式分 $x=0$、$x>0$ 两种情形（与官方 `_loss.py` 用的 `ZeroInflatedNegativeBinomial` 等价）。完整实现见脚本 `log_zinb()`。

### 4.5 KL 与预热

KL 用上面的解析式；**预热**让权重逐轮从 0 升到 1，防后验坍缩（见 [知识框架 §1.4i](01_concepts_and_toolbox.md)）。

```python
kl = 0.5 * (var_z + mu_z.pow(2) - 1 - torch.log(var_z)).sum(dim=1)  # 解析 KL
...
kl_weight = min(1.0, epoch / max_epoch)   # 训练循环里：预热权重
```

### 4.6 分类头（半监督，单头）

在潜向量 z 上接一个线性层预测细胞类型；无标签细胞用 `ignore_index=-1` 跳过（见 [知识框架 §1.4h](01_concepts_and_toolbox.md)）。**单 atlas 只需一个头**——这是相对官方"多头"的合法简化。

### 4.7 训练循环（默认超参照抄论文）

`AdamW`、`lr=5e-5`、`weight_decay=1e-6`、`batch_size=128`、`seed=12`、`max_epoch=min(round(20000/N\cdot400),400)`、KL 预热贯穿训练——全部与官方默认一致。

---

## 5. 训练手写版并与官方对照

用 [`phase3_train_and_compare.py`](../scripts/phase3_train_and_compare.py) 在 TCellLandscape 上训练手写模型，得到 `obsm['X_minimal']`，再与阶段二的官方 `obsm['X_scAtlasVAE']` 对比：

- **定性**：两套嵌入各出一张 UMAP（按细胞类型上色），并排看结构是否相似；
- **定量**：算两套嵌入的 **kNN 邻域平均 Jaccard**（随机取细胞，比较它们在两套嵌入里的近邻集合有多重合）。

```powershell
conda activate scatlasvae
python phase3_train_and_compare.py
```

**预期结果（示意，待实跑替换）**：手写版能把批次混开、把主要亚型分开，UMAP 与官方**趋势一致但非逐点一致**；kNN 邻域 Jaccard **≈ 0.4–0.6**（结构相似即成功——绝不会是 1.0，因为随机种子、实现细节、浮点都会带来差异，这正常）。

**记录区（实跑后填）**：
```
手写版训练 epoch=____  最终loss=____  有无NaN=____
官方 vs 手写 kNN Jaccard=____
UMAP 定性：主要亚型是否都分开=____   批次是否混开=____
```

---

## 6. 「我的实现 vs 原实现」差异清单（本阶段最有价值的一节）

复现的价值一半在这张表——**知道自己简化了什么、为什么、有什么影响**：

| 差异点 | 官方实现 | 我的最小版 | 为什么 / 影响 |
|---|---|---|---|
| 分类头数量 | 多头 `additional_fc`（多套标签） | **单头** | 单 atlas 只需一个头；多头只在跨图谱对齐时用（见 §1.5） |
| 离散度参数化 | 可选 `gene` / `gene-batch` / `gene-cell` | 固定 `gene-cell`（每细胞每基因一个 θ） | 最通用；影响拟合灵活度，趋势不变 |
| 编码器类型 | 可选 MLP 或 **TabNet** | 仅 MLP | TabNet 是可选特性，非核心 |
| 批次层级 | 支持多层级 `n_additional_batch` | 单一 batch 键 | 单 atlas 够用 |
| MMD / latent constraint | 可选开启 | 未实现 | 可选正则；不影响核心机制 |
| 验证集/早停/lr 调度 | 有 | 简化为固定 epoch | 教学最小化；对结论趋势无碍 |
| 数值稳定细节 | 多处 eps、init 策略 | 保留关键 eps（`1e-4`/`1e-8`） | 防 NaN 的最低要求 |

> 每一行都是一个知识点，也是报告素材：**你不是"没写完"，而是做了有依据的范围削减**——把这些讲清楚，比硬凑一个全功能版更能体现理解。

---

## 7. 「代码 > 论文」发现（读代码才看得到，单列）

论文正文没展开、但源码里实现了的东西——发现并理解它们本身就是复现价值：

- **MMD loss** / `hierarchical_mmd_loss`：另一种批次校正手段，可选开启（`mmd_key`）。
- **Latent constraint**（`constrain_latent_embedding`）：可用预先算好的 PCA 嵌入约束潜空间。
- **可选 TabNet 编码器**（`EncoderType.TABNET`），而非只有普通 MLP。
- **多 batch 层级**（`n_additional_batch`）与**多 label 头**（`n_additional_label`）——对应论文的分层 batch 与跨图谱标注对齐。

这几条回答了 [总纲](00_overview_and_learning_map.md) 的"北极星"问题 7。

---

## 8. 检查点与完成标准（DoD）

- [ ] 读懂 `minimal_scatlasvae.py` 每一段对应哪条公式（能自己复述 §3 映射表）
- [ ] 手写版在 TCellLandscape 上训练不出 `NaN`，得到 `X_minimal`
- [ ] 与官方的定性 UMAP + 定量 Jaccard 对比完成，趋势一致
- [ ] 完成"差异清单"与"代码>论文"两节（用自己的话）

---

## 9. 自测题

1. 编码器为什么**不接收 batch**？这带来什么能力？batch 具体在代码哪一步注入？
2. `z = μ + σ·ε` 里，为什么要这样写而不是直接采样？（可导性）
3. ZINB 的三个输出各是什么？文库大小在哪一步乘进去？
4. 你的手写 UMAP 和官方不一样、Jaccard 不是 1.0——这说明失败了吗？判断成功的标准是什么？
5. 说出两个"论文正文没写、你在代码里发现"的东西。

---

## 10. 延伸阅读

- VAE 原始论文（Kingma & Welling, 2014, *Auto-Encoding Variational Bayes*）
- scVI 方法论文（Lopez et al., 2018, *Nature Methods*）——ZINB + 单细胞 VAE 的经典
- 官方源码：`scatlasvae/model/_gex_model.py`、`scatlasvae/utils/_loss.py`

---

> **导航**：[← 阶段 2](phase2_integration_and_benchmark.md)　·　[总纲](00_overview_and_learning_map.md)　·　[阶段 4 · 消融实验 →](phase4_ablation_studies.md)
