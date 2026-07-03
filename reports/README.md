# scAtlasVAE 复现报告

> **论文**：Xue, Wu, Tian *et al.* (2024) *Integrative mapping of human CD8⁺ T cells in inflammation and cancer.* **Nature Methods**. DOI: [10.1038/s41592-024-02530-0](https://doi.org/10.1038/s41592-024-02530-0)
> **代码**：https://github.com/WanluLiuLab/scAtlasVAE ｜ **文档**：https://scatlasvae.readthedocs.io/en/latest/
> **复现者**：HEIMA-programmer ｜ **军师**：Claude Code

## 复现目标
以「**从零手写核心 VAE**」（L2）为必达底线，配 1–2 个消融实验，产出一份*理解透彻、有独立发现*的复现报告。
**判断成功的标准是结论与趋势对得上**（batch 被校正、Tex 分三亚型、指标量级接近），不是数字/像素完全一致。

## 硬件分工
| 角色 | 机器 | 负责 |
|---|---|---|
| 🧠 军师 | Claude Code 所在的 VM（1核/1GB/无GPU） | 读代码、写手写VAE、写脚本/报告、排错、分析结果 |
| 💪 执行 | 本地 **RTX 4060**（Windows + conda） | 装环境、下数据、跑训练/评测、贴回日志与图 |

## 进度追踪
| 阶段 | 内容 | 状态 | 报告 |
|---|---|---|---|
| 1 | 环境搭建（4060/Windows/conda） | 🔄 进行中 | [phase1_environment_setup.md](phase1_environment_setup.md) |
| 2 | 端到端跑通 + scib-metrics 指标对比 | ⬜ 未开始 | — |
| 3 | 核心 VAE 从零重写 ★ | ⬜ 未开始 | — |
| 4 | 消融实验 | ⬜ 未开始 | — |
| 5 | 汇总报告 / slides | ⬜ 未开始 | — |

## 已核实的关键事实（贯穿全程）
- **数据**：主力 TCellLandscape = GEO **GSE156728**，110,218 细胞 / 28 studies / 17 亚型 / 4000 HVG（论文原文数字已核对）。全 atlas（115 万，68 studies）不自己跑，引用论文。
- **算力现实**：论文 benchmark 跑在 A10(24GB) + 512GB RAM 上；11 万细胞在 4060(8GB) 上很轻松，瓶颈只在系统内存与墙钟时间。
- **模型核心**：编码器 `F(X)→z` 只吃基因表达（**batch-invariant**），解码器 `F(z,B)→ZINB` 才注入 batch——这是它区别于 scVI `F(X,B,S)` 的本质，也是能 zero-shot 迁移的原因。
- **评测**：论文用旧 `scib`(1.1.4)，本复现用现代 `scib-metrics`；**两者数值不可直接比**，只看方法间相对排序。
