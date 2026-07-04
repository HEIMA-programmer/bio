# scAtlasVAE 复现报告

> **论文**：Xue, Wu, Tian *et al.* (2024) *Integrative mapping of human CD8⁺ T cells in inflammation and cancer.* **Nature Methods**. DOI: [10.1038/s41592-024-02530-0](https://doi.org/10.1038/s41592-024-02530-0)
> **代码**：https://github.com/WanluLiuLab/scAtlasVAE　·　**文档**：https://scatlasvae.readthedocs.io/en/latest/
> **复现者**：HEIMA-programmer　·　**协作**：Claude Code（"军师"）

一套**面向完全新手**的论文复现教学套件：既能一步步照着做，又能同时把知识框架搭起来。书写逻辑贯穿始终——**带你「找什么 → 怎么找 → 结论 → 怎么做 → 学到什么」**，而非直接灌输结论。

![复现全流程](fig_pipeline_overview.svg)

---

## 1. 阅读顺序

新手请按此顺序读，先建立框架再动手：

**[总纲](00_overview_and_learning_map.md) → [知识框架](01_concepts_and_toolbox.md) → [阶段 1](phase1_environment_setup.md) → [阶段 2](phase2_integration_and_benchmark.md) → [阶段 3 ★](phase3_reimplement_vae.md) → [阶段 4](phase4_ablation_studies.md) → [阶段 5](phase5_final_report.md)**

---

## 2. 文档索引与进度

| # | 内容 | 报告 | 配套脚本 / 图 |
|---|---|---|---|
| 总纲 | 复现总纲与学习地图 | [00_overview_and_learning_map.md](00_overview_and_learning_map.md) | `fig_pipeline_overview.svg` |
| 框架 | 知识框架与工具箱 | [01_concepts_and_toolbox.md](01_concepts_and_toolbox.md) | `fig_ae_vs_vae_latent_space.svg` |
| 1 | 环境搭建 + 摸清陌生库 | [phase1_environment_setup.md](phase1_environment_setup.md) | `phase1_smoke_test.py` |
| 2 | 端到端跑通 + 整合评测 | [phase2_integration_and_benchmark.md](phase2_integration_and_benchmark.md) | `phase2_*.py` · 2 图 |
| 3 ★ | 核心 VAE 从零重写 | [phase3_reimplement_vae.md](phase3_reimplement_vae.md) | `minimal_scatlasvae.py` · `phase3_train_and_compare.py` · 架构图 |
| 4 | 消融实验 | [phase4_ablation_studies.md](phase4_ablation_studies.md) | `phase4_ablations.py` · 消融图 |
| 5 | 复现汇总报告（组会稿） | [phase5_final_report.md](phase5_final_report.md) | — |

> **重要**：阶段 2–5 里所有需实跑的**数字与结果图，均为「预期（示意）」占位**——先按复现顺利写完，方便你从头通读；你在 4060 上跑出真实结果后，替换各报告的「记录区」。所有脚本在 [`../scripts/`](../scripts/)。

---

## 3. 复现目标与硬件分工

**目标**：以「从零手写核心 VAE」（**L2**）为必达底线，配 1–2 个消融，产出理解透彻、有独立发现的复现报告。**判断成功看结论与趋势**（批次被校正、Tex 分三亚型、指标量级接近、方法相对排序符合论文），不是数字/像素一致（谱系与判据详见 [总纲](00_overview_and_learning_map.md)）。

| 角色 | 机器 | 负责 |
|---|---|---|
| 军师 | Claude Code 所在 VM（无 GPU） | 读代码、写手写 VAE、写脚本/报告、排错、分析结果 |
| 执行 | 本地 **RTX 4060**（Windows + conda） | 装环境、下数据、跑训练/评测、贴回日志与图 |

---

## 4. 阶段报告的统一结构（教学模板）

每份阶段报告都遵循：**统一头部/导航 → 阶段概览 → 学习目标 → 侦查（去哪找/为什么/结论）→ 背景原理 → 操作步骤（目的→命令/代码→预期→讲解+坑）→ 检查点 DoD → 自测题 → 延伸阅读**。
提示框：`包速览` / `为什么这么做` / `常见坑` / `试一试` / `深入（可选）` / `公式→代码`。

---

## 5. 写作与排版规范

- **少 emoji**：正文不用装饰性表情；状态用文字或复选框（`[ ]`/`[x]`，仅记录区/DoD）。
- **学术排版**：章节编号；多用表格；提示框用引用块并加粗标签；标识符/路径/命令用 `等宽`；术语首现附英文原词。
- **数学**：关键公式用 LaTeX（GitHub 渲染 `$...$` / `$$...$$`），旁配大白话。
- **图**：SVG 示意图（架构/管线/结果），概念图力求准确、**结果图标注「示意/预期」**。
- **代码注释**：模块级 docstring（用途/用法/前置/预期输出）；注释解释"为什么"，不复述代码。

---

## 6. 已核实的关键事实（贯穿全程）

- **数据**：主力 TCellLandscape = GEO **GSE156728**，110,218 细胞 / 28 studies / 17 亚型 / 4000 HVG（论文原文数字已核对）。全 atlas（115 万）不自己跑，引用论文。
- **模型核心**：编码器 $F(X)\to z$ 只吃基因表达（**batch-invariant**），解码器 $F(z,B)\to$ ZINB 才注入 batch——区别于 scVI $F(X,B,S)$ 的本质，也是能 zero-shot 迁移的原因。
- **评测**：论文用旧 `scib`(1.1.4)，本复现用现代 `scib-metrics`；**两者数值不可直接比**，只看方法间相对排序。
