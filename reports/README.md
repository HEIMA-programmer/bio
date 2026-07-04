# scAtlasVAE 复现报告

> **论文**：Xue, Wu, Tian *et al.* (2024) *Integrative mapping of human CD8⁺ T cells in inflammation and cancer.* **Nature Methods**. DOI: [10.1038/s41592-024-02530-0](https://doi.org/10.1038/s41592-024-02530-0)
> **代码**：https://github.com/WanluLiuLab/scAtlasVAE　·　**文档**：https://scatlasvae.readthedocs.io/en/latest/
> **复现者**：HEIMA-programmer　·　**协作**：Claude Code（"军师"）

一套**面向完全新手**的论文复现教学套件。它和普通教程最大的不同，是贯穿始终的一条规矩——**不直接灌结论，而是每条结论都先带你走一遍"我是怎么找到它的"**：

> 为什么要知道这个 → 去哪找 → 怎么动手查（命令 / 看论文哪张图 / 打开源码哪一行）→ 你会看到什么 → 门道 → 结论

你既能一步步照着做，又能在探索中学会**方法论**（怎么摸清一个陌生库、怎么把论文公式翻译成代码、怎么验证设计选择），并真正学懂这个项目。

---

## 1. 阅读顺序

新手请按此顺序读，先建立框架再动手：

**[总纲](00_overview_and_learning_map.md) → [知识框架](01_concepts_and_toolbox.md) → [阶段 1](phase1_environment_setup.md) → [阶段 2](phase2_integration_and_benchmark.md) → [阶段 3 ★](phase3_reimplement_vae.md) → [阶段 4](phase4_ablation_studies.md) → [阶段 5](phase5_final_report.md)**

---

## 2. 文档索引

| # | 内容 | 报告 | 配套脚本 / 图 |
|---|---|---|---|
| 总纲 | 先探索再上路：读论文 Fig1、走仓库树、推出复现路线 | [00_overview_and_learning_map.md](00_overview_and_learning_map.md) | `fig_paper_story` · `fig_repo_map` · `fig_pipeline_overview` |
| 框架 | 从生物问题到 VAE 的直觉 + 工具箱 | [01_concepts_and_toolbox.md](01_concepts_and_toolbox.md) | `fig_ae_vs_vae_latent_space` · `fig_zinb_construction` |
| 1 | 环境搭建 + 摸清陌生库 | [phase1_environment_setup.md](phase1_environment_setup.md) | `phase1_smoke_test.py` |
| 2 | 端到端跑通 + 整合评测 | [phase2_integration_and_benchmark.md](phase2_integration_and_benchmark.md) | `phase2_*.py` · 3 图 |
| 3 ★ | 核心 VAE 从零重写（含源码逐行走读） | [phase3_reimplement_vae.md](phase3_reimplement_vae.md) | `minimal_scatlasvae.py` · `phase3_train_and_compare.py` · 5 图 |
| 4 | 消融实验 | [phase4_ablation_studies.md](phase4_ablation_studies.md) | `phase4_ablations.py` · 2 图 |
| 5 | 复现汇总报告（组会稿） | [phase5_final_report.md](phase5_final_report.md) | 引用全部图 |

> **重要**：阶段 2–5 里所有需实跑的**数字与结果图，均为「预期（示意）」占位**——军师所在机器无 GPU/数据，先按复现顺利写完方便你从头通读；你在 4060 上跑出真实结果后，替换各报告的「记录区」。所有脚本在 [`../scripts/`](../scripts/)。

---

## 3. 复现目标与硬件分工

**目标**：以「从零手写核心 VAE」（**L2**）为必达底线，配 1–2 个消融，产出理解透彻、有独立发现的复现报告。**判断成功看结论与趋势**（批次被校正、Tex 分三亚型、指标量级接近、方法相对排序符合论文），不是数字/像素一致（谱系与判据详见 [总纲](00_overview_and_learning_map.md)）。

| 角色 | 机器 | 负责 |
|---|---|---|
| 军师 | Claude Code 所在 VM（无 GPU） | 读代码、带走读源码、写手写 VAE、写脚本/报告、生成配图、排错、分析结果 |
| 执行 | 本地 **RTX 4060**（Windows + conda） | 装环境、下数据、跑训练/评测、贴回日志与图 |

---

## 4. 配图

全部 16 张配图由 [`../scripts/figgen/`](../scripts/figgen/) 用 **matplotlib** 程序化生成（中文经字体探测 + 矢量路径嵌入，任何机器打开都不掉字）：`theme.py`（设计系统）、`build_structures.py`（结构/架构/概念图）、`build_data.py`（数据/定量图，均标注「示意/预期」）。重跑：

```bash
cd ../scripts/figgen && python3 build_structures.py && python3 build_data.py
```

---

## 5. 写作与排版规范

- **探索优先**：任何关于论文/仓库/代码的事实，先给"去哪找/怎么查/看到什么"，再给结论。带 `文件:行号` 摘录真实源码。
- **少 emoji**：正文不用装饰性表情；状态用文字或复选框（`[ ]`/`[x]`，仅记录区/DoD）。
- **学术排版**：章节编号；多用表格；提示框用引用块并加粗标签；标识符/路径/命令用 `等宽`；术语首现附英文原词。
- **数学**：关键公式用 LaTeX（GitHub 渲染 `$...$`/`$$...$$`），旁配大白话。
- **代码注释**：模块级 docstring（用途/用法/前置/预期）；注释解释"为什么"，不复述代码。
