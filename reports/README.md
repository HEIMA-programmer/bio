# scAtlasVAE 复现报告

> **论文**：Xue, Wu, Tian *et al.* (2024) *Integrative mapping of human CD8⁺ T cells in inflammation and cancer.* **Nature Methods**. DOI: [10.1038/s41592-024-02530-0](https://doi.org/10.1038/s41592-024-02530-0)
> **代码**：https://github.com/WanluLiuLab/scAtlasVAE
> **文档**：https://scatlasvae.readthedocs.io/en/latest/
> **复现者**：HEIMA-programmer　·　**协作**：Claude Code（"军师"）

一套**面向完全新手**的论文复现教学套件。它和普通教程最大的不同，是贯穿始终的一条规矩——**不直接灌结论，而是每条结论都先带你走一遍"我是怎么找到它的"**：

> 为什么要知道这个 → 去哪找 → 怎么动手查（命令 / 看论文哪张图 / 打开源码哪一行）→ 你会看到什么 → 门道 → 结论

你既能一步步照着做，又能在探索中学会**方法论**（怎么摸清一个陌生库、怎么把论文公式翻译成代码、怎么验证设计选择），并真正学懂这个项目。

---

## 1. 阅读顺序

新手请按此顺序读，先建立框架再动手：

**[总纲](00_overview_and_learning_map.md) → [知识框架](01_concepts_and_toolbox.md) → [阶段 1](phase1_environment_setup.md) → [阶段 2](phase2_integration_and_benchmark.md) → [阶段 3 ★](phase3_reimplement_vae.md) → [阶段 4](phase4_ablation_studies.md) → [阶段 5 深入验证](phase5_deeper_validation.md) → [阶段 6 汇总](phase6_final_report.md)**

---

## 2. 文档索引

| # | 内容 | 报告 | 配套脚本 / 图 |
|---|---|---|---|
| 总纲 | 先探索再上路：读论文 Fig1、走仓库树、推出复现路线 | [00_overview_and_learning_map.md](00_overview_and_learning_map.md) | Mermaid 示意图 |
| 框架 | 从生物问题到 VAE 的直觉 + 工具箱 | [01_concepts_and_toolbox.md](01_concepts_and_toolbox.md) | Mermaid 示意图 |
| 1 | 环境搭建 + 摸清陌生库 | [phase1_environment_setup.md](phase1_environment_setup.md) | `phase1_smoke_test.py` |
| 2 | 端到端跑通 + 整合评测 | [phase2_integration_and_benchmark.md](phase2_integration_and_benchmark.md) | `phase2_*.py` · 结果图 |
| 3 ★ | 核心 VAE 从零重写（含源码逐行走读） | [phase3_reimplement_vae.md](phase3_reimplement_vae.md) | `minimal_scatlasvae.py` · `phase3_train_and_compare.py` |
| 4 | 消融实验 | [phase4_ablation_studies.md](phase4_ablation_studies.md) | `phase4_ablations.py` |
| 5 | 深入验证与扩展：Task1四方 / Task3注释迁移 / Task2跨图谱对齐 / 批不变探针 / 可扩展性 / 手写VAE上标尺 / 指标对照 | [phase5_deeper_validation.md](phase5_deeper_validation.md) | `phase5_*.py` · 结果图 |
| 6 | 复现汇总报告（组会稿，含深入验证摘要） | [phase6_final_report.md](phase6_final_report.md) | 引用全部图 |

> **状态（2026-07 更新）**：阶段 1–6 已在**本机 RTX 4060 上真实跑通**，各报告的「记录区」、指标表、结果图均为**真实数据**（数据 = GSE156728 的 10X CD8 **全量 ~10.5 万细胞**，与论文 TCellLandscape 同量级；baseline 用 scVI，scvi-tools 在 Windows 需先开长路径才能装，另附 Harmony 可选第二基线；Task 2 跨图谱另用 Yost 2019 GSE123813 作第二图谱）。真实产物见 `../data/`（h5ad/npz/csv，gitignore 不入库），真实结果图由 `../scripts/figgen/build_real.py` 生成于 [`figures/`](figures/)。所有脚本在 [`../scripts/`](../scripts/)。

---

## 3. 复现目标与硬件分工

**目标**：以「从零手写核心 VAE」（**L2**）为必达底线，配 1–2 个消融，产出理解透彻、有独立发现的复现报告。**判断成功看结论与趋势**（批次被校正、Tex 分三亚型、指标量级接近、方法相对排序符合论文），不是数字/像素一致（谱系与判据详见 [总纲](00_overview_and_learning_map.md)）。

| 角色 | 机器 | 负责 |
|---|---|---|
| 军师 | Claude Code 所在 VM（无 GPU） | 读代码、带走读源码、写手写 VAE、写脚本/报告、生成配图、排错、分析结果 |
| 执行 | 本地 **RTX 4060**（Windows + conda） | 装环境、下数据、跑训练/评测、贴回日志与图 |

---

## 4. 配图

图分两类，各按最合适的方式呈现（既好看又不让一堆 SVG 散落仓库）：

- **示意 / 流程 / 概念图** → 用 [Mermaid](https://mermaid.js.org/) 代码块**直接内嵌**在 `.md` 里，GitHub 原生渲染、零文件、可 diff、可直接改文字。
- **实验结果图**（UMAP、条形、曲线、混淆矩阵、探针）→ 由 [`../scripts/figgen/build_real.py`](../scripts/figgen/build_real.py) 从**真实实跑产物**生成为 PNG，集中放 [`figures/`](figures/)。`theme.py` 负责中文字体。重跑：

```bash
cd ../scripts/figgen && python build_real.py all   # 在 scib 环境
```

> 为何不用 data-URI 内嵌图：GitHub Markdown 会拦截 `data:` 图片（camo/CSP）导致裂图；所以**结果图走 PNG 文件、示意图走 Mermaid**是 GitHub 上都能渲染的稳妥组合。`build_structures.py` / `extract_sc_font.py`（旧的手绘示意图生成器）已随示意图转 Mermaid 而弃用。

---

## 5. 写作与排版规范

- **探索优先**：任何关于论文/仓库/代码的事实，先给"去哪找/怎么查/看到什么"，再给结论。带 `文件:行号` 摘录真实源码。
- **少 emoji**：正文不用装饰性表情；状态用文字或复选框（`[ ]`/`[x]`，仅记录区/DoD）。
- **学术排版**：章节编号；多用表格；提示框用引用块并加粗标签；标识符/路径/命令用 `等宽`；术语首现附英文原词。
- **数学**：关键公式用 LaTeX（GitHub 渲染 `$...$`/`$$...$$`），旁配大白话。
- **代码注释**：模块级 docstring（用途/用法/前置/预期）；注释解释"为什么"，不复述代码。
