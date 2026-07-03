# scAtlasVAE 复现报告

> **论文**：Xue, Wu, Tian *et al.* (2024) *Integrative mapping of human CD8⁺ T cells in inflammation and cancer.* **Nature Methods**. DOI: [10.1038/s41592-024-02530-0](https://doi.org/10.1038/s41592-024-02530-0)
> **代码**：https://github.com/WanluLiuLab/scAtlasVAE ｜ **文档**：https://scatlasvae.readthedocs.io/en/latest/
> **复现者**：HEIMA-programmer ｜ **协作**：Claude Code（"军师"）

面向**完全新手**的论文复现教学套件：既能一步步照着做，又能同时把知识框架搭起来。

---

## 1. 阅读顺序

新手请按此顺序读，先建立框架再动手：

1. [`00_overview_and_learning_map.md`](00_overview_and_learning_map.md) — **总纲**：复现是什么、你最终要达到什么、五阶段学习地图
2. [`01_concepts_and_toolbox.md`](01_concepts_and_toolbox.md) — **知识框架**：生物问题 → 数据整合 → VAE 原理 → scAtlasVAE 独特设计；工具箱总表
3. [`phase1_environment_setup.md`](phase1_environment_setup.md) 起 — **分阶段实操**

---

## 2. 文档索引与进度

| 阶段 | 内容 | 状态 | 报告 |
|---|---|---|---|
| 总纲 | 复现总纲与学习地图 | 完成 | [00_overview_and_learning_map.md](00_overview_and_learning_map.md) |
| 框架 | 知识框架与工具箱 | 完成 | [01_concepts_and_toolbox.md](01_concepts_and_toolbox.md) |
| 1 | 环境搭建（4060 / Windows / conda） | 进行中 | [phase1_environment_setup.md](phase1_environment_setup.md) |
| 2 | 端到端跑通 + scib-metrics 指标对比 | 未开始 | — |
| 3 | 核心 VAE 从零重写（重点） | 未开始 | — |
| 4 | 消融实验 | 未开始 | — |
| 5 | 汇总报告 / slides | 未开始 | — |

配套脚本在 [`../scripts/`](../scripts/)。

---

## 3. 复现目标与硬件分工

**目标**：以「从零手写核心 VAE」（L2）为必达底线，配 1–2 个消融，产出理解透彻、有独立发现的复现报告。**判断成功看结论与趋势**（batch 被校正、Tex 分三亚型、指标量级接近），不是数字/像素一致。（谱系与判据详见 `00`。）

| 角色 | 机器 | 负责 |
|---|---|---|
| 军师 | Claude Code 所在 VM（1 核 / 1GB / 无 GPU） | 读代码、写手写 VAE、写脚本/报告、排错、分析结果 |
| 执行 | 本地 **RTX 4060**（Windows + conda） | 装环境、下数据、跑训练/评测、贴回日志与图 |

---

## 4. 阶段报告的统一教学模板

后续每份阶段报告都遵循这一结构（供撰写与自查）：

1. **阶段概览** — 一段话：做什么 / 为什么 / 在整个旅程中的位置
2. **学习目标** — 明确列「完成后你将能……」
3. **会遇到的工具** — 「包速览」小方块（是什么 / 本项目干啥 / 官方文档）
4. **背景与原理** — 这一步背后的概念
5. **操作步骤** — 每步固定四段：**目的 → 命令/代码 → 预期输出 → 讲解 + 常见坑**
6. **检查点与完成标准（DoD）**
7. **自测题** — 不看资料能答上即达标
8. **记录区** — 填实际输出
9. **延伸阅读** — 权威链接

复用的提示框（引用语法）：`包速览`、`为什么这么做`、`常见坑`、`试一试`、`心态`、`深入（可选）`、`公式→代码`（阶段 3）。

---

## 5. 写作与排版规范

- **少用 emoji**：正文不用装饰性表情；状态用文字或复选框（`[ ]`/`[x]`，仅记录区/DoD）。
- **结构化**：章节编号；多用表格；提示框用引用块并加粗标签，如 `> **包速览 — PyTorch**：…`。
- **技术化排版**：标识符、路径、命令一律用 `等宽`；代码块带语言标签；行内注释尽量对齐。
- **术语规范**：专有名词首次出现附英文原词，如「潜空间 (latent space)」。
- **代码注释规范**（脚本沿用）：模块级 docstring 写明用途/用法/前置条件/预期输出；注释解释**为什么/背景/坑**，不复述代码字面；关键行注释可指回报告对应小节。

---

## 6. 已核实的关键事实（贯穿全程）

- **数据**：主力 TCellLandscape = GEO **GSE156728**，110,218 细胞 / 28 studies / 17 亚型 / 4000 HVG（论文原文数字已核对）。全 atlas（115 万，68 studies）不自己跑，引用论文。
- **算力现实**：论文 benchmark 跑在 A10(24GB) + 512GB RAM 上；11 万细胞在 4060(8GB) 上很轻松，瓶颈只在系统内存与墙钟时间。
- **模型核心**：编码器 `F(X)→z` 只吃基因表达（**batch-invariant**），解码器 `F(z,B)→ZINB` 才注入 batch——区别于 scVI `F(X,B,S)` 的本质，也是能 zero-shot 迁移的原因。
- **评测**：论文用旧 `scib`(1.1.4)，本复现用现代 `scib-metrics`；**两者数值不可直接比**，只看方法间相对排序。
