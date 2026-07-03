# 阶段一 · 环境搭建（Windows + RTX 4060 + conda）

> 目标机器：本地 **RTX 4060（8GB, Ada Lovelace 架构, 算力 sm_89）**，**Windows 原生**，用 **conda** 管理环境。
> 本文既是操作指南，也是记录（每步留了"记录区"）。遇到任何报错，把**完整报错**贴给军师。
> 先读过 [`00_overview_and_learning_map.md`](00_overview_and_learning_map.md) 和 [`01_concepts_and_toolbox.md`](01_concepts_and_toolbox.md) 会更顺。

---

## 1. 阶段概览

这一阶段要在你的 4060 上搭出一个**能真正训练 scAtlasVAE 的 Python 环境**，并用一个"冒烟测试"证明整条链路（环境 → GPU → 模型训练）是通的。

在整个复现旅程里，这是第 1 站，也是**最容易在第一天翻车**的一站——不是因为难，而是因为这个仓库锁定的依赖是 2022 年的旧版本，和你 2024 年的新显卡不匹配。本阶段的价值，一半在"把环境搭好"，另一半在**理解为什么会不匹配、怎么系统地解决**。

> **心态**：环境问题几乎人人都会遇到，卡住很正常。原则是**装不上就停下贴报错**，不要自己反复试导致环境更乱。目标是别在这一阶段耗超过两天。

---

## 2. 学习目标

完成本阶段后，你应该能说清：

- GPU 的**算力架构 (compute capability, 如 sm_89)** 和 **CUDA/PyTorch 版本**之间是什么关系，为什么它决定了你装哪个版本的 PyTorch；
- **conda** 和 **pip** 各自负责什么，为什么两者配合用；
- 什么是**冒烟测试 (smoke test)**，为什么要先用最小例子验证环境、再上真实数据；
- 本阶段三个"为什么这么做"：为什么**先装 PyTorch**、为什么用 **`--no-deps`** 装本体、为什么给 `chunked_anndata` 打**补丁**。

---

## 3. 会遇到的工具

> **包速览 — conda**：环境与包管理器。核心能力是**环境隔离**——为不同项目建互不干扰的"沙盒"，各自锁定不同的 Python 和库版本。我们用它建一个 `python=3.8` 的干净环境，避免污染系统。

> **包速览 — pip**：Python 官方装包工具。conda 建好环境后，具体的库大多用 pip 装（尤其 PyTorch 的 CUDA 版本，官方推荐用 pip 从其专用源装）。

> **包速览 — NVIDIA 驱动 / CUDA**：**驱动 (driver)** 让操作系统能驱动显卡；**CUDA** 是在 GPU 上做通用计算的平台。`nvidia-smi` 右上角显示的 `CUDA Version` 是**你的驱动所能支持的最高 CUDA 版本**（向下兼容），不是你必须用这个版本。PyTorch 的 GPU 版会**自带**它需要的 CUDA 运行时，只要驱动支持的版本 ≥ 它需要的即可。

> **包速览 — PyTorch**：深度学习框架（见 `01` 文档）。它的 GPU 版本按 CUDA 版本区分，写作 `cu117`、`cu118` 等——**选错会导致显卡用不了**，这正是本阶段的核心坑。

---

## 4. 背景与原理：为什么不能照仓库原样装

仓库的 `requirements.txt` / `setup.py` 把 PyTorch 锁死成 **`torch==1.13.1`**（默认 cu117，2022 年的构建）。而你的 4060 是 **Ada Lovelace 架构，算力 sm_89**。

关键事实：**CUDA 11.7（cu117）里没有为 sm_89 预编译的计算核 (kernel)**。后果是——`torch.cuda.is_available()` 可能返回 `True`（骗过你），但一做真正的矩阵乘法就抛：

```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE ...
```

这正是仓库 README "Common Issues" 里的第一条。**支持 Ada 的最低 CUDA 是 11.8（cu118）**，所以我们把 PyTorch 换成 `2.0.1 + cu118`。

> **为什么用 2.0.1 而不是更旧/更新**：torch 2.0.1 的 cu118 构建同时支持 Python 3.8（本仓库要求）和 sm_89。而且军师已通读源码，确认 **scAtlasVAE 没有用任何 torch 2.x 已删除的 API**，从 1.13 升到 2.0.1 是平滑的，不会引入新问题。

本阶段还有另外两个"军师提前排好的坑"（步骤里会用到）：

- **坑 A（`--no-deps`）**：`setup.py` 锁了 `torch==1.13.1`。如果直接 `pip install -e .`，pip 会为了满足这个约束，把你刚装好的新 torch **又降回 1.13.1**。所以要先把依赖装齐，再用 `pip install -e . --no-deps` 装本体（只装它自己、不动依赖）。
- **坑 B（`chunked_anndata` 补丁）**：源码顶部 `import chunked_anndata`（依赖 `tensorstore`，在 Windows + Python 3.8 上常常没有预编译包、装不上）。但军师核实过：它只在"分块读取超大图谱"的路径里才真正用到，我们用 11 万细胞的**内存加载**方式**完全用不到它**。所以打一个 3 行小补丁把这个 import 变成"可选"，彻底绕开 tensorstore。

---

## 5. 操作步骤

> 在 **Anaconda Prompt**（开始菜单可搜到）里执行 conda/pip 命令；`nvidia-smi` 用普通 PowerShell 也行。

### 步骤 0 · 前置检查

**目的**：确认显卡驱动支持我们要用的 CUDA，且 conda 已就绪。

```powershell
nvidia-smi
conda --version
```

**预期 / 讲解**：
- `nvidia-smi` 显示显卡型号、驱动版本、右上角 `CUDA Version`。只要 `CUDA Version ≥ 11.8`，cu118 就能用。
- `conda --version` 打印版本号即表示已装；报"不是内部或外部命令"则需先做步骤 1。

**本机实测记录（2026-07-03）**：
```
GPU = RTX 4060 Laptop（8GB, Ada/sm_89）   驱动 = 595.97   CUDA Version = 13.2  → 远高于 cu118 所需，稳
conda 24.11.3（已装好，步骤 1 跳过）
```

### 步骤 1 · 安装 Miniconda（若步骤 0 显示没装）

本机已装 `conda 24.11.3`，**本步骤跳过**。（若你在别的机器上从零开始：从 https://docs.conda.io/en/latest/miniconda.html 下载 Windows 64-bit 安装包，一路默认安装，然后从开始菜单打开 Anaconda Prompt。）

### 步骤 2 · 克隆 scAtlasVAE 源码

**目的**：拿到源码。我们用"可编辑安装"，这样你能随时读/改源码——阶段 3 手写 VAE 时要频繁对照它。

```powershell
git clone https://github.com/WanluLiuLab/scAtlasVAE.git
cd scAtlasVAE
```

> **常见坑**：若提示 `git 不是内部或外部命令`——装 [Git for Windows](https://git-scm.com/download/win) 后重开 Anaconda Prompt；或网页点 `Code → Download ZIP` 解压后 `cd` 进去。

### 步骤 3 · 创建并激活训练环境

**目的**：建一个隔离的 Python 3.8 沙盒，之后所有操作都在里面进行。

```powershell
conda create -n scatlasvae python=3.8 -y
conda activate scatlasvae
```

**预期**：激活后命令行前缀从 `(base)` 变成 `(scatlasvae)`。**此后所有 pip 命令都要在这个前缀下执行。**

### 步骤 4 · 先装支持 4060 的 PyTorch（cu118）

**目的**：装对显卡能用的 PyTorch。**顺序很重要——先装它**，后面装依赖时才不会被牵连降级。

```powershell
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**讲解**：`--index-url .../cu118` 指定从 PyTorch 的 cu118 专用源下载。约 2GB，稍慢。装完先别急着装别的，先做步骤 5 验证。

### 步骤 5 · CUDA 冒烟测试（关键检查点，必须通过）

**目的**：在装其他任何东西之前，先证明"PyTorch + 4060"这一层是通的。

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); x=torch.randn(2048,2048,device='cuda'); print('matmul OK:', float((x@x).mean())); print(torch.cuda.get_device_name(0))"
```

**预期**：打印类似 `2.0.1+cu118 11.8 True` → `matmul OK: ...` → `NVIDIA GeForce RTX 4060 ...`，且**不报 cuBLAS 错**。

> **讲解**：那句 `x @ x`（矩阵乘法）会真正调用 cuBLAS 库——如果 PyTorch 装成了不支持 sm_89 的 cu117，这一步就会抛 `CUBLAS_STATUS_INVALID_VALUE`。能过，就说明 PyTorch 换对了。

> **常见坑**：这里报 `CUBLAS_STATUS_INVALID_VALUE` 或 `sm_89 is not compatible` → PyTorch 没装成 cu118。停下贴报错，别继续。

**记录区**：
```
torch 版本/CUDA/可用：______
matmul：______   GPU：______
```

### 步骤 6 · 安装其余依赖

**目的**：按仓库要求的版本装齐科学计算依赖（不含 torch，不含 tensorstore）。

```powershell
pip install "anndata==0.8.0" "scanpy==1.8.1" "scirpy==0.10.1" "numpy==1.21.6" "numba==0.57.1" "scikit-learn==0.24.1" "umap-learn==0.5.1" "einops==0.4.1" "seaborn==0.12.2" "pandas==1.4.2" "matplotlib==3.5.2" "biopython==1.79" "tabulate==0.9.0" "plotly==5.10.0"
```

> **常见坑**：若某个包报"无法编译 / 没有 wheel"（Windows 上偶发），把报错贴给军师，换成 conda-forge 预编译版或放宽版本——不要自己硬刚版本号。

### 步骤 7 · 给 `chunked_anndata` 打补丁（坑 B）

**目的**：把源码顶部那个会依赖 tensorstore 的硬导入，改成"装不上就跳过"的可选导入。

在 `scAtlasVAE` 仓库根目录（`(scatlasvae)` 已激活）执行：

```powershell
python -c "import pathlib; p=pathlib.Path('scatlasvae/model/_gex_model.py'); s=p.read_text(encoding='utf-8'); s=s.replace('import chunked_anndata as ca', 'try:\n    import chunked_anndata as ca\nexcept ModuleNotFoundError:\n    ca = None  # 仅 chunked_adata_path 路径用到；in-memory 工作流不需要', 1); p.write_text(s, encoding='utf-8'); print('patched OK')"
```

**预期**：打印 `patched OK`。（这条命令做的事等价于：把 `import chunked_anndata as ca` 这一行改成 `try/except` 包裹的版本。军师已在源码副本上验证过它不会破坏语法。）

### 步骤 8 · 安装 scAtlasVAE 本体（`--no-deps`，坑 A）

**目的**：装模型本体，且**不让它把 torch 降回 1.13.1**。

```powershell
pip install -e . --no-deps
```

**讲解**：`-e` 是"可编辑安装"（源码改动即时生效）；`--no-deps` 表示"只装它自己，不碰依赖"——因为依赖我们已在步骤 4、6 亲手装好、版本可控。

### 步骤 9 · 跑完整冒烟测试（本阶段收尾）

**目的**：用一份极小的合成数据，验证"环境 + GPU + 模型训练 + 取 latent"整条链路。

把军师写的 [`scripts/phase1_smoke_test.py`](../scripts/phase1_smoke_test.py) 拷到本机运行：

```powershell
python phase1_smoke_test.py
```

**预期**：依次打印 `[1/4]`~`[4/4]`，最后出现
`冒烟测试全部通过：环境 + GPU + 模型训练链路完全 OK`。

> **常见坑**：若 `import scatlasvae` 报 `No module named 'xxx'`——多半是某个小依赖漏装，一条 `pip install xxx` 补上即可，并告诉军师。

**记录区**：
```
phase1_smoke_test.py 结果：[ ] 通过   [ ] 报错(贴下方)
______
```

---

## 6. 检查点与完成标准（DoD）

同时满足即算阶段一完成，可进入阶段二：

- [ ] 步骤 5：`torch.cuda.is_available()` 为 `True`，GPU 矩阵乘法不报 cuBLAS 错
- [ ] 步骤 9：`import scatlasvae` 成功，且合成数据能 `fit()` 并 `get_latent_embedding()` 得到 `(512, 10)` 的 latent

---

## 7. 自测题（能答上说明这一阶段真的懂了）

1. 为什么 4060 不能用仓库锁定的 `torch==1.13.1(cu117)`？换成 cu118 解决了什么？
2. `nvidia-smi` 显示 `CUDA Version 13.2`，这是否意味着我必须装 CUDA 13.2 的 PyTorch？为什么？
3. 为什么要先装 PyTorch、最后再用 `--no-deps` 装 scAtlasVAE 本体？
4. `chunked_anndata` 的补丁绕过了什么？我们为什么可以安全地绕过它？

---

## 8. 附录 · 评测环境 B（阶段二才用，可先跳过）

评测用的 `scib-metrics` 要 Python ≥ 3.10，与训练环境（3.8）冲突，所以**单独建一个环境**。阶段二开始时再细化，这里先知道有这么回事：

```powershell
conda create -n scib python=3.10 -y
conda activate scib
pip install scib-metrics scanpy scvi-tools
```

---

## 9. 延伸阅读

- PyTorch 本地安装（按 CUDA 版本选）：https://pytorch.org/get-started/locally/
- NVIDIA 各 GPU 的算力（compute capability）对照：https://developer.nvidia.com/cuda-gpus
- conda 环境管理入门：https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
- 仓库 README 的 Common Issues：https://github.com/WanluLiuLab/scAtlasVAE

---

## 10. 阶段一小结（跑通后补）

- 最终环境：`scatlasvae`（py3.8, torch 2.0.1+cu118, scAtlasVAE 可编辑安装）
- 三处关键改动：① torch 换 cu118；② `--no-deps` 装本体；③ `chunked_anndata` 补丁
- 冒烟测试结果：______
