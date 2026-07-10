# 阶段一 · 环境搭建 + 学会"摸清一个陌生库"（Windows + RTX 4060 + conda）

> **阶段** 1 / 5　·　**前置**：[总纲](00_overview_and_learning_map.md)、[知识框架](01_concepts_and_toolbox.md)　·　**产出**：可训练环境 + 冒烟测试　·　**预计** 1–2 天
> **导航**：[总纲](00_overview_and_learning_map.md)　·　[索引](README.md)　·　[阶段 2 →](phase2_integration_and_benchmark.md)
>
> 目标机器：本地 **RTX 4060（8GB, Ada Lovelace 架构, 算力 sm_89）**，**Windows 原生**，用 **conda** 管理环境。

---

## 1. 阶段概览：这一阶段有两个目标

- **目标 A（操作）**：在你的 4060 上搭出一个能真正训练 scAtlasVAE 的环境，并用冒烟测试证明整条链路通。
- **目标 B（能力，同样重要）**：学会一套**"拿到一个陌生的库，如何自己摸清它"**的通用方法。

> **为什么强调目标 B**：复现的核心能力，不是记住"scAtlasVAE 把 torch 锁成 1.13.1"这种结论，而是**下次拿到任何一个陌生库，你都知道去哪看、看什么、怎么判断**。所以本报告里每一个"事实"，我都会先告诉你**我是怎么找到它的**（§3、§4），你完全可以打开仓库跟着自己查一遍，再进入动手安装（§6）。

---

## 2. 学习目标

完成本阶段后你应该能：

- 复述一套"侦查陌生库"的通用清单：**去哪看、看什么、为什么**（§3）；
- 自己动手在 scAtlasVAE 仓库里**查证**关键事实（依赖版本、能否在你显卡上跑、入口 API、隐藏依赖）（§4）；
- 说清 GPU 算力（如 sm_89）与 CUDA / PyTorch 版本的关系，并会自己查；
- 理解 conda / pip 的分工，会用冒烟测试验证环境。

---

## 3. 通用方法：拿到一个陌生的库，先看这六个地方

不管以后遇到什么 Python 库，开局都问同样六个问题、去同样几个地方找答案：

| 看哪里 | 回答什么问题 | 在哪打开 |
|---|---|---|
| **README** | 这库是干嘛的？怎么装？怎么用？有没有已知坑？怎么引用？ | 仓库首页会自动显示；或根目录 `README.md` |
| **依赖文件** | 依赖哪些包、锁死了哪些版本？支持哪个 Python？ | `requirements.txt` / `setup.py`（`install_requires`）/ `environment.yml` / `pyproject.toml` |
| **文档 (docs)** | 有没有教程、API 说明、入门例子？ | `docs/` 目录 或 README 里的 readthedocs 链接 |
| **源码（包目录）** | 核心类叫什么、怎么调、默认参数多少、有哪些关键方法？ | 包目录下的 `.py`；从 `__init__.py` 找它导出了什么 |
| **Issues / Changelog** | 别人踩过什么坑？最近改了什么？ | GitHub 的 Issues 标签页 / README 的 Change Log / `CHANGELOG` |
| **你自己的硬件/系统** | 这库（尤其锁定的深度学习版本）能在我的机器上跑吗？ | 交叉核对：你的 GPU 算力 ↔ 它要求的 CUDA（见 §4.3） |

**怎么在源码里"找东西"（三种办法，任选）**：

1. **在 GitHub 上浏览**（对新手最省事）：打开仓库页，直接点开文件；按键盘 `t` 可模糊搜文件名，`.` 可在浏览器里打开 VSCode 网页版全局搜索。
2. **本地看**（你 §6 步骤 2 会 clone 下来）：用 VSCode 打开文件夹，`Ctrl+Shift+F` 全局搜索关键字。
3. **命令行搜**：Windows 用 `findstr /s /n "关键字" *.py`（相当于搜索），Linux/Mac 用 `grep -rn "关键字"`。

> **心态**：你不需要读懂整个仓库。**带着上面六个问题去"定点侦查"**，找到答案就走——这正是我给你整理这些报告时做的事。

---

## 4. 实战：用这套方法侦查 scAtlasVAE（每条都含"怎么找到的"）

下面每一小节都是同一个套路：**为什么找 → 去哪找 → 怎么找（你可自己动手） → 找到什么 → 结论**。仓库地址：https://github.com/WanluLiuLab/scAtlasVAE 。

### 4.1 它是干嘛的 / 怎么装 / 有没有已知坑 —— 看 README

- **为什么找**：README 是一个库的"说明书"，开局第一份必读。
- **去哪 / 怎么找**：打开仓库首页往下滚（首页展示的就是 `README.md`）。
- **找到什么**：① 一句话定位——"用于 atlas 级大规模 scRNA-seq 数据整合与查询数据迁移"；② 安装方式——`conda env create -f environment.yml`、`pip3 install scatlasvae`，装 PyTorch 用的是 `torch==1.13.1+cu117`；③ 一节 **"Common Issues"**，里面第一条正是 `CUDA error: CUBLAS_STATUS_INVALID_VALUE`，第二条是 `fit()` 出 `nan`；④ 作者说测试过的显卡是 **2080Ti / 3090Ti / A10 / A100 / A800**——注意**没有 40 系**。
- **结论**：这库是做单细胞整合的；官方装法把 PyTorch 锁在 `cu117`；README 自己就埋了 cuBLAS 报错的提示。"没测过 40 系" + "cuBLAS 报错" 是我们后面要换 PyTorch 的**第一条线索**。

### 4.2 它锁死了哪些版本 / 支持哪个 Python —— 看依赖文件

- **为什么找**：依赖的版本决定环境怎么建、会不会和你的硬件或别的包冲突。
- **去哪 / 怎么找**：打开根目录的 `requirements.txt`、`setup.py`（看里面的 `install_requires`）、`environment.yml`。
- **找到什么**：`torch==1.13.1`、`torchvision==0.14.1`；`python=3.8`；`scanpy==1.8.1`、`numpy==1.21.6`、`numba==0.57.1`、`scikit-learn==0.24.1`…；`environment.yml` 里带 `--extra-index-url https://download.pytorch.org/whl/cu117`。
- **结论**：这是一套 **2022 年的旧依赖栈**，而且 **PyTorch 被死锁在 1.13.1（cu117）**。先把"torch 被锁在 cu117"这个事实记下，去 §4.3 判断它跟你的显卡冲不冲突。

### 4.3 它能在我的 4060 上跑吗 —— 交叉核对（这一步最能学到东西）

- **为什么找**：**库锁定的 PyTorch 版本，不一定支持你的新显卡。** 必须交叉核对两件事：你的 GPU 需要多高的 CUDA、这个 torch 提供多高的 CUDA。
- **怎么查"你的 GPU 需要什么"**：
  1. 查你 GPU 的**算力 (compute capability)**：RTX 4060 属于 **Ada Lovelace** 架构，算力 **sm_89**。去哪查——NVIDIA 官方 [CUDA GPUs 页](https://developer.nvidia.com/cuda-gpus)，或直接搜 `RTX 4060 compute capability`。
  2. 查"支持 sm_89 的最低 CUDA"：**Ada / sm_89 需要 CUDA ≥ 11.8**。去哪查——搜 `Ada Lovelace CUDA 11.8 sm_89`，或看 PyTorch/NVIDIA 的 release notes。
- **怎么查"这个 torch 提供什么"**：§4.2 已知它锁 `torch 1.13.1`，默认构建是 **cu117（CUDA 11.7）**，而 **cu117 最高只预编译到 sm_86**，没有 sm_89 的计算核。
- **结论**：`cu117 (11.7) < 11.8` → **不支持 4060** → 一做 GPU 矩阵乘法就会抛 `CUBLAS_STATUS_INVALID_VALUE`（正好对上 §4.1 README 里那条 Common Issues）。所以**必须把 PyTorch 换成 cu118 的构建**（我们用 `torch==2.0.1+cu118`）。这条完整推理链，就是"否掉 README 装法、改用 cu118"的依据。
- **装完怎么当场验证**：`torch.cuda.get_device_capability()` 应返回 `(8, 9)`（即 sm_89）；再跑一次 GPU 矩阵乘法看报不报错——这就是 §6 步骤 5 的冒烟测试。

### 4.4 入口 API 与默认超参 —— 读源码里的类定义

- **为什么找**：要用这个库，得知道**入口类叫什么、怎么调、默认值是多少**（深度学习库的默认值往往就是论文用的超参，阶段 3 手写要对照）。
- **怎么找（顺藤摸瓜）**：
  1. 打开 `scatlasvae/__init__.py` → 看到它 `from . import model`；
  2. 打开 `scatlasvae/model/__init__.py` → 看到 `from ._gex_model import scAtlasVAE`——**核心文件就是 `_gex_model.py`**；
  3. 在 `_gex_model.py` 里搜 `class scAtlasVAE`、`def __init__`、`def fit`、`def get_latent`，读它们的参数默认值。（GitHub 按 `t` 搜文件、文件内 `Ctrl/Cmd+F`；本地 `findstr /n "def fit" _gex_model.py`。）
- **找到什么**：构造函数是 **keyword-only**（`def __init__(self, *, adata=..., batch_key=..., label_key=...)`——所以必须写 `adata=adata`）；默认 `reconstruction_method='zinb'`、`n_latent=10`、`hidden=[128]`、`batch_hidden_dim=8`；`fit` 默认 `lr=5e-5`、`batch_size=128`、`random_seed=12`、`max_epoch=min(round(20000/N×400), 400)`；取隐向量用 `get_latent_embedding()`。
- **结论**：这些就是冒烟测试脚本和阶段 3 手写对照要用的 API 与默认超参——全是从源码**读**出来的，不是背来的。你自己也能这样读任何库的入口类。

### 4.5 有没有会卡安装的隐藏依赖 —— 搜源码的 import

- **为什么找**：有些库在源码顶部 `import` 一些非标准包，`pip` 装本体时可能带不全、或在你的系统根本装不上，导致 `import` 就崩。提前扫一遍能防患未然。
- **怎么找**：在源码里搜行首的 `import` / `from`，特别留意不认识的名字。这里在 `_gex_model.py:17` 搜到 `import chunked_anndata`；再搜它实际在哪用（搜 `ca.`），发现只在"分块读取超大数据"的分支里出现。
- **结论**：我们用**内存加载**（`adata=...`）根本走不到那些代码，而它依赖的 `tensorstore` 在 Windows + py3.8 上常常装不上——所以把这行 `import` 改成**可选**（try/except 置空），彻底绕开（见 §6 步骤 7）。

> 到这里，你已经把 §6 里每一步"为什么这么做"的来龙去脉都侦查清楚了。下面才是动手。

---

## 5. 会遇到的工具（包速览）

> **包速览 — conda**：环境与包管理器。核心是**环境隔离**——为不同项目建互不干扰的"沙盒"。我们用它建一个 `python=3.8` 的干净环境。

> **包速览 — pip**：Python 官方装包工具。conda 建好环境后，具体的库大多用 pip 装（尤其 PyTorch 的 CUDA 版）。

> **包速览 — NVIDIA 驱动 / CUDA**：**驱动**让系统能驱动显卡；**CUDA** 是在 GPU 上做通用计算的平台。`nvidia-smi` 右上角的 `CUDA Version` 是**驱动支持的最高 CUDA**（向下兼容），不是你必须用它。PyTorch 的 GPU 版**自带**它需要的 CUDA 运行时，只要驱动 ≥ 它需要的即可。

> **包速览 — PyTorch**：深度学习框架。GPU 版本按 CUDA 版本区分（`cu117`、`cu118`…）——**选错显卡就用不了**，这正是 §4.3 的核心。

---

## 6. 动手：搭建环境（在 Anaconda Prompt 里执行）

下面每一步都是 §4 侦查结论的"落地"。命令照抄即可；`nvidia-smi` 用普通 PowerShell 也行。

### 步骤 0 · 前置检查

**目的**：确认显卡驱动支持我们要用的 CUDA，且 conda 已就绪。

```powershell
nvidia-smi
conda --version
```

**预期**：`nvidia-smi` 显示显卡、驱动、右上角 `CUDA Version`（只要 ≥ 11.8 即可用 cu118）；`conda --version` 打印版本号即已装。

**本机实测记录（2026-07-03）**：
```
GPU = RTX 4060 Laptop（8GB, Ada/sm_89）   驱动 = 595.97   CUDA Version = 13.2  → 远高于 cu118 所需
conda 24.11.3（已装好，步骤 1 跳过）
```

### 步骤 1 · 安装 Miniconda（若步骤 0 显示没装）

本机已装 `conda 24.11.3`，**跳过**。（从零开始的话：到 https://docs.conda.io/en/latest/miniconda.html 下 Windows 64-bit 安装包，一路默认，然后从开始菜单开 Anaconda Prompt。）

### 步骤 2 · 克隆源码（也方便你按 §3/§4 亲手侦查）

```powershell
git clone https://github.com/WanluLiuLab/scAtlasVAE.git
cd scAtlasVAE
```

> 没装 git？装 [Git for Windows](https://git-scm.com/download/win) 后重开 Anaconda Prompt；或网页 `Code → Download ZIP` 解压后 `cd` 进去。

### 步骤 3 · 建并激活训练环境（Python 3.8，依据 §4.2）

```powershell
conda create -n scatlasvae python=3.8 -y
conda activate scatlasvae
```

**预期**：命令行前缀从 `(base)` 变成 `(scatlasvae)`。此后所有 pip 命令都在这个前缀下执行。

### 步骤 4 · 装支持 4060 的 PyTorch（cu118，依据 §4.3；先装它）

```powershell
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

约 2GB，稍慢。装完先做步骤 5 验证，别急着装别的。

### 步骤 5 · CUDA 冒烟测试（关键检查点，必须通过）

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('算力=', torch.cuda.get_device_capability()); x=torch.randn(2048,2048,device='cuda'); print('matmul OK:', float((x@x).mean())); print(torch.cuda.get_device_name(0))"
```

**预期**：`2.0.1+cu118 11.8 True` → `算力= (8, 9)`（正是 §4.3 说的 sm_89）→ `matmul OK: ...` → `NVIDIA GeForce RTX 4060 ...`，且**不报 cuBLAS 错**。

> **常见坑**：这里报 `CUBLAS_STATUS_INVALID_VALUE` 或 `sm_89 is not compatible` → torch 没装成 cu118，停下贴报错。

**记录区（本机实测 2026-07-09）**：
```
torch/CUDA/可用：torch 2.0.1+cu118 / CUDA 11.8 / True   算力：(8, 9)   matmul：OK   GPU：NVIDIA GeForce RTX 4060 Laptop GPU
```
> 本机安装小记：`download.pytorch.org` 直连一度 `Read timed out`（**网络质量问题**，与是否在国内无关）。加大 `--timeout 300 --retries 10` 后经 `--extra-index-url https://download.pytorch.org/whl/cu118` 装成；网络实在差可退到镜像 `-f https://mirrors.aliyun.com/pytorch-wheels/cu118/`。装到的是 `2.0.1+cu118`（`torch.version.cuda==11.8`），算力 `(8,9)` 正是 sm_89 —— 对上 §4.3 的推理。

### 步骤 6 · 装其余依赖（版本依据 §4.2，不含 torch、不含 tensorstore）

```powershell
pip install "anndata==0.8.0" "scanpy==1.8.1" "scirpy==0.10.1" "numpy==1.21.6" "numba==0.57.1" "scikit-learn==0.24.1" "umap-learn==0.5.1" "einops==0.4.1" "seaborn==0.12.2" "pandas==1.4.2" "matplotlib==3.5.2" "biopython==1.79" "tabulate==0.9.0" "plotly==5.10.0"
```

> **常见坑**：若某包报"无法编译 / 没有 wheel"，贴报错给我，换 conda-forge 版或放宽版本——别自己硬刚。

### 步骤 7 · 给 `chunked_anndata` 打补丁（依据 §4.5）

在 `scAtlasVAE` 仓库根目录执行（把 §4.5 那行硬导入改成可选）：

```powershell
python -c "import pathlib; p=pathlib.Path('scatlasvae/model/_gex_model.py'); s=p.read_text(encoding='utf-8'); s=s.replace('import chunked_anndata as ca', 'try:\n    import chunked_anndata as ca\nexcept ModuleNotFoundError:\n    ca = None  # 仅 chunked_adata_path 路径用到；in-memory 工作流不需要', 1); p.write_text(s, encoding='utf-8'); print('patched OK')"
```

**预期**：打印 `patched OK`。

### 步骤 8 · 装 scAtlasVAE 本体（`--no-deps`）

**为什么加 `--no-deps`**：§4.2 查到 `setup.py` 锁了 `torch==1.13.1`；若直接 `pip install -e .`，pip 会为满足它把你刚装的新 torch 又降回去。`--no-deps` 表示"只装它自己、不动依赖"（依赖我们已在步骤 4、6 亲手装好）。

```powershell
pip install -e . --no-deps
```

### 步骤 9 · 跑完整冒烟测试（本阶段收尾）

把 [`scripts/phase1_smoke_test.py`](../scripts/phase1_smoke_test.py) 拷到本机运行：

```powershell
python phase1_smoke_test.py
```

**预期**：依次打印 `[1/4]`~`[4/4]`，最后出现 `冒烟测试全部通过：环境 + GPU + 模型训练链路完全 OK`。

> **常见坑**：`import scatlasvae` 报 `No module named 'xxx'` → 某小依赖漏装，`pip install xxx` 补上并告诉我。

**记录区（本机实测 2026-07-09）**：
```
phase1_smoke_test.py 结果：[x] 通过   [ ] 报错(贴下方)
[1/4]~[4/4] 全过；GPU=RTX 4060 Laptop；合成数据 3 epoch 训练无 NaN，latent 形状 (512,10)；
末行 "冒烟测试全部通过：环境 + GPU + 模型训练链路完全 OK"
```
> 实跑时除 §4.5 的 `chunked_anndata` 外，还遇到**另两处**要处理（都属"库版本与本仓库假设不符"）：
> ① `scatlasvae/preprocessing/_preprocess.py` 顶部硬 `import scirpy`（仅 TCR/VDJ 函数用到，GEX 整合不需要）——同样改成 `try/except` 置空，免装 scirpy 一堆重依赖；
> ② `scatlasvae/utils/_utilities.py` 的 `get_default_device()` 调 `torch.mps.device_count()`，但 `torch.mps` 在 torch<2.1 不存在（我们用的 2.0.1）——加 `hasattr` 守卫即可（见 §10 报错表）。

---

## 7. 检查点与完成标准（DoD）

- [ ] 步骤 5：`torch.cuda.is_available()` 为 `True`、算力 `(8,9)`、GPU 矩阵乘法不报 cuBLAS 错
- [ ] 步骤 9：`import scatlasvae` 成功，合成数据能 `fit()` 并 `get_latent_embedding()` 得到 `(512, 10)` 的 latent

---

## 8. 自测题（这一阶段真学到了才答得上）

**关于"摸清一个库"的能力：**
1. 拿到一个陌生 Python 库，你会先看哪几个地方、各自回答什么问题？
2. 不看这份报告，你会**去哪、用什么操作**查到 scAtlasVAE 的 `batch_size` 默认值？查到 torch 的版本约束？
3. 怎么判断"一个库锁定的 PyTorch 能不能在我的显卡上跑"？需要交叉核对哪两个信息？

**关于环境本身：**
4. 为什么 4060 不能用仓库锁的 `torch==1.13.1(cu117)`？换 cu118 解决了什么？
5. `nvidia-smi` 显示 `CUDA Version 13.2`，是否意味着必须装 CUDA 13.2 的 PyTorch？为什么？
6. 为什么先装 PyTorch、最后再用 `--no-deps` 装本体？`chunked_anndata` 补丁绕过了什么、为什么能安全绕过？

---

## 9. 附录 · 评测环境 B（阶段二才用，可先跳过）

评测用的 `scib-metrics` 要 Python ≥ 3.10，与训练环境（3.8）冲突，所以**单独建一个环境**：

```powershell
conda create -n scib python=3.10 -y
conda activate scib
pip install scib-metrics scanpy scvi-tools
```

---

## 10. 常见报错速查

| 报错 | 原因 | 处理 |
|---|---|---|
| `CUBLAS_STATUS_INVALID_VALUE` / `sm_89 is not compatible` | torch 还是 cu117 | 重装 cu118 版 torch（步骤 4） |
| `import scatlasvae` → `No module named chunked_anndata` | 没打步骤 7 补丁 | 执行步骤 7 |
| `import scatlasvae` → `No module named 'scirpy'` | `_preprocess.py` 硬 import scirpy（TCR 依赖） | 把该行改 `try/except`（GEX 不需要），见 §9 记录区 |
| `AttributeError: module 'torch' has no attribute 'mps'` | `get_default_device()` 调 `torch.mps`，torch<2.1 没有 | 给该调用加 `hasattr(torch,'mps')` 守卫，见 §9 记录区 |
| `import scatlasvae` → `No module named 'xxx'` | 某依赖漏装 | `pip install xxx`，并告诉我 |
| `fit()` 出 `nan` | 有细胞 total-count=0 / 学习率偏大 | 确保每细胞 count>0；`fit(lr=1e-5)` |
| 装包报 `Microsoft Visual C++ 14.0 required` | 某包要现编译 | 贴给我，换 conda-forge 预编译版 |

---

## 11. 延伸阅读

- 如何读源码/上手一个库的通用思路：先 README → 依赖 → 文档 → 入口类 → Issues（本报告 §3 就是这套）
- PyTorch 本地安装（按 CUDA 选版本）：https://pytorch.org/get-started/locally/
- NVIDIA 各 GPU 算力对照：https://developer.nvidia.com/cuda-gpus
- conda 环境管理：https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

---

## 12. 阶段一小结（本机 2026-07-09 跑通）

- 最终环境：`scatlasvae`（py3.8.20, torch 2.0.1+cu118, scAtlasVAE 1.0.6a3 可编辑安装），其余旧 pin 栈（scanpy 1.8.1 / anndata 0.8.0 / numpy 1.21.6 / numba 0.57.1 / scikit-learn 0.24.1 …）一次装好、无版本冲突。评测环境 `scib`（py3.10）另见 §9。
- 侦查/实跑据此处理的**五处**（前三处纯侦查即可预判，后两处要真跑才暴露）：① torch 换 cu118（§4.3）；② `--no-deps` 装本体（§4.2）；③ `chunked_anndata` 改可选（§4.5）；④ `scirpy` 改可选（GEX 不需要 TCR 依赖）；⑤ `torch.mps` 守卫（torch 2.0.1 无 `torch.mps`）。
- 冒烟测试结果：**全部通过**——GPU 矩阵乘不报 cuBLAS、`import scatlasvae` 成功、合成数据 `fit()` 无 NaN、`get_latent_embedding()` 得到 `(512, 10)`。
