# 阶段一报告 · 环境搭建（Windows 原生 + RTX 4060 + conda）

> 目标机器：本地 **RTX 4060（8GB, Ada Lovelace, 算力 sm_89）**，**Windows 原生**，用 **conda/miniconda** 管理环境。
> 本文既是**操作指南**（你照着做），也是**记录**（每步留了「记录区」填你实际看到的输出）。遇到任何报错，把完整报错贴给军师（Claude Code）。

---

## 0. 本阶段要达成什么（完成标准 DoD）

跑通下面这份 `phase1_smoke_test.py`（在 `bio/scripts/`），一路打印到最后的 🎉，即算阶段一完成：

- ✅ `torch.cuda.is_available()` 为 `True`，且能在 GPU 上做矩阵乘法不报 cuBLAS 错
- ✅ `import scatlasvae` 成功
- ✅ 用合成小数据能 `scAtlasVAE(...).fit()` 并 `get_latent_embedding()` 拿到 `(512, 10)` 的 latent

**别在这一阶段耗超过两天**。装不上就贴报错给我，我们一起排。

---

## 1. 背景：为什么不能照仓库原样装（这一步的核心知识点）

仓库 `requirements.txt` / `setup.py` 把 PyTorch 锁死成 **`torch==1.13.1`**（默认 cu117，2022 年的构建）。
你的 4060 是 **Ada Lovelace 架构，算力 sm_89**。**CUDA 11.7（cu117）里没有为 sm_89 预编译的计算核**。
后果：`torch.cuda.is_available()` 可能是 `True`，但一做矩阵乘法就抛
`RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE`（正是 README「Common Issues」里的那条）。

👉 **支持 Ada 的最低 CUDA 是 11.8（cu118）**。所以我们把 PyTorch 换成 `2.0.1 + cu118`。
好消息：我已通读源码，**scAtlasVAE 没有用任何 torch 2.x 已删除的 API**，从 1.13 升到 2.0.1 是平滑的。

另外两个"军师提前排好的坑"（下面步骤里会用到）：
- **坑 A**：`setup.py` 锁了 `torch==1.13.1`，所以要用 `pip install -e . --no-deps` 装本体，否则它会把我们的 2.0.1 又降回去。
- **坑 B**：源码顶部 `import chunked_anndata`（依赖 tensorstore，py3.8+Windows 常常装不上）。但它只在"分块读超大数据"时才真正用到，我们用不着——所以打个 3 行小补丁把它变成"可选依赖"，彻底绕开。

---

## 2. 步骤

> Windows 上请用 **Anaconda Prompt**（装完 Miniconda 后在开始菜单能搜到）来敲 conda / pip 命令；`nvidia-smi` 用普通 PowerShell 也行。

### 步骤 0 · 前置检查（先跑这两条，把输出贴给我）

```powershell
nvidia-smi
conda --version
```

- `nvidia-smi`：确认显卡是 RTX 4060、看**驱动版本**和右上角 **CUDA Version**（这是驱动支持的*最高* CUDA；只要 ≥ 11.8 我们的 cu118 就能用。4060 的现代驱动通常显示 12.x，没问题）。
- `conda --version`：能打印版本号 → 已装好，跳到步骤 2；报"不是内部或外部命令" → 还没装，做步骤 1。

**记录区**：
```
nvidia-smi 关键信息：GPU=RTX 4060 Laptop（8GB, Ada/sm_89）  驱动=595.97  CUDA Version=13.2 → 远高于 cu118 所需，稳
conda --version：conda 24.11.3（已装好，步骤 1 跳过）
```

### 步骤 1 · 安装 Miniconda（若步骤 0 显示没装）

> ✅ **本机已装 conda 24.11.3（2026-07-03 实测），本步骤跳过，直接进步骤 2。**

下载 Windows 64-bit 安装包并安装（一路默认即可）：
- 官网：https://docs.conda.io/en/latest/miniconda.html
- 或用 winget：`winget install -e --id Anaconda.Miniconda3`

装完从开始菜单打开 **Anaconda Prompt**，重跑 `conda --version` 确认。

### 步骤 2 · 在 Windows 上克隆 scAtlasVAE 源码

我们用"可编辑安装"，这样你能直接读/改源码（阶段三手写 VAE 时要频繁对照）。

```powershell
git clone https://github.com/WanluLiuLab/scAtlasVAE.git
cd scAtlasVAE
```

> 没装 git？可以装 [Git for Windows](https://git-scm.com/download/win)，或直接在网页点 `Code → Download ZIP` 解压后 `cd` 进去。

### 步骤 3 · 创建并激活训练环境 A（Python 3.8）

```powershell
conda create -n scatlasvae python=3.8 -y
conda activate scatlasvae
```

激活后命令行前面会出现 `(scatlasvae)`。**之后所有 pip 命令都要在这个环境里跑。**

### 步骤 4 · 先装支持 Ada 的 PyTorch（cu118）—— 顺序很重要，先装它

```powershell
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

这一步会下载约 2GB，稍慢。装完先别急着装别的，先做步骤 5 的冒烟测试。

### 步骤 5 · CUDA 冒烟测试（必须通过，否则后面白搭）

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); x=torch.randn(2048,2048,device='cuda'); print('matmul OK:', float((x@x).mean())); print(torch.cuda.get_device_name(0))"
```

**预期**：打印形如 `2.0.1+cu118 11.8 True` → `matmul OK: ...` → `NVIDIA GeForce RTX 4060`，**不报 cuBLAS 错**。
若这里就报 `CUBLAS_STATUS_INVALID_VALUE` 或 `sm_89 is not compatible`，**先停下贴报错给我**——说明 torch 没装成 cu118。

**记录区**：
```
torch 版本/CUDA/可用：______
matmul：______
GPU：______
```

### 步骤 6 · 安装其余依赖（按仓库版本，不含 torch、不含 tensorstore）

```powershell
pip install "anndata==0.8.0" "scanpy==1.8.1" "scirpy==0.10.1" "numpy==1.21.6" "numba==0.57.1" "scikit-learn==0.24.1" "umap-learn==0.5.1" "einops==0.4.1" "seaborn==0.12.2" "pandas==1.4.2" "matplotlib==3.5.2" "biopython==1.79" "tabulate==0.9.0" "plotly==5.10.0"
```

> 若某个包在 Windows 上报"无法编译/没有 wheel"，把报错贴给我，我给你换成 conda-forge 版或放宽版本——**不要自己硬刚**。

### 步骤 7 · 打补丁：让 `chunked_anndata` 变成可选依赖（坑 B）

在 `scAtlasVAE` 仓库根目录（`(scatlasvae)` 环境激活状态）跑这一条，它会自动把源码第 17 行的硬导入改成 try/except：

```powershell
python -c "import pathlib; p=pathlib.Path('scatlasvae/model/_gex_model.py'); s=p.read_text(encoding='utf-8'); s=s.replace('import chunked_anndata as ca', 'try:\n    import chunked_anndata as ca\nexcept ModuleNotFoundError:\n    ca = None  # 仅 chunked_adata_path 路径用到；in-memory 工作流不需要', 1); p.write_text(s, encoding='utf-8'); print('patched OK')"
```

打印 `patched OK` 即可。（也可以用编辑器手动把 `import chunked_anndata as ca` 这一行改成上面的 try/except 版本，效果一样。）

### 步骤 8 · 安装 scAtlasVAE 本体（`--no-deps`，避免把 torch 降回去，坑 A）

```powershell
pip install -e . --no-deps
```

### 步骤 9 · 拿冒烟测试脚本跑一遍（本阶段的收尾）

把军师写好的 `phase1_smoke_test.py`（在你的 `bio` 仓库 `scripts/` 下）拷到当前机器运行。最简单的办法：在 `scatlasvae` 环境里新建一个文件粘贴脚本内容，或从 bio 仓库拉取。然后：

```powershell
python phase1_smoke_test.py
```

**预期**：依次打印 [1/4]~[4/4]，最后出现
`🎉 冒烟测试全部通过：环境 + GPU + 模型训练链路完全 OK！`

若中途某步报错（比如 `import scatlasvae` 报 `No module named xxx`），把**完整报错**贴给我：多半是某个小依赖没装上，一条 `pip install xxx` 就能补。

**记录区**：
```
phase1_smoke_test.py 结果：☐ 通过  ☐ 报错(贴下方)
______
```

---

## 附录 A · 评测环境 B（阶段二才用，可以现在建也可以先跳过）

评测用的 `scib-metrics` 要 Python ≥ 3.10，与训练环境（3.8）冲突，所以**单独建一个环境**：

```powershell
conda create -n scib python=3.10 -y
conda activate scib
pip install scib-metrics scanpy scvi-tools
```

> `scvi-tools` 用来跑 baseline（scVI）；`scib-metrics` 用来算整合指标。阶段二开始时我们再细化这里，现在知道有这么回事即可。

---

## 常见报错速查

| 报错 | 原因 | 处理 |
|---|---|---|
| `CUBLAS_STATUS_INVALID_VALUE` | torch 还是 cu117 / 装错了 | 卸掉重装 cu118 版 torch（步骤 4） |
| `sm_89 is not compatible` | 同上 | 同上 |
| `import scatlasvae` → `No module named chunked_anndata` | 没打步骤 7 的补丁 | 执行步骤 7 |
| `import scatlasvae` → `No module named 'xxx'` | 某依赖漏装 | `pip install xxx`，并告诉我 |
| `fit()` 出现 `nan` | 有细胞 total-count=0 / 学习率偏大 | 确保每细胞 count>0；`fit(lr=1e-5)` |
| 装包报"Microsoft Visual C++ 14.0 required" | 某包要现编译 | 贴给我，换 conda-forge 预编译版 |

---

## 阶段一小结（跑通后我来补）

- 最终环境：`scatlasvae`（py3.8, torch 2.0.1+cu118, scAtlasVAE -e 安装）
- 关键改动记录：① torch 换 cu118；② `--no-deps` 装本体；③ chunked_anndata 补丁
- 冒烟测试结果：______
