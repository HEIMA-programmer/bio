"""阶段一冒烟测试：验证 scAtlasVAE 训练环境是否搭建成功。

用途
    在不下载任何真实数据的前提下，用一份极小的合成数据（512 细胞 × 100 基因），
    一次性验证【环境 → GPU → 模型训练 → 取 latent】整条链路是否连通。

用法
    在 RTX 4060 机器上、激活 conda 环境 `scatlasvae` 后运行：
        python phase1_smoke_test.py

前置条件
    已完成 phase1_environment_setup.md 的步骤 4~8：
    torch 2.0.1+cu118 可用、scAtlasVAE 已 `pip install -e . --no-deps`。

预期输出
    依次打印 [1/4]~[4/4] 四个阶段，最后出现
    "冒烟测试全部通过：环境 + GPU + 模型训练链路完全 OK"。
    任一 assert 失败即说明对应环节有问题，按报错回到报告排查。

对应报告
    reports/phase1_environment_setup.md 步骤 9；概念背景见 reports/01_concepts_and_toolbox.md。
"""

import numpy as np


# ============================================================
# [1/4] 检查 PyTorch 与 CUDA
# ------------------------------------------------------------
# 先确认最底层的一环（PyTorch 能否真正驱动 4060）再往上验证，
# 便于把问题定位到具体层，而不是最后一锅端地报错。
# ============================================================
print("[1/4] 检查 PyTorch 与 CUDA")

import torch

print("  torch 版本        :", torch.__version__)
print("  torch 编译的 CUDA :", torch.version.cuda)
print("  CUDA 是否可用     :", torch.cuda.is_available())
assert torch.cuda.is_available(), \
    "CUDA 不可用：检查 NVIDIA 驱动，以及 torch 是否装的是 cu118 版本（见报告步骤 4）。"
print("  GPU 型号          :", torch.cuda.get_device_name(0))

# 关键验证：矩阵乘法会真正调用 cuBLAS。若 torch 装成了不支持 sm_89 的 cu117，
# 这一行会抛 CUBLAS_STATUS_INVALID_VALUE —— 能过即证明 PyTorch 换对了。
x = torch.randn(2048, 2048, device="cuda")
y = (x @ x).sum()
torch.cuda.synchronize()  # 等 GPU 算完，确保异常在此处（而非稍后）暴露
print("  CUDA 矩阵乘法     : OK, 结果均值 =", float(y.mean()))


# ============================================================
# [2/4] 构造极小合成 AnnData（512 细胞 × 100 基因）
# ------------------------------------------------------------
# 用合成数据而非真实数据，是为了让本测试快速、可复现、且与网络/磁盘解耦。
# ============================================================
print("\n[2/4] 构造极小合成 AnnData（512 细胞 × 100 基因）")

import anndata as ad
import pandas as pd

rng = np.random.default_rng(0)
n_cells, n_genes = 512, 100

# scAtlasVAE 默认走 ZINB 重构，要求 adata.X 为整数计数，故用泊松分布模拟原始 count。
X = rng.poisson(lam=1.0, size=(n_cells, n_genes)).astype(np.int32)

# 兜底：把任何“全零细胞”的第 0 个基因置 1。总计数为 0 的细胞会导致训练出 NaN
# （见报告“常见坑”与 01 文档 ZINB 小节）。
X[X.sum(1) == 0, 0] = 1

batch = rng.choice(["batchA", "batchB", "batchC"], size=n_cells)
adata = ad.AnnData(
    X=X,
    obs=pd.DataFrame(
        {"batch": pd.Categorical(batch)},
        index=[f"cell_{i}" for i in range(n_cells)],
    ),
)
adata.var_names = [f"gene_{j}" for j in range(n_genes)]

print(adata)
print("  每个细胞 total_count > 0 :", bool((np.asarray(adata.X).sum(1) > 0).all()))


# ============================================================
# [3/4] 构建并训练 scAtlasVAE（max_epoch=3，仅验证链路）
# ------------------------------------------------------------
# 只跑 3 个 epoch：本测试的目的是验证“能训练”，而非训练出好结果。
# ============================================================
print("\n[3/4] 构建并训练 scAtlasVAE（max_epoch=3）")

import scatlasvae

print("  scatlasvae 版本 :", getattr(scatlasvae, "__version__", "unknown"))

# 构造函数是 keyword-only（def __init__(self, *, ...)）：adata 必须写成 adata=adata，
# 否则报 “takes 1 positional argument”。这是 README 示例里容易踩的坑。
model = scatlasvae.model.scAtlasVAE(
    adata=adata,
    batch_key="batch",
)
model.fit(max_epoch=3)


# ============================================================
# [4/4] 取 latent embedding
# ------------------------------------------------------------
# get_latent_embedding() 默认返回潜均值 q_mu，形状 (n_cells, n_latent=10)。
# ============================================================
print("\n[4/4] 取 latent embedding")

z = np.asarray(model.get_latent_embedding())
print("  latent 形状 :", z.shape, "（应为 (512, 10)）")
assert z.shape == (n_cells, 10), "latent 维度不符合预期，模型或数据可能有问题。"

print("\n冒烟测试全部通过：环境 + GPU + 模型训练链路完全 OK")
