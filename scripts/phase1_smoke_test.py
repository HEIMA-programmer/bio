"""
scAtlasVAE 阶段一 · 冒烟测试
================================
在 RTX 4060 机器上、激活 conda 环境 scatlasvae 后运行：

    python phase1_smoke_test.py

目的：不下载任何真实数据，用一份极小的合成数据（512 细胞 × 100 基因），
一次性验证【环境 + GPU + 模型训练 + 取 latent】整条链路是否通。
能一路打印到最后的 🎉，就说明训练环境彻底 OK，可以进入阶段二。
"""
import numpy as np

print("=" * 60)
print("[1/4] 检查 PyTorch 与 CUDA")
print("=" * 60)
import torch
print("torch 版本        :", torch.__version__)
print("torch 编译的 CUDA :", torch.version.cuda)
print("CUDA 是否可用     :", torch.cuda.is_available())
assert torch.cuda.is_available(), \
    "❌ CUDA 不可用！检查 NVIDIA 驱动，以及 torch 是否装的是 cu118 版本。"
print("GPU 型号          :", torch.cuda.get_device_name(0))

# 触发 cuBLAS：旧的 cu117 版本在 4060(sm_89) 上这一步会抛
# CUDA error: CUBLAS_STATUS_INVALID_VALUE —— 能过说明 PyTorch 换对了。
x = torch.randn(2048, 2048, device="cuda")
y = (x @ x).sum()
torch.cuda.synchronize()
print("CUDA 矩阵乘法     : OK, 结果均值 =", float(y.mean()))

print()
print("=" * 60)
print("[2/4] 构造极小合成 AnnData（512 细胞 × 100 基因）")
print("=" * 60)
import anndata as ad
import pandas as pd

rng = np.random.default_rng(0)
n_cells, n_genes = 512, 100
# 泊松计数模拟 scRNA raw count；scAtlasVAE 的 ZINB 重构需要整数 count
X = rng.poisson(lam=1.0, size=(n_cells, n_genes)).astype(np.int32)
# 兜底：任何“全零细胞”给第 0 个基因 +1，避免 total-count=0 导致训练 NaN
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
print("每个细胞 total_count > 0 :", bool((np.asarray(adata.X).sum(1) > 0).all()))

print()
print("=" * 60)
print("[3/4] 构建并训练 scAtlasVAE（max_epoch=3，只为验证链路）")
print("=" * 60)
import scatlasvae
print("scatlasvae 版本 :", getattr(scatlasvae, "__version__", "unknown"))

# 注意：构造函数是 keyword-only，adata 必须写成 adata=adata
model = scatlasvae.model.scAtlasVAE(
    adata=adata,
    batch_key="batch",
)
model.fit(max_epoch=3)

print()
print("=" * 60)
print("[4/4] 取 latent embedding")
print("=" * 60)
z = np.asarray(model.get_latent_embedding())
print("latent 形状 :", z.shape, " （应为 (512, 10)）")
assert z.shape == (n_cells, 10), "❌ latent 维度不符合预期！"

print()
print("🎉 冒烟测试全部通过：环境 + GPU + 模型训练链路完全 OK！")
