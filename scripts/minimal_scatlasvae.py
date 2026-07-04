"""阶段三核心交付：从零手写的最小 scAtlasVAE（纯 PyTorch）。

这是复现的 L2 目标——不看官方 `_gex_model.py`，只对着论文公式和 01 文档，
用最少的代码复刻 scAtlasVAE 的核心机制，理解"论文公式 → 代码"的每一步映射。

忠实但最小：实现 批不变编码器 / 重参数化 / 批条件解码器 / ZINB 重构 / KL 预热 /
单个分类头（半监督）。刻意不实现的：MMD、TabNet 编码器、latent constraint、
多 batch 层级、多 label 头（这些是可选特性，见 01 文档 §1.5 与阶段三报告差异清单）。

公式对照（论文 Methods / 01 文档 §1.4）：
    编码器  q(z|X) = N(mu, sigma^2)            <->  encode()
    采样    z = mu + sigma * eps               <->  reparameterize()
    解码器  mu = softmax(f(z,B)) * libsize     <->  decode()
    损失    L = -E[log p(X|z,B)] + w_kl*KL      <->  elbo()
                (+ w_ct * CrossEntropy)

对应报告：reports/phase3_reimplement_vae.md
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------------------------------------
# ZINB 的对数似然（数值稳定版，遵循 scVI 的公式）
# 参数：x=真实计数, mu=均值, theta=离散度, pi=零膨胀门控(logits)
# 返回：每个基因的 log p(x)（越大越贴合）—— 重构损失 = 它的相反数
# ------------------------------------------------------------
def log_zinb(x, mu, theta, pi, eps=1e-8):
    softplus_pi = F.softplus(-pi)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    # 门控项：-pi + theta*(log theta - log(theta+mu))
    pi_theta_log = -pi + theta * (torch.log(theta + eps) - log_theta_mu_eps)
    # x == 0 与 x > 0 两种情形分别给对数概率
    case_zero = F.softplus(pi_theta_log) - softplus_pi
    case_nonzero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return torch.where(x < eps, case_zero, case_nonzero)


class MinimalScAtlasVAE(nn.Module):
    def __init__(self, n_genes, n_batch, n_label=0,
                 n_latent=10, n_hidden=128, batch_dim=8):
        super().__init__()
        self.n_label = n_label

        # === 批不变编码器 F(X) ===（输入只有基因表达，不含 batch —— 这是题眼）
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU()
        )
        self.z_mean = nn.Linear(n_hidden, n_latent)     # -> mu
        self.z_logvar = nn.Linear(n_hidden, n_latent)   # -> log sigma^2

        # === 批条件解码器 F(z, B) ===（batch 只在这里注入）
        self.batch_emb = nn.Embedding(n_batch, batch_dim)   # 把批次索引变成向量
        self.decoder = nn.Sequential(
            nn.Linear(n_latent + batch_dim, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU()
        )
        self.px_scale = nn.Linear(n_hidden, n_genes)    # -> 各基因占比(过 softmax)
        self.px_rate = nn.Linear(n_hidden, n_genes)     # -> 离散度 theta 的 log
        self.px_dropout = nn.Linear(n_hidden, n_genes)  # -> 零膨胀门控 logits

        # === 单个分类头（半监督，可选）===
        self.classifier = nn.Linear(n_latent, n_label) if n_label > 0 else None

    def encode(self, x):
        # 编码器输入先 log1p（log_variational=True）；重构目标仍是原始计数
        h = self.encoder(torch.log1p(x))
        mu = self.z_mean(h)
        var = torch.exp(self.z_logvar(h)) + 1e-4        # +eps 保证正、数值稳定
        return mu, var

    @staticmethod
    def reparameterize(mu, var):
        # z = mu + sigma * eps, eps ~ N(0,1) —— 让随机采样可导（见 01 文档 §1.4d）
        return mu + var.sqrt() * torch.randn_like(mu)

    def decode(self, z, batch_index, libsize):
        h = self.decoder(torch.cat([z, self.batch_emb(batch_index)], dim=-1))
        scale = F.softmax(self.px_scale(h), dim=-1)     # 各基因占比，和为 1
        mu = scale * libsize                            # 占比 × 文库大小 = 均值
        theta = torch.exp(self.px_rate(h))              # 离散度 > 0
        pi = self.px_dropout(h)                         # 门控 logits
        return mu, theta, pi

    def elbo(self, x, batch_index, kl_weight):
        libsize = x.sum(dim=1, keepdim=True)            # 文库大小 = 每细胞总计数
        mu_z, var_z = self.encode(x)
        z = self.reparameterize(mu_z, var_z)
        mu, theta, pi = self.decode(z, batch_index, libsize)

        recon = -log_zinb(x, mu, theta, pi).sum(dim=1)  # ZINB 负对数似然
        # 解析 KL: 0.5 * sum(var + mu^2 - 1 - log var)
        kl = 0.5 * (var_z + mu_z.pow(2) - 1 - torch.log(var_z)).sum(dim=1)
        loss = (recon + kl_weight * kl).mean()
        return loss, z, recon.mean(), kl.mean()

    @torch.no_grad()
    def get_latent_embedding(self, X, device="cpu"):
        self.eval()
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        mu, _ = self.encode(x)
        return mu.cpu().numpy()

    def fit(self, X, batch_index, labels=None,
            max_epoch=None, lr=5e-5, weight_decay=1e-6,
            batch_size=128, seed=12, pred_weight=1.0, device="cuda"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        N = X.shape[0]
        if max_epoch is None:                           # 论文默认 epoch 公式
            max_epoch = int(np.min([round((20000 / N) * 400), 400]))

        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        b = torch.as_tensor(np.asarray(batch_index), dtype=torch.long)
        y = (torch.as_tensor(np.asarray(labels), dtype=torch.long)
             if labels is not None else torch.full((N,), -1))
        loader = DataLoader(TensorDataset(X, b, y), batch_size=batch_size, shuffle=True)

        self.to(device)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        ce = nn.CrossEntropyLoss(ignore_index=-1)       # 没标签的细胞跳过分类损失

        history = []
        for epoch in range(1, max_epoch + 1):
            # KL 预热：权重从 0 线性升到 1（贯穿训练，防后验坍缩，见 01 文档 §1.4i）
            kl_weight = min(1.0, epoch / max_epoch)
            self.train()
            running = 0.0
            for xb, bb, yb in loader:
                xb, bb, yb = xb.to(device), bb.to(device), yb.to(device)
                loss, z, _, _ = self.elbo(xb, bb, kl_weight)
                if self.classifier is not None and (yb >= 0).any():
                    loss = loss + pred_weight * ce(self.classifier(z), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += float(loss) * xb.size(0)
            history.append(running / N)
            if epoch % 10 == 0 or epoch == max_epoch:
                print(f"epoch {epoch:3d}/{max_epoch}  loss={history[-1]:.3f}  kl_w={kl_weight:.2f}")
        return history


if __name__ == "__main__":
    # 自测：合成小数据验证前向/训练能跑通（不代表科学结果）
    rng = np.random.default_rng(0)
    n_cells, n_genes, n_batch = 512, 200, 3
    X = rng.poisson(1.0, size=(n_cells, n_genes)).astype("float32")
    X[X.sum(1) == 0, 0] = 1
    batch = rng.integers(0, n_batch, size=n_cells)
    model = MinimalScAtlasVAE(n_genes, n_batch, n_label=0)
    model.fit(X, batch, max_epoch=3, device="cpu")
    print("latent shape:", model.get_latent_embedding(X).shape)
