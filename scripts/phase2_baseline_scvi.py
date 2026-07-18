"""阶段二 · 步骤 5：scVI baseline（用官方 scvi-tools，别自己手写）。

为什么要它
    scVI 是经典的批次整合 VAE。scvi-tools 默认 ``encode_covariates=False``，因此默认
    encoder 也只接收 X；batch 在 decoder 中使用。它仍是与 scAtlasVAE 对照的重要基线。

Windows 安装小记
    scvi-tools 依赖 JAX 生态的 orbax-checkpoint，包内有超长路径的测试文件，
    在**未开长路径**的 Windows 上会触发 260 字符上限而装不上。解决：以管理员执行
    `Set-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name LongPathsEnabled -Value 1`，
    重开终端后即可 `pip install scvi-tools`。本机据此单独建了 `scvi`(py3.10, CPU torch) 环境。

用法（在 `scvi` 环境中；默认同时保存 embedding、模型与 provenance）
    python phase2_baseline_scvi.py
    python phase2_baseline_scvi.py --overwrite-model   # 明确允许覆盖已有匹配模型

对应报告
    reports/phase2_integration_and_benchmark.md 步骤 5。
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import scanpy as sc
import scvi
import torch

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"
DEFAULT_MODEL_DIR = "scvi_model"
DEFAULT_SEED = 0


def _ordered_digest(values):
    """对有序 obs/var 名称生成可跨环境复核的 SHA-256。"""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _array_digest(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_h5_dataset_digest(digest, dataset, block_bytes=8 * 1024 * 1024):
    digest.update(str(dataset.dtype).encode("ascii"))
    digest.update(np.asarray(dataset.shape, dtype="<i8").tobytes())
    if dataset.ndim == 0:
        digest.update(np.ascontiguousarray(dataset[()]).tobytes())
        return
    row_items = int(np.prod(dataset.shape[1:], dtype=np.int64)) or 1
    rows_per_block = max(1, block_bytes // (row_items * dataset.dtype.itemsize))
    for start in range(0, dataset.shape[0], rows_per_block):
        values = np.ascontiguousarray(dataset[start:start + rows_per_block])
        digest.update(values.tobytes())


def _h5ad_counts_digest(path):
    """按 staged H5AD 的实际落盘 CSR 表示分块计算 counts SHA-256。"""
    with h5py.File(path, "r") as h5:
        counts = h5["layers"]["counts"]
        if not isinstance(counts, h5py.Group):
            raise TypeError("layers['counts'] 必须以 CSR group 存储")
        encoding = counts.attrs.get("encoding-type")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        if encoding != "csr_matrix":
            raise TypeError(f"layers['counts'] 不是 CSR：{encoding!r}")
        digest = hashlib.sha256(b"h5ad.csr_matrix\0")
        digest.update(np.asarray(counts.attrs["shape"], dtype="<i8").tobytes())
        for name in ("data", "indices", "indptr"):
            _update_h5_dataset_digest(digest, counts[name])
        return digest.hexdigest()


def _remove_staged_target(path):
    """事务回滚时移走本脚本刚提交的单个目标。"""
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main(max_epochs=10, seed=DEFAULT_SEED, model_dir=DEFAULT_MODEL_DIR,
         overwrite_model=False):
    model_path = Path(model_dir)
    proc_path = Path(PROC_PATH)
    manifest_path = model_path.with_name(f"{model_path.name}_manifest.json")
    if (model_path.exists() or manifest_path.exists()) and not overwrite_model:
        raise FileExistsError(
            f"模型或 manifest 已存在：{model_path} / {manifest_path}。"
            "为防止静默复用/覆盖旧 checkpoint，"
            "请先核对 provenance；确认重训后使用 --overwrite-model。"
        )
    if model_path.exists() and not model_path.is_dir():
        raise NotADirectoryError(f"模型目标不是目录：{model_path}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    scvi.settings.seed = seed

    adata = sc.read_h5ad(proc_path)
    # scVI 也要原始整数计数：用预处理时备份的 layers['counts']。
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=BATCH_KEY)
    model = scvi.model.SCVI(adata, encode_covariates=False)

    # 论文 baseline 固定 max_epochs=10；这里沿用并显式记录。
    model.train(max_epochs=max_epochs)
    embedding = model.get_latent_representation()

    # 在临时目录一次性 stage 模型、H5AD 与 manifest；正式路径只接收已验证文件。
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=model_path.parent, prefix=f".{model_path.name}-stage-"
    ) as model_tmp_root, tempfile.TemporaryDirectory(
        dir=proc_path.parent, prefix=f".{proc_path.stem}-stage-"
    ) as data_tmp_root:
        model_tmp_root = Path(model_tmp_root)
        data_tmp_root = Path(data_tmp_root)
        staged_model = model_tmp_root / "model"
        staged_h5ad = data_tmp_root / proc_path.name
        staged_manifest = model_tmp_root / manifest_path.name

        model.save(staged_model, save_anndata=False)
        reloaded = scvi.model.SCVI.load(
            staged_model, adata=adata, accelerator="cpu", device=1
        )
        reloaded_embedding = reloaded.get_latent_representation()
        reload_max_abs_error = float(np.max(np.abs(embedding - reloaded_embedding)))
        if not np.allclose(embedding, reloaded_embedding, atol=1e-6, rtol=1e-6):
            raise RuntimeError(
                f"scVI save/reload 后 latent 不一致，max_abs_error={reload_max_abs_error:.3e}"
            )

        adata.obsm["X_scVI"] = embedding
        adata.write_h5ad(staged_h5ad)
        checkpoint_path = staged_model / "model.pt"
        provenance = {
            "artifact": "scVI baseline matching tcell_processed.h5ad::obsm['X_scVI']",
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "latent_dim": int(embedding.shape[1]),
            "batch_key": BATCH_KEY,
            "counts_layer": "counts",
            "counts_layer_sha256": _h5ad_counts_digest(staged_h5ad),
            "encode_covariates": False,
            "max_epochs": int(max_epochs),
            "seed": int(seed),
            "scvi_version": str(scvi.__version__),
            "torch_version": str(torch.__version__),
            "ordered_obs_names_sha256": _ordered_digest(adata.obs_names),
            "ordered_var_names_sha256": _ordered_digest(adata.var_names),
            "x_scvi_sha256": _array_digest(embedding),
            "model_pt_sha256": _file_digest(checkpoint_path),
            "reload_max_abs_error": reload_max_abs_error,
        }
        staged_manifest.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Manifest 最后提交，作为三件套完整的 commit marker。原产物备份放在持久 sibling
        # 目录；即便进程被强制终止，也不会被 TemporaryDirectory 自动清理掉。
        backup_model_root = Path(tempfile.mkdtemp(
            dir=model_path.parent, prefix=f".{model_path.name}-backup-"
        ))
        backup_data_root = Path(tempfile.mkdtemp(
            dir=proc_path.parent, prefix=f".{proc_path.stem}-backup-"
        ))
        backup_h5ad = backup_data_root / "previous.h5ad"
        backup_model = backup_model_root / "previous_model"
        backup_manifest = backup_model_root / "previous_manifest.json"
        h5ad_backed_up = model_backed_up = manifest_backed_up = False
        model_committed = manifest_committed = False
        try:
            if manifest_path.exists():
                os.replace(manifest_path, backup_manifest)
                manifest_backed_up = True
            if model_path.exists():
                os.replace(model_path, backup_model)
                model_backed_up = True
            os.replace(proc_path, backup_h5ad)
            h5ad_backed_up = True

            os.replace(staged_model, model_path)
            model_committed = True
            # model 先于 H5AD：硬崩溃后至少会形成 model/manifest 半套，三态门禁必失败；
            # 不会出现只有 X_scVI 已变化、model/manifest 仍双缺而被误判 absent_by_design。
            os.replace(staged_h5ad, proc_path)
            os.replace(staged_manifest, manifest_path)
            manifest_committed = True
        except BaseException as commit_error:
            rollback_errors = []
            try:
                if h5ad_backed_up:
                    # 文件 os.replace 可直接覆盖新 H5AD，避免先删后恢复留下空窗。
                    os.replace(backup_h5ad, proc_path)
            except BaseException as exc:
                rollback_errors.append(f"H5AD: {exc}")
            try:
                if model_backed_up:
                    if model_path.exists():
                        _remove_staged_target(model_path)
                    os.replace(backup_model, model_path)
                elif model_committed:
                    _remove_staged_target(model_path)
            except BaseException as exc:
                rollback_errors.append(f"model: {exc}")
            try:
                if manifest_backed_up:
                    os.replace(backup_manifest, manifest_path)
                elif manifest_committed:
                    _remove_staged_target(manifest_path)
            except BaseException as exc:
                rollback_errors.append(f"manifest: {exc}")

            if rollback_errors:
                raise RuntimeError(
                    "scVI artifact commit failed and rollback was incomplete; "
                    f"recoverable backups remain at {backup_data_root} and "
                    f"{backup_model_root}. Errors: {'; '.join(rollback_errors)}"
                ) from commit_error
            for backup_root in (backup_data_root, backup_model_root):
                shutil.rmtree(backup_root, ignore_errors=True)
            raise
        else:
            # 完整提交后才清理旧产物；清理失败只留下可识别备份，不破坏新三件套。
            for backup_root in (backup_data_root, backup_model_root):
                try:
                    shutil.rmtree(backup_root)
                except OSError as exc:
                    print(f"警告：旧产物备份未能自动清理：{backup_root} ({exc})")
    print(
        "scVI 训练完成：已写入 obsm['X_scVI']，并保存匹配模型与 provenance -> "
        f"{model_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--overwrite-model", action="store_true",
        help="明确允许覆盖已有模型目录；默认拒绝，防止 checkpoint 与 embedding provenance 混淆",
    )
    args = parser.parse_args()
    main(args.max_epochs, args.seed, args.model_dir, args.overwrite_model)
