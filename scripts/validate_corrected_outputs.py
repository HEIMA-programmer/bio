#!/usr/bin/env python3
"""Final, read-only validation gate for the corrected reproduction outputs.

Run this script from ``data`` after the rerun pipelines have finished::

    python ../scripts/validate_corrected_outputs.py

The input artifacts are never modified.  The only file written is
``final_validation.json`` (or the path selected with ``--output``).  This is
deliberately a strict *local* acceptance gate: provenance checks require the
ignored training logs/checkpoints and large cross-atlas arrays produced by the
rerun scripts.  A fresh clone can inspect the committed JSON summary, but must
regenerate those local artifacts before rerunning the complete gate.

During an incomplete rerun, ``--allow-missing`` turns missing *required*
artifacts into SKIP records and exits successfully as long as every artifact
that is present is valid.  Without that flag, a missing required artifact is
a failure.  Optional Phase 3 outputs are checked when present.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd


ATOL = 1e-9

ZERO_SHOT = "scAtlasVAE (zero-shot)"
FULL_SHOT = "scAtlasVAE (full-shot)"
SCVI_KNN = "kNN on scVI latent"
METHOD_TO_NPZ_STEM = {
    ZERO_SHOT: "zeroshot",
    FULL_SHOT: "fullshot",
    SCVI_KNN: "knn_scvi",
}


class ValidationError(AssertionError):
    """Raised when an artifact exists but violates its expected contract."""


@dataclass
class CheckRecord:
    name: str
    status: str
    message: str
    required: bool
    details: dict[str, Any]


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _relative_names(paths: Sequence[Path], root: Path) -> list[str]:
    names = []
    for path in paths:
        try:
            names.append(str(path.relative_to(root)))
        except ValueError:
            names.append(str(path))
    return names


def _report_path(path: Path, base: Path) -> str:
    """Return a portable path for the persisted validation summary."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(base.resolve()))
    except ValueError:
        return str(resolved)


class ValidationSuite:
    def __init__(self, data_dir: Path, allow_missing: bool) -> None:
        self.data_dir = data_dir
        self.allow_missing = allow_missing
        self.records: list[CheckRecord] = []
        self.required_missing_skipped = False

    def add(
        self,
        name: str,
        status: str,
        message: str,
        *,
        required: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        record = CheckRecord(
            name=name,
            status=status,
            message=message,
            required=required,
            details=_json_safe(details or {}),
        )
        self.records.append(record)
        print(f"[{status:4}] {name}: {message}")

    def check(
        self,
        name: str,
        paths: Sequence[Path],
        callback: Callable[[], dict[str, Any] | None],
        *,
        required: bool = True,
    ) -> None:
        missing = [path for path in paths if not path.is_file()]
        empty = [path for path in paths if path.is_file() and path.stat().st_size == 0]
        if missing or (empty and required and self.allow_missing):
            details = {
                "missing": _relative_names(missing, self.data_dir),
                "empty_not_ready": _relative_names(empty, self.data_dir),
            }
            if required and not self.allow_missing:
                self.add(
                    name,
                    "FAIL",
                    "missing required artifact(s)",
                    required=True,
                    details=details,
                )
            else:
                if required:
                    self.required_missing_skipped = True
                self.add(
                    name,
                    "SKIP",
                    "required artifacts not ready (--allow-missing)"
                    if required
                    else "optional artifact not present",
                    required=required,
                    details=details,
                )
            return

        try:
            details = callback() or {}
        except Exception as exc:  # each check must become a machine-readable FAIL
            self.add(
                name,
                "FAIL",
                f"{type(exc).__name__}: {exc}",
                required=required,
            )
            return

        self.add(name, "PASS", "validated", required=required, details=details)

    def optional_pair(
        self,
        name: str,
        paths: Sequence[Path],
        callback: Callable[[], dict[str, Any] | None],
    ) -> None:
        existing = [path.is_file() for path in paths]
        if not any(existing):
            self.check(name, paths, callback, required=False)
            return
        if not all(existing):
            missing = [path for path, exists in zip(paths, existing) if not exists]
            self.add(
                name,
                "FAIL",
                "optional artifact set is only partially present",
                required=False,
                details={"missing": _relative_names(missing, self.data_dir)},
            )
            return
        self.check(name, paths, callback, required=False)

    def summary(self) -> dict[str, int]:
        return {
            status.lower(): sum(record.status == status for record in self.records)
            for status in ("PASS", "FAIL", "SKIP")
        }


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    ensure(path.stat().st_size > 0, f"{path.name} is empty")
    frame = pd.read_csv(path, **kwargs)
    ensure(len(frame) > 0, f"{path.name} has no rows")
    return frame


def _numeric_values(
    frame: pd.DataFrame, columns: Iterable[str], context: str
) -> np.ndarray:
    columns = list(columns)
    missing = [column for column in columns if column not in frame.columns]
    ensure(not missing, f"{context} missing column(s): {missing}")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ensure(np.isfinite(numeric).all(), f"{context} contains non-finite numeric values")
    return numeric


def _ensure_probability_metrics(values: np.ndarray, context: str) -> None:
    ensure(
        ((values >= -ATOL) & (values <= 1.0 + ATOL)).all(),
        f"{context} contains a metric outside [0, 1]",
    )


def _label_array(npz: Any, key: str) -> np.ndarray:
    ensure(key in npz.files, f"NPZ missing key {key!r}")
    values = np.asarray(npz[key])
    ensure(values.ndim == 1, f"{key} must be one-dimensional, got {values.shape}")
    ensure(values.size > 0, f"{key} is empty")
    for value in values:
        ensure(value is not None, f"{key} contains None")
        if isinstance(value, (float, np.floating)):
            ensure(math.isfinite(float(value)), f"{key} contains a non-finite label")
        ensure(str(value).strip() != "", f"{key} contains an empty label")
    return values.astype(object, copy=False)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true.astype(str), y_pred.astype(str)]))
    scores = []
    true = y_true.astype(str)
    pred = y_pred.astype(str)
    for label in labels:
        tp = int(np.sum((true == label) & (pred == label)))
        fp = int(np.sum((true != label) & (pred == label)))
        fn = int(np.sum((true == label) & (pred != label)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
    return float(np.mean(scores))


def validate_transfer_pair(
    csv_path: Path,
    npz_path: Path,
    expected: dict[str, set[str]],
) -> dict[str, Any]:
    frame = _read_csv(csv_path)
    required_columns = {
        "design",
        "method",
        "n_query",
        "accuracy",
        "macro_f1",
        "macro_ovr_auc",
    }
    ensure(required_columns.issubset(frame.columns), f"missing columns: {sorted(required_columns - set(frame.columns))}")
    ensure(not frame[["design", "method"]].isna().any().any(), "design/method contains null")
    ensure(not frame.duplicated(["design", "method"]).any(), "duplicate design/method row")

    expected_pairs = {(design, method) for design, methods in expected.items() for method in methods}
    actual_pairs = set(zip(frame["design"].astype(str), frame["method"].astype(str)))
    ensure(actual_pairs == expected_pairs, f"design/method rows differ: expected={sorted(expected_pairs)}, actual={sorted(actual_pairs)}")

    metric_values = _numeric_values(
        frame,
        ["accuracy", "macro_f1", "macro_ovr_auc"],
        csv_path.name,
    )
    _ensure_probability_metrics(metric_values, csv_path.name)
    query_values = _numeric_values(frame, ["n_query"], csv_path.name).ravel()
    ensure((query_values > 0).all(), "n_query must be positive")
    ensure(np.allclose(query_values, np.rint(query_values), atol=0, rtol=0), "n_query must be integral")

    max_accuracy_error = 0.0
    max_f1_error = 0.0
    query_sizes: dict[str, int] = {}
    true_by_design: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=True) as store:
        for _, row in frame.iterrows():
            design = str(row["design"])
            method = str(row["method"])
            stem = METHOD_TO_NPZ_STEM.get(method)
            ensure(stem is not None, f"unknown transfer method {method!r}")
            true_key = f"{design}_{stem}_true"
            pred_key = f"{design}_{stem}_pred"
            y_true = _label_array(store, true_key)
            y_pred = _label_array(store, pred_key)
            n_query = int(round(float(row["n_query"])))
            ensure(len(y_true) == len(y_pred) == n_query, f"{design}/{method}: NPZ length does not match n_query={n_query}")

            if design in true_by_design:
                ensure(np.array_equal(true_by_design[design], y_true), f"{design}: true-label order differs across methods")
            else:
                true_by_design[design] = y_true.copy()
                query_sizes[design] = n_query

            accuracy = float(np.mean(y_true.astype(str) == y_pred.astype(str)))
            macro_f1 = _macro_f1(y_true, y_pred)
            accuracy_error = abs(accuracy - float(row["accuracy"]))
            f1_error = abs(macro_f1 - float(row["macro_f1"]))
            max_accuracy_error = max(max_accuracy_error, accuracy_error)
            max_f1_error = max(max_f1_error, f1_error)
            ensure(accuracy_error <= ATOL, f"{design}/{method}: accuracy mismatch {accuracy} vs {row['accuracy']}")
            ensure(f1_error <= ATOL, f"{design}/{method}: macro-F1 mismatch {macro_f1} vs {row['macro_f1']}")

    return {
        "csv": csv_path.name,
        "npz": npz_path.name,
        "rows": len(frame),
        "designs": sorted(expected),
        "n_query": query_sizes,
        "max_accuracy_abs_error": max_accuracy_error,
        "max_macro_f1_abs_error": max_f1_error,
    }


def validate_patient_selection(path: Path, patient_transfer_csv: Path) -> dict[str, Any]:
    selection = _read_csv(path)
    required = {"design", "interpretation", "patient", "n_cells", "n_labels"}
    ensure(required.issubset(selection.columns), f"missing columns: {sorted(required - set(selection.columns))}")
    ensure(len(selection) == 1, "patient selection must contain exactly one row")
    row = selection.iloc[0]
    ensure(str(row["design"]) == "P", "patient selection design must be P")
    ensure(str(row["patient"]) == "RC.P20190923", "unexpected held-out patient")
    numeric = _numeric_values(selection, ["n_cells", "n_labels"], path.name).ravel()
    ensure((numeric > 0).all(), "n_cells/n_labels must be positive")

    transfer = _read_csv(patient_transfer_csv)
    query_sizes = pd.to_numeric(transfer["n_query"], errors="coerce").to_numpy(float)
    ensure(np.isfinite(query_sizes).all(), "patient transfer n_query is non-finite")
    ensure(np.all(query_sizes == float(row["n_cells"])), "patient selection n_cells differs from transfer n_query")
    return {
        "patient": str(row["patient"]),
        "n_cells": int(row["n_cells"]),
        "n_labels": int(row["n_labels"]),
    }


def validate_patient_fulltime_run(
    status_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    def read_powershell_redirect(path: Path) -> str:
        raw = path.read_bytes()
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16", errors="replace")
        if raw[:4096].count(b"\x00") > max(8, len(raw[:4096]) // 8):
            return raw.decode("utf-16-le", errors="replace")
        return raw.decode("utf-8", errors="replace")

    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    ensure(status.get("status") == "complete", "patient full-time status is not complete")
    ensure(status.get("step") == "transfer_patient_fulltime", "unexpected full-time status step")
    ensure(checkpoint_path.stat().st_size > 1_000_000, "patient full-time checkpoint is unexpectedly small")

    stdout = read_powershell_redirect(stdout_path)
    stderr = read_powershell_redirect(stderr_path)
    ensure("protocol=fulltime" in stdout, "full-time stdout does not identify protocol=fulltime")
    ensure("max_epoch=150, pred_last_n_epoch=150" in stdout,
           "full-time stdout does not prove the 150/150 schedule")
    completed_epoch = any(
        "Epoch 150:" in line and "100%" in line and "150/150" in line
        for line in re.split(r"[\r\n]+", stderr)
    )
    ensure(completed_epoch, "full-time stderr does not show Epoch 150 completed at 150/150")
    combined = (stdout + "\n" + stderr).lower()
    forbidden = (
        "nan loss detected", "pred=nan", "=nan", "=inf",
        "traceback", "non-finite", "nonfinite",
    )
    hits = [token for token in forbidden if token in combined]
    ensure(not hits, f"patient full-time logs contain forbidden token(s): {hits}")
    return {
        "status": "complete",
        "schedule": "max_epoch=150,pred_last_n_epoch=150",
        "completed_epoch": 150,
        "checkpoint_mb": checkpoint_path.stat().st_size / (1024 ** 2),
        "forbidden_log_hits": hits,
    }


def validate_fair_knn(path: Path) -> dict[str, Any]:
    frame = _read_csv(path)
    required = {"design", "query_unit", "kind", "accuracy", "macro_f1", "macro_ovr_auc"}
    ensure(required.issubset(frame.columns), f"missing columns: {sorted(required - set(frame.columns))}")
    ensure(not frame[["design", "query_unit", "kind"]].isna().any().any(), "categorical fields contain null")
    ensure(not frame.duplicated(["design", "kind"]).any(), "duplicate design/kind row")

    kinds = {
        "transductive(full-data scVI)",
        "fair-inductive(reference-encoder-direct(no-query-training))",
    }
    expected_pairs = {(design, kind) for design in ("A", "B", "P") for kind in kinds}
    actual_pairs = set(zip(frame["design"].astype(str), frame["kind"].astype(str)))
    ensure(actual_pairs == expected_pairs, f"expected A/B/P x two modes, got {sorted(actual_pairs)}")
    expected_units = {"A": "random 5% cells", "B": "cancer:UCEC", "P": "patient:RC.P20190923"}
    for design, unit in expected_units.items():
        observed = set(frame.loc[frame["design"].astype(str) == design, "query_unit"].astype(str))
        ensure(observed == {unit}, f"{design}: expected query_unit={unit!r}, got {sorted(observed)}")

    values = _numeric_values(frame, ["accuracy", "macro_f1", "macro_ovr_auc"], path.name)
    _ensure_probability_metrics(values, path.name)
    return {"rows": len(frame), "designs": ["A", "B", "P"], "modes_per_design": 2}


def _ensure_finite_array(array: np.ndarray, name: str) -> None:
    ensure(np.issubdtype(array.dtype, np.number), f"{name} must be numeric, got {array.dtype}")
    ensure(np.isfinite(array).all(), f"{name} contains non-finite values")


def _ensure_nonempty_strings(array: np.ndarray, name: str) -> None:
    ensure(array.ndim == 1, f"{name} must be one-dimensional")
    ensure(array.size > 0, f"{name} is empty")
    ensure(
        all(value is not None and not pd.isna(value) and str(value).strip() for value in array),
        f"{name} contains null/empty values",
    )


def cross_paths(data_dir: Path, tag: str) -> dict[str, Path]:
    return {
        "npz": data_dir / f"phase5_cross_atlas{tag}.npz",
        "predictions": data_dir / f"phase5_cross_atlas_head_predictions{tag}.csv.gz",
        "counts": data_dir / f"phase5_cross_atlas_head_alignment_counts{tag}.csv",
        "alignment": data_dir / f"phase5_cross_atlas_head_alignment{tag}.csv",
        "links": data_dir / f"phase5_cross_atlas_head_alignment_links{tag}.csv",
        "mixing": data_dir / f"phase5_cross_atlas_mixing{tag}.csv",
        "latent": data_dir / f"phase5_cross_atlas_latent_knn_alignment{tag}.csv",
    }


def validate_cross_outputs(paths: dict[str, Path], label: str) -> dict[str, Any]:
    with np.load(paths["npz"], allow_pickle=True) as store:
        required_keys = {"X_cross", "X_pca_cross", "atlas", "ct_zheng", "ct_yost"}
        ensure(required_keys.issubset(store.files), f"NPZ missing keys: {sorted(required_keys - set(store.files))}")
        x_cross = np.asarray(store["X_cross"])
        x_pca = np.asarray(store["X_pca_cross"])
        ensure(x_cross.ndim == 2 and x_cross.shape[1] > 0, f"X_cross has invalid shape {x_cross.shape}")
        ensure(x_pca.ndim == 2 and x_pca.shape[1] > 0, f"X_pca_cross has invalid shape {x_pca.shape}")
        ensure(x_cross.shape[0] == x_pca.shape[0] > 0, "cross/PCA row counts differ")
        _ensure_finite_array(x_cross, "X_cross")
        _ensure_finite_array(x_pca, "X_pca_cross")
        n_cells = x_cross.shape[0]
        for key in ("atlas", "ct_zheng", "ct_yost"):
            values = np.asarray(store[key])
            ensure(len(values) == n_cells, f"{key} length differs from X_cross")
            _ensure_nonempty_strings(values, key)
        atlas_counts = pd.Series(np.asarray(store["atlas"]).astype(str)).value_counts().to_dict()

    predictions = _read_csv(paths["predictions"], index_col=0)
    expected_prediction_columns = ["predicted_zheng_label", "predicted_yost_label"]
    ensure(list(predictions.columns) == expected_prediction_columns, f"prediction columns differ: {list(predictions.columns)}")
    ensure(len(predictions) == n_cells, "prediction row count differs from NPZ")
    ensure(predictions.index.is_unique, "prediction index is not unique")
    ensure(not predictions.isna().any().any(), "predictions contain null")
    ensure((predictions.astype(str).apply(lambda col: col.str.len()) > 0).all().all(), "predictions contain empty labels")

    counts = _read_csv(paths["counts"], index_col=0)
    alignment = _read_csv(paths["alignment"], index_col=0)
    ensure(counts.index.is_unique and counts.columns.is_unique, "counts axes are not unique")
    ensure(alignment.index.is_unique and alignment.columns.is_unique, "alignment axes are not unique")
    ensure(counts.index.equals(alignment.index) and counts.columns.equals(alignment.columns), "counts/alignment axes differ")
    count_values = counts.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    align_values = alignment.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ensure(np.isfinite(count_values).all(), "counts contain non-finite values")
    ensure(np.isfinite(align_values).all(), "alignment contains non-finite values")
    ensure((count_values >= 0).all(), "counts contain negative values")
    ensure(np.allclose(count_values, np.rint(count_values), atol=0, rtol=0), "counts are not integral")
    _ensure_probability_metrics(align_values, "head alignment")

    yost_labels = predictions["predicted_yost_label"].astype(str)
    zheng_labels = predictions["predicted_zheng_label"].astype(str)
    ensure(set(yost_labels.unique()) == set(counts.index.astype(str)), "predicted Yost labels differ from count rows")
    ensure(set(zheng_labels.unique()) == set(counts.columns.astype(str)), "predicted Zheng labels differ from count columns")
    crosstab = pd.crosstab(yost_labels, zheng_labels).reindex(index=counts.index, columns=counts.columns, fill_value=0)
    ensure(np.array_equal(crosstab.to_numpy(), count_values.astype(np.int64)), "counts do not equal the prediction crosstab")
    ensure(int(count_values.sum()) == n_cells, "counts total differs from prediction/NPZ row count")

    row_totals = count_values.sum(axis=1)
    ensure((row_totals > 0).all(), "counts contain an empty Yost row")
    expected_alignment = count_values / row_totals[:, None]
    ensure(np.allclose(align_values, expected_alignment, atol=ATOL, rtol=ATOL), "alignment is not row-normalized counts")
    ensure(np.allclose(align_values.sum(axis=1), 1.0, atol=ATOL, rtol=0), "alignment rows do not sum to 1")

    links = _read_csv(paths["links"])
    link_columns = {"predicted_yost_label", "predicted_zheng_label", "p_zheng_given_yost", "cooccurrence_count", "threshold"}
    ensure(link_columns.issubset(links.columns), f"links missing columns: {sorted(link_columns - set(links.columns))}")
    ensure(not links.isna().any().any(), "links contain null")
    ensure(not links.duplicated(["predicted_yost_label", "predicted_zheng_label"]).any(), "links contain duplicate label pairs")
    link_numeric = _numeric_values(links, ["p_zheng_given_yost", "cooccurrence_count", "threshold"], paths["links"].name)
    ensure((link_numeric[:, 1] >= 0).all() and np.allclose(link_numeric[:, 1], np.rint(link_numeric[:, 1]), atol=0, rtol=0), "link cooccurrence counts are invalid")
    _ensure_probability_metrics(link_numeric[:, [0, 2]], "head links")
    thresholds = np.unique(link_numeric[:, 2])
    ensure(len(thresholds) == 1, f"links contain multiple thresholds: {thresholds.tolist()}")
    threshold = float(thresholds[0])
    expected_link_pairs = {
        (str(counts.index[i]), str(counts.columns[j]))
        for i, j in zip(*np.where(expected_alignment >= threshold))
    }
    observed_link_pairs = set(zip(links["predicted_yost_label"].astype(str), links["predicted_zheng_label"].astype(str)))
    ensure(observed_link_pairs == expected_link_pairs, "links are not exactly the alignment entries at/above threshold")
    for _, row in links.iterrows():
        yost = str(row["predicted_yost_label"])
        zheng = str(row["predicted_zheng_label"])
        ensure(abs(float(row["p_zheng_given_yost"]) - float(alignment.loc[yost, zheng])) <= ATOL, f"link probability mismatch for {yost}/{zheng}")
        ensure(int(row["cooccurrence_count"]) == int(counts.loc[yost, zheng]), f"link count mismatch for {yost}/{zheng}")

    mixing = _read_csv(paths["mixing"])
    ensure("embedding" in mixing.columns, "mixing CSV lacks embedding")
    mixing_numeric_columns = [column for column in mixing.columns if column != "embedding"]
    ensure(mixing_numeric_columns, "mixing CSV lacks metric columns")
    _numeric_values(mixing, mixing_numeric_columns, paths["mixing"].name)

    latent = _read_csv(paths["latent"], index_col=0)
    latent_values = latent.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ensure(np.isfinite(latent_values).all(), "latent-kNN alignment contains non-finite values")
    _ensure_probability_metrics(latent_values, "latent-kNN alignment")
    ensure(np.allclose(latent_values.sum(axis=1), 1.0, atol=ATOL, rtol=0), "latent-kNN alignment rows do not sum to 1")

    return {
        "variant": label,
        "n_cells": n_cells,
        "latent_shape": list(x_cross.shape),
        "pca_shape": list(x_pca.shape),
        "atlas_counts": atlas_counts,
        "head_matrix_shape": list(counts.shape),
        "head_links": len(links),
        "alignment_max_abs_error": float(np.max(np.abs(align_values - expected_alignment))),
    }


def validate_cross_shared_inputs(full_npz: Path, pl10_npz: Path) -> dict[str, Any]:
    with np.load(full_npz, allow_pickle=True) as full, np.load(pl10_npz, allow_pickle=True) as pl10:
        for key in ("atlas", "ct_zheng", "ct_yost"):
            ensure(np.array_equal(full[key], pl10[key]), f"full/pl10 {key} arrays differ")
        ensure(full["X_pca_cross"].shape == pl10["X_pca_cross"].shape, "full/pl10 PCA shapes differ")
        ensure(np.allclose(full["X_pca_cross"], pl10["X_pca_cross"], atol=1e-6, rtol=1e-6), "full/pl10 PCA inputs differ")
        ensure(full["X_cross"].shape == pl10["X_cross"].shape, "full/pl10 latent shapes differ")
        return {"n_cells": int(len(full["atlas"])), "pca_inputs_match": True}


def choose_cross_log_paths(log_dir: Path, variant: str) -> list[Path]:
    suffix = "full_classifier" if variant == "full" else "last10_classifier"
    prefixes = [f"cross_atlas_{suffix}_guarded", f"cross_atlas_{suffix}"]
    candidates = [[log_dir / f"{prefix}.stdout.log", log_dir / f"{prefix}.stderr.log"] for prefix in prefixes]
    for pair in candidates:
        if all(path.is_file() for path in pair):
            return pair
    return candidates[0]


def validate_cross_logs(paths: Sequence[Path], variant: str) -> dict[str, Any]:
    forbidden = ("nan loss detected", "pred=nan")
    hits: list[dict[str, Any]] = []
    total_hits = 0
    total_bytes = 0
    for path in paths:
        ensure(path.stat().st_size > 0, f"{path.name} is empty")
        total_bytes += path.stat().st_size
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                lowered = line.lower()
                for token in forbidden:
                    if token in lowered:
                        total_hits += lowered.count(token)
                        if len(hits) < 10:
                            hits.append(
                                {
                                    "file": path.name,
                                    "line": line_number,
                                    "token": token,
                                    "excerpt": line.strip()[:300],
                                }
                            )
    ensure(total_hits == 0, f"{variant} cross-atlas logs contain {total_hits} forbidden NaN marker(s): {hits}")
    return {"variant": variant, "files": [path.name for path in paths], "bytes_scanned": total_bytes, "forbidden_hits": 0}


def validate_invariance_csvs(scatlas_path: Path, scvi_path: Path) -> dict[str, Any]:
    scatlas = _read_csv(scatlas_path)
    scvi = _read_csv(scvi_path)
    common = ["n_cells_probed", "max_abs_dz_perm_batch", "mean_l2_drift_perm"]
    scatlas_common = _numeric_values(scatlas, common, scatlas_path.name)
    scvi_common = _numeric_values(scvi, common, scvi_path.name)
    ensure((scatlas_common[:, 0] > 0).all() and (scvi_common[:, 0] > 0).all(), "n_cells_probed must be positive")
    ensure((scatlas_common[:, 1:] >= 0).all() and (scvi_common[:, 1:] >= 0).all(), "invariance drifts must be non-negative")
    scatlas_none = _numeric_values(scatlas, ["max_abs_dz_none_batch"], scatlas_path.name)
    ensure((scatlas_none >= 0).all(), "scAtlasVAE none-batch drift is negative")
    ensure(float(np.max(scatlas_common[:, 1:])) <= 1e-7 and float(np.max(scatlas_none)) <= 1e-7, "scAtlasVAE encoder is not batch-invariant within 1e-7")
    # max_abs_dz_none_batch is intentionally not applicable/blank for scVI.
    ensure("max_abs_dz_none_batch" in scvi.columns, "scVI invariance CSV lacks max_abs_dz_none_batch")
    return {"scatlas_rows": len(scatlas), "scvi_rows": len(scvi), "scatlas_max_drift": float(np.max(scatlas_common[:, 1:]))}


def validate_invariance_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as store:
        ensure(store.files, "invariance NPZ has no arrays")
        for key in store.files:
            values = np.asarray(store[key])
            _ensure_finite_array(values, key)
            ensure(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] > 0, f"{key} has invalid shape {values.shape}")
        for key in store.files:
            if not key.endswith("_real"):
                continue
            perm_key = key[:-5] + "_perm"
            ensure(perm_key in store.files, f"missing paired array {perm_key}")
            ensure(store[key].shape == store[perm_key].shape, f"{key}/{perm_key} shapes differ")
        ensure("scAtlasVAE_real" in store.files and "scAtlasVAE_perm" in store.files, "missing scAtlasVAE real/perm arrays")
        max_abs = float(np.max(np.abs(store["scAtlasVAE_real"] - store["scAtlasVAE_perm"])))
        ensure(max_abs <= 1e-7, f"scAtlasVAE stored real/perm drift is {max_abs}")
        return {"keys": list(store.files), "scatlas_stored_max_abs_drift": max_abs}


def validate_scib_table(path: Path, expected_embeddings: set[str]) -> dict[str, Any]:
    frame = _read_csv(path)
    embedding_column = "Embedding" if "Embedding" in frame.columns else frame.columns[0]
    metric_rows = frame.loc[frame[embedding_column].astype(str) != "Metric Type"].copy()
    ensure(len(metric_rows) > 0, "no embedding metric rows")
    ensure(not metric_rows[embedding_column].isna().any(), "embedding name contains null")
    ensure(not metric_rows[embedding_column].duplicated().any(), "embedding rows are duplicated")
    actual = set(metric_rows[embedding_column].astype(str))
    ensure(expected_embeddings.issubset(actual), f"missing embedding rows: {sorted(expected_embeddings - actual)}")
    metric_columns = [column for column in frame.columns if column != embedding_column]
    ensure(metric_columns, "no metric columns")
    values = metric_rows[metric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ensure(np.isfinite(values).all(), "embedding metric rows contain non-finite values")
    ensure("Total" in metric_rows.columns, "missing Total aggregate")
    totals = pd.to_numeric(metric_rows["Total"], errors="coerce").to_numpy(float)
    _ensure_probability_metrics(totals[:, None], "Total")
    return {"embedding_rows": sorted(actual), "metric_columns": len(metric_columns)}


def validate_scalability(path: Path) -> dict[str, Any]:
    frame = _read_csv(path)
    required_numeric = [
        "n_cells",
        "fit_seconds",
        "peak_gpu_mb",
        "sec_per_epoch",
        "sec_per_10k_cells",
        "setup_and_fit_seconds",
        "data_load_seconds",
        "runtime_import_seconds",
        "model_setup_seconds",
        "load_setup_fit_seconds",
        "start_process_rss_mb",
        "peak_process_rss_mb",
        "peak_process_rss_delta_mb",
        "start_process_working_set_mb",
        "peak_process_working_set_mb",
        "peak_process_working_set_delta_mb",
        "start_process_private_mb",
        "peak_process_private_mb",
        "peak_process_private_delta_mb",
        "peak_cuda_allocated_mb",
        "peak_cuda_reserved_mb",
        "process_memory_samples",
        "process_memory_sample_interval_ms",
        "worker_pid",
    ]
    required_text = ["process_memory_backend", "process_memory_scope"]
    missing = [column for column in required_numeric + required_text if column not in frame.columns]
    ensure(not missing, f"missing new memory/timing columns: {missing}")
    values = _numeric_values(frame, required_numeric, path.name)
    ensure(len(frame) >= 4, f"expected at least four cell scales, got {len(frame)}")
    numeric = {column: pd.to_numeric(frame[column], errors="coerce").to_numpy(float) for column in required_numeric}
    ensure((numeric["n_cells"] > 0).all(), "n_cells must be positive")
    ensure(len(np.unique(numeric["n_cells"])) == len(frame), "n_cells contains duplicate scales")
    for column in ("fit_seconds", "sec_per_epoch", "sec_per_10k_cells", "peak_cuda_allocated_mb", "peak_cuda_reserved_mb", "peak_process_rss_mb", "peak_process_private_mb", "process_memory_samples"):
        ensure((numeric[column] > 0).all(), f"{column} must be positive")
    for start, peak, delta in (
        ("start_process_rss_mb", "peak_process_rss_mb", "peak_process_rss_delta_mb"),
        ("start_process_working_set_mb", "peak_process_working_set_mb", "peak_process_working_set_delta_mb"),
        ("start_process_private_mb", "peak_process_private_mb", "peak_process_private_delta_mb"),
    ):
        ensure((numeric[peak] + ATOL >= numeric[start]).all(), f"{peak} is below {start}")
        ensure((numeric[delta] >= -ATOL).all(), f"{delta} is negative")
        ensure(np.allclose(numeric[delta], np.maximum(0.0, numeric[peak] - numeric[start]), atol=1e-6, rtol=1e-6), f"{delta} does not equal peak-start")
    ensure((numeric["peak_cuda_reserved_mb"] + ATOL >= numeric["peak_cuda_allocated_mb"]).all(), "CUDA reserved peak is below allocated peak")
    ensure(np.allclose(numeric["peak_gpu_mb"], numeric["peak_cuda_allocated_mb"], atol=1e-9, rtol=1e-9), "peak_gpu_mb compatibility alias differs from peak_cuda_allocated_mb")
    ensure((numeric["worker_pid"] > 0).all(), "worker_pid must be positive")
    for column in required_text:
        ensure(not frame[column].isna().any(), f"{column} contains null")
        ensure((frame[column].astype(str).str.strip().str.len() > 0).all(), f"{column} contains empty values")
    ensure(set(frame["process_memory_scope"].astype(str)) == {"fresh_worker_load_setup_fit"}, "unexpected process_memory_scope")
    return {
        "rows": len(frame),
        "n_cells": [int(value) for value in numeric["n_cells"]],
        "numeric_columns_checked": len(required_numeric),
        "process_memory_backends": sorted(set(frame["process_memory_backend"].astype(str))),
        "peak_rss_mb_range": [float(numeric["peak_process_rss_mb"].min()), float(numeric["peak_process_rss_mb"].max())],
        "peak_private_mb_range": [float(numeric["peak_process_private_mb"].min()), float(numeric["peak_process_private_mb"].max())],
        "peak_cuda_allocated_mb_range": [float(numeric["peak_cuda_allocated_mb"].min()), float(numeric["peak_cuda_allocated_mb"].max())],
        "peak_cuda_reserved_mb_range": [float(numeric["peak_cuda_reserved_mb"].min()), float(numeric["peak_cuda_reserved_mb"].max())],
        "all_numeric_values_finite": bool(np.isfinite(values).all()),
    }


def validate_phase3_overlap(path: Path) -> dict[str, Any]:
    frame = _read_csv(path)
    required = {"official_embedding", "minimal_embedding", "k", "n_sample", "mean_knn_jaccard"}
    ensure(required.issubset(frame.columns), f"missing columns: {sorted(required - set(frame.columns))}")
    values = _numeric_values(frame, ["k", "n_sample", "mean_knn_jaccard"], path.name)
    ensure((values[:, :2] > 0).all(), "k/n_sample must be positive")
    _ensure_probability_metrics(values[:, 2, None], "mean_knn_jaccard")
    ensure(not frame[["official_embedding", "minimal_embedding"]].isna().any().any(), "embedding names contain null")
    return {"rows": len(frame), "mean_knn_jaccard": [float(value) for value in values[:, 2]]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="artifact directory; default is the current directory (run from data)",
    )
    parser.add_argument(
        "--cross-log-dir",
        type=Path,
        default=Path("remaining_pipeline_logs"),
        help="cross-atlas rerun log directory, relative to data-dir unless absolute",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("final_validation.json"),
        help="JSON summary, relative to data-dir unless absolute",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="SKIP missing required outputs while pipelines are still running; present-but-invalid outputs still FAIL",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    ensure(data_dir.is_dir(), f"data directory does not exist: {data_dir}")
    log_dir = args.cross_log_dir if args.cross_log_dir.is_absolute() else data_dir / args.cross_log_dir
    output_path = args.output if args.output.is_absolute() else data_dir / args.output
    suite = ValidationSuite(data_dir, args.allow_missing)

    paper_csv = data_dir / "phase5_transfer_results_paper.csv"
    paper_npz = data_dir / "phase5_transfer_cm_paper.npz"
    patient_csv = data_dir / "phase5_transfer_results_patient_paper.csv"
    patient_npz = data_dir / "phase5_transfer_cm_patient_paper.npz"
    fulltime_csv = data_dir / "phase5_transfer_results.csv"
    fulltime_npz = data_dir / "phase5_transfer_cm.npz"
    patient_fulltime_csv = data_dir / "phase5_transfer_results_patient_fulltime.csv"
    patient_fulltime_npz = data_dir / "phase5_transfer_cm_patient_fulltime.npz"

    expected_ab = {"A": {ZERO_SHOT, FULL_SHOT, SCVI_KNN}, "B": {ZERO_SHOT, SCVI_KNN}}
    expected_p = {"P": {ZERO_SHOT, SCVI_KNN}}
    suite.check(
        "transfer.paper.AB",
        [paper_csv, paper_npz],
        lambda: validate_transfer_pair(paper_csv, paper_npz, expected_ab),
    )
    suite.check(
        "transfer.paper.P",
        [patient_csv, patient_npz],
        lambda: validate_transfer_pair(patient_csv, patient_npz, expected_p),
    )
    suite.optional_pair(
        "transfer.fulltime.AB",
        [fulltime_csv, fulltime_npz],
        lambda: validate_transfer_pair(fulltime_csv, fulltime_npz, expected_ab),
    )
    patient_fulltime_status = data_dir / "patient_fulltime_status.json"
    patient_fulltime_stdout = data_dir / "patient_fulltime_logs" / "transfer_patient_fulltime.stdout.log"
    patient_fulltime_stderr = data_dir / "patient_fulltime_logs" / "transfer_patient_fulltime.stderr.log"
    patient_fulltime_checkpoint = data_dir / "ref_model_designP_fulltime.pt"
    suite.check(
        "transfer.fulltime.P",
        [patient_fulltime_csv, patient_fulltime_npz, patient_fulltime_status,
         patient_fulltime_stdout, patient_fulltime_stderr, patient_fulltime_checkpoint],
        lambda: {
            "artifacts": validate_transfer_pair(
                patient_fulltime_csv, patient_fulltime_npz, expected_p
            ),
            "provenance": validate_patient_fulltime_run(
                patient_fulltime_status,
                patient_fulltime_stdout,
                patient_fulltime_stderr,
                patient_fulltime_checkpoint,
            ),
        },
    )
    patient_selection = data_dir / "phase5_patient_holdout_selection.csv"
    suite.check(
        "transfer.patient_selection",
        [patient_selection, patient_csv],
        lambda: validate_patient_selection(patient_selection, patient_csv),
    )

    fair_path = data_dir / "phase5_fair_knn_results.csv"
    suite.check("fair_knn.ABP_two_modes", [fair_path], lambda: validate_fair_knn(fair_path))

    full_cross = cross_paths(data_dir, "")
    pl10_cross = cross_paths(data_dir, "_pl10")
    suite.check(
        "cross_atlas.full.outputs",
        list(full_cross.values()),
        lambda: validate_cross_outputs(full_cross, "full"),
    )
    suite.check(
        "cross_atlas.pl10.outputs",
        list(pl10_cross.values()),
        lambda: validate_cross_outputs(pl10_cross, "pl10"),
    )
    suite.check(
        "cross_atlas.shared_inputs",
        [full_cross["npz"], pl10_cross["npz"]],
        lambda: validate_cross_shared_inputs(full_cross["npz"], pl10_cross["npz"]),
    )
    for variant in ("full", "pl10"):
        log_paths = choose_cross_log_paths(log_dir, variant)
        suite.check(
            f"cross_atlas.{variant}.logs_no_nan",
            log_paths,
            lambda paths=log_paths, name=variant: validate_cross_logs(paths, name),
        )

    scatlas_invariance = data_dir / "phase5_invariance_scatlasvae.csv"
    scvi_invariance = data_dir / "phase5_invariance_scvi.csv"
    invariance_npz = data_dir / "phase5_invariance_z.npz"
    suite.check(
        "invariance.metrics",
        [scatlas_invariance, scvi_invariance],
        lambda: validate_invariance_csvs(scatlas_invariance, scvi_invariance),
    )
    suite.check(
        "invariance.latents",
        [invariance_npz],
        lambda: validate_invariance_npz(invariance_npz),
    )

    phase4 = data_dir / "phase4_ablation_results.csv"
    suite.check(
        "phase4.ablation_metrics",
        [phase4],
        lambda: validate_scib_table(phase4, {"X_nlat2", "X_nlat10", "X_nlat50", "X_nowarmup"}),
    )

    scalability = data_dir / "phase5_scalability.csv"
    suite.check(
        "scalability.complete_memory_schema",
        [scalability],
        lambda: validate_scalability(scalability),
    )

    phase3_overlap = data_dir / "phase3_knn_overlap.csv"
    suite.check(
        "phase3.knn_overlap.optional",
        [phase3_overlap],
        lambda: validate_phase3_overlap(phase3_overlap),
        required=False,
    )
    minimal_benchmark = data_dir / "phase5_minimal_bench.csv"
    suite.check(
        "phase3.minimal_benchmark.optional",
        [minimal_benchmark],
        lambda: validate_scib_table(
            minimal_benchmark,
            {"X_pca", "X_scVI", "X_scAtlasVAE_sup", "X_minimal"},
        ),
        required=False,
    )

    summary = suite.summary()
    failed = summary["fail"] > 0
    complete = not suite.required_missing_skipped
    payload = {
        "status": "FAIL" if failed else "PASS",
        "complete": complete,
        "allow_missing": bool(args.allow_missing),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": _report_path(data_dir, output_path.parent),
        "cross_log_dir": _report_path(log_dir, output_path.parent),
        "summary": summary,
        "checks": [asdict(record) for record in suite.records],
    }
    try:
        output_path.write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[FAIL] could not write validation summary {output_path}: {exc}", file=sys.stderr)
        return 1

    qualifier = " (required outputs missing but allowed)" if not complete and not failed else ""
    print(
        f"\nOVERALL {'FAIL' if failed else 'PASS'}{qualifier}: "
        f"{summary['pass']} passed, {summary['fail']} failed, {summary['skip']} skipped"
    )
    print(f"JSON: {output_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"[FAIL] startup: {exc}", file=sys.stderr)
        raise SystemExit(1)
