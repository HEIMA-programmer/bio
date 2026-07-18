#!/usr/bin/env python3
"""Build the Phase 3 structural-comparison evidence artifacts without training.

This script is deliberately read-only with respect to the AnnData object.  It
uses the already stored official/minimal latent spaces, their cached UMAPs and
Scanpy neighbour graphs.  Exact label ASW values are unscaled from the existing
``phase5_minimal_bench.csv`` scib-metrics output, which was computed from the
same stored embeddings; embedding digests bind that result to the current H5AD.

Run from the repository root (the scib environment has all dependencies)::

    python scripts/phase3_structure_metrics.py

Outputs:
    data/phase3_structure_metrics.csv
    data/phase3_structure_metrics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

OFFICIAL_LATENT = "X_scAtlasVAE_sup"
MINIMAL_LATENT = "X_minimal"
OFFICIAL_UMAP = "X_umap_official"
MINIMAL_UMAP = "X_umap_mine"
OFFICIAL_GRAPH = "official_distances"
MINIMAL_GRAPH = "mine_distances"
LABEL_KEY = "cell_type"
SCANPY_N_NEIGHBORS = 15
LITERAL_K = 15
UMAP_K = 30
JACCARD_K = 30


def array_digest(values: np.ndarray) -> str:
    """Stable digest including dtype and shape, matching the final validator."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def categorical(h5: h5py.File, key: str) -> tuple[np.ndarray, np.ndarray]:
    node = h5["obs"][key]
    categories = node["categories"].asstr()[...]
    codes = node["codes"][...].astype(np.int64, copy=False)
    if np.any(codes < 0):
        raise ValueError(f"obs[{key!r}] contains missing category codes")
    return categories, codes


def pairwise_centroid_distances(
    values: np.ndarray, labels: np.ndarray, n_labels: int
) -> np.ndarray:
    centroids = np.vstack([values[labels == code].mean(axis=0) for code in range(n_labels)])
    delta = centroids[:, None, :] - centroids[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    return distances[np.triu_indices(n_labels, k=1)]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    # Average ranks make this correct even if a future embedding contains ties.
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return pearson(xr, yr)


def row_purity(
    indptr: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    counts = np.diff(indptr).astype(np.int64, copy=False)
    if np.any(counts <= 0):
        raise ValueError("neighbour graph contains an empty row")
    repeated_labels = np.repeat(labels, counts)
    matches = (labels[indices] == repeated_labels).astype(np.int64, copy=False)
    cumulative = np.concatenate(([0], np.cumsum(matches, dtype=np.int64)))
    per_cell = (cumulative[indptr[1:]] - cumulative[indptr[:-1]]) / counts
    micro = float(per_cell.mean())
    macro = float(np.mean([per_cell[labels == code].mean() for code in np.unique(labels)]))
    return micro, macro, counts


def exact_knn_purity(
    values: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[float, float]:
    # k+1 includes self; remove it explicitly so ``k`` is literal non-self kNN.
    _, indices = cKDTree(values).query(values, k=k + 1, workers=-1)
    indices = indices[:, 1:]
    per_cell = (labels[indices] == labels[:, None]).mean(axis=1)
    micro = float(per_cell.mean())
    macro = float(np.mean([per_cell[labels == code].mean() for code in np.unique(labels)]))
    return micro, macro


def read_benchmark_metrics(path: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(path)
    embedding_column = frame.columns[0]
    frame = frame.loc[frame[embedding_column].astype(str) != "Metric Type"].set_index(embedding_column)
    required_rows = {OFFICIAL_LATENT, MINIMAL_LATENT}
    missing = required_rows - set(frame.index.astype(str))
    if missing:
        raise ValueError(f"{path.name} lacks benchmark row(s): {sorted(missing)}")
    required_columns = {"Silhouette label", "KMeans NMI", "KMeans ARI"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path.name} lacks column(s): {sorted(missing_columns)}")
    result: dict[str, dict[str, float]] = {}
    for embedding in (OFFICIAL_LATENT, MINIMAL_LATENT):
        scaled_asw = float(frame.loc[embedding, "Silhouette label"])
        result[embedding] = {
            "silhouette_label_scaled": scaled_asw,
            "label_asw_raw": 2.0 * scaled_asw - 1.0,
            "kmeans_nmi": float(frame.loc[embedding, "KMeans NMI"]),
            "kmeans_ari": float(frame.loc[embedding, "KMeans ARI"]),
        }
    return result


def metric_row(
    metric: str,
    scope: str,
    embedding: str,
    value: float,
    source: str,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "metric": metric,
        "scope": scope,
        "embedding": embedding,
        "value": float(value),
        "source": source,
        "detail": detail,
    }


def build_metrics(h5ad_path: Path, benchmark_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    benchmark = read_benchmark_metrics(benchmark_path)
    with h5py.File(h5ad_path, "r") as h5:
        categories, labels = categorical(h5, LABEL_KEY)
        n_obs = int(labels.size)
        n_labels = int(categories.size)
        embeddings = {
            key: h5["obsm"][key][...]
            for key in (OFFICIAL_LATENT, MINIMAL_LATENT, OFFICIAL_UMAP, MINIMAL_UMAP)
        }
        for key, values in embeddings.items():
            if values.shape[0] != n_obs:
                raise ValueError(f"obsm[{key!r}] row count does not match obs")
            if not np.isfinite(values).all():
                raise ValueError(f"obsm[{key!r}] contains non-finite values")

        graph_results: dict[str, dict[str, Any]] = {}
        for embedding, graph_key in (
            (OFFICIAL_LATENT, OFFICIAL_GRAPH),
            (MINIMAL_LATENT, MINIMAL_GRAPH),
        ):
            graph = h5["obsp"][graph_key]
            indptr = graph["indptr"][...].astype(np.int64, copy=False)
            indices = graph["indices"][...].astype(np.int64, copy=False)
            micro, macro, counts = row_purity(indptr, indices, labels)
            coarse_by_category = np.asarray([str(value).split(".")[2] for value in categories])
            coarse_labels = coarse_by_category[labels]
            coarse_micro, coarse_macro, _ = row_purity(indptr, indices, coarse_labels)
            graph_results[embedding] = {
                "graph_key": graph_key,
                "scanpy_n_neighbors_config": SCANPY_N_NEIGHBORS,
                "stored_nonself_neighbors_min": int(counts.min()),
                "stored_nonself_neighbors_max": int(counts.max()),
                "stored_nonself_neighbors_mean": float(counts.mean()),
                "fine_label_purity_micro": micro,
                "fine_label_purity_macro": macro,
                "coarse_lineage_purity_micro": coarse_micro,
                "coarse_lineage_purity_macro": coarse_macro,
            }

    latent_official_distances = pairwise_centroid_distances(
        embeddings[OFFICIAL_LATENT], labels, n_labels
    )
    latent_minimal_distances = pairwise_centroid_distances(
        embeddings[MINIMAL_LATENT], labels, n_labels
    )
    umap_official_distances = pairwise_centroid_distances(
        embeddings[OFFICIAL_UMAP], labels, n_labels
    )
    umap_minimal_distances = pairwise_centroid_distances(
        embeddings[MINIMAL_UMAP], labels, n_labels
    )

    literal_knn: dict[str, dict[str, float]] = {}
    umap_knn: dict[str, dict[str, float]] = {}
    coarse_by_category = np.asarray([str(value).split(".")[2] for value in categories])
    coarse_labels = coarse_by_category[labels]
    for embedding in (OFFICIAL_LATENT, MINIMAL_LATENT):
        micro, macro = exact_knn_purity(embeddings[embedding], labels, LITERAL_K)
        literal_knn[embedding] = {"micro": micro, "macro": macro}
    for embedding in (OFFICIAL_UMAP, MINIMAL_UMAP):
        fine_micro, fine_macro = exact_knn_purity(embeddings[embedding], labels, UMAP_K)
        coarse_micro, coarse_macro = exact_knn_purity(
            embeddings[embedding], coarse_labels, UMAP_K
        )
        umap_knn[embedding] = {
            "fine_micro": fine_micro,
            "fine_macro": fine_macro,
            "coarse_micro": coarse_micro,
            "coarse_macro": coarse_macro,
        }

    label_fractions = np.bincount(labels, minlength=n_labels).astype(float) / n_obs
    random_label_baseline = float(np.sum(label_fractions**2))
    random_jaccard_approx = float(JACCARD_K / (2 * (n_obs - 1) - JACCARD_K))
    correlations = {
        "latent_centroid_distance_pearson": pearson(
            latent_official_distances, latent_minimal_distances
        ),
        "latent_centroid_distance_spearman": spearman(
            latent_official_distances, latent_minimal_distances
        ),
        "umap_centroid_distance_pearson": pearson(
            umap_official_distances, umap_minimal_distances
        ),
        "umap_centroid_distance_spearman": spearman(
            umap_official_distances, umap_minimal_distances
        ),
    }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": "read_only_existing_embeddings_no_training",
        "inputs": {
            "h5ad": h5ad_path.name,
            "benchmark_csv": benchmark_path.name,
            "label_key": LABEL_KEY,
            "n_obs": n_obs,
            "n_labels": n_labels,
            "label_categories": [str(value) for value in categories],
            "embedding_sha256": {
                key: array_digest(values) for key, values in embeddings.items()
            },
        },
        "centroid_distance_correlations": correlations,
        "label_asw": {
            "source": "phase5_minimal_bench.csv scib-metrics Silhouette label; raw=2*scaled-1",
            OFFICIAL_LATENT: benchmark[OFFICIAL_LATENT],
            MINIMAL_LATENT: benchmark[MINIMAL_LATENT],
        },
        "scanpy_neighbor_graph": graph_results,
        "literal_15_nonself_knn": {
            "k": LITERAL_K,
            "method": "scipy.spatial.cKDTree, self removed explicitly",
            **literal_knn,
        },
        "umap_30_nonself_knn": {
            "k": UMAP_K,
            "method": "scipy.spatial.cKDTree, self removed explicitly",
            **umap_knn,
        },
        "random_baselines": {
            "same_label_probability_sum_p_squared": random_label_baseline,
            "independent_30nn_jaccard_ratio_of_expectations_approx": random_jaccard_approx,
            "jaccard_k": JACCARD_K,
        },
    }

    rows: list[dict[str, Any]] = []
    for name, value in correlations.items():
        rows.append(metric_row(name, "official_vs_minimal", "pair", value, "H5AD obsm centroids"))
    for embedding in (OFFICIAL_LATENT, MINIMAL_LATENT):
        rows.extend(
            [
                metric_row(
                    "label_asw_raw",
                    "latent",
                    embedding,
                    benchmark[embedding]["label_asw_raw"],
                    "phase5_minimal_bench.csv",
                    "raw=2*Silhouette label-1",
                ),
                metric_row(
                    "scanpy_neighbor_graph_label_purity_micro",
                    "latent",
                    embedding,
                    graph_results[embedding]["fine_label_purity_micro"],
                    f"H5AD obsp/{graph_results[embedding]['graph_key']}",
                    "Scanpy n_neighbors=15; stored graph has 14 non-self neighbours per cell",
                ),
                metric_row(
                    "scanpy_neighbor_graph_coarse_purity_micro",
                    "latent",
                    embedding,
                    graph_results[embedding]["coarse_lineage_purity_micro"],
                    f"H5AD obsp/{graph_results[embedding]['graph_key']}",
                ),
                metric_row(
                    "literal_15_nonself_knn_label_purity_micro",
                    "latent",
                    embedding,
                    literal_knn[embedding]["micro"],
                    "H5AD obsm + scipy cKDTree",
                ),
                metric_row(
                    "kmeans_nmi",
                    "latent",
                    embedding,
                    benchmark[embedding]["kmeans_nmi"],
                    "phase5_minimal_bench.csv",
                ),
                metric_row(
                    "kmeans_ari",
                    "latent",
                    embedding,
                    benchmark[embedding]["kmeans_ari"],
                    "phase5_minimal_bench.csv",
                ),
            ]
        )
    for embedding in (OFFICIAL_UMAP, MINIMAL_UMAP):
        rows.extend(
            [
                metric_row(
                    "umap_30_nonself_knn_label_purity_micro",
                    "umap",
                    embedding,
                    umap_knn[embedding]["fine_micro"],
                    "H5AD obsm + scipy cKDTree",
                ),
                metric_row(
                    "umap_30_nonself_knn_coarse_purity_micro",
                    "umap",
                    embedding,
                    umap_knn[embedding]["coarse_micro"],
                    "H5AD obsm + scipy cKDTree",
                ),
            ]
        )
    rows.extend(
        [
            metric_row(
                "same_label_random_baseline",
                "dataset",
                LABEL_KEY,
                random_label_baseline,
                "H5AD obs label frequencies",
                "sum of squared class fractions",
            ),
            metric_row(
                "independent_30nn_jaccard_random_approx",
                "dataset",
                "random_neighbour_sets",
                random_jaccard_approx,
                "analytic approximation",
                "ratio of expected intersection and union",
            ),
        ]
    )
    return payload, pd.DataFrame(rows)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_DATA_DIR / "tcell_processed.h5ad")
    parser.add_argument(
        "--benchmark", type=Path, default=DEFAULT_DATA_DIR / "phase5_minimal_bench.csv"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=DEFAULT_DATA_DIR / "phase3_structure_metrics.csv"
    )
    parser.add_argument(
        "--output-json", type=Path, default=DEFAULT_DATA_DIR / "phase3_structure_metrics.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, frame = build_metrics(args.h5ad.resolve(), args.benchmark.resolve())
    atomic_write_text(args.output_csv.resolve(), frame.to_csv(index=False, lineterminator="\n"))
    atomic_write_text(
        args.output_json.resolve(),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(f"wrote {args.output_csv.resolve()}")
    print(f"wrote {args.output_json.resolve()}")
    print(json.dumps(payload["centroid_distance_correlations"], indent=2))


if __name__ == "__main__":
    main()
