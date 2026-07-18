#!/usr/bin/env python3
"""Validate that canonical report PNGs match the current generator and inputs.

``build_real.py all`` writes ``data/figure_manifest.json`` after every canonical
figure has been rendered successfully.  This validator is deliberately read-only:
it checks generator/input/output hashes, PNG dimensions, the complete target set,
and that every canonical PNG is referenced by at least one report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


EXPECTED_TARGETS = {
    "loss",
    "bench",
    "ablation",
    "umap_integration",
    "umap_compare",
    "bench_minimal",
    "transfer",
    "transfer_protocol_p",
    "invariance",
    "scalability",
    "cross_atlas",
}


class FigureManifestError(AssertionError):
    """Raised when the canonical figure evidence chain is stale or incomplete."""


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise FigureManifestError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    _ensure(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR",
            f"not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def _within_project(project_root: Path, relative_path: str) -> Path:
    _ensure(relative_path and not Path(relative_path).is_absolute(),
            f"manifest path must be relative: {relative_path!r}")
    candidate = (project_root / relative_path).resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as exc:
        raise FigureManifestError(f"manifest path escapes project: {relative_path}") from exc
    return candidate


def validate_figure_manifest(data_dir: Path, project_root: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    project_root = (project_root or data_dir.parent).resolve()
    manifest_path = data_dir / "figure_manifest.json"
    _ensure(manifest_path.is_file(), f"missing figure manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _ensure(manifest.get("schema_version") == 1, "unexpected figure manifest schema")

    generator = manifest.get("generator", {})
    generator_path = _within_project(project_root, str(generator.get("path", "")))
    _ensure(generator_path.is_file(), f"missing figure generator: {generator_path}")
    _ensure(_sha256(generator_path) == generator.get("sha256"),
            "build_real.py changed after canonical figures were rendered")

    sources = manifest.get("sources")
    _ensure(isinstance(sources, dict) and sources, "figure manifest has no source records")
    source_details: dict[str, dict[str, Any]] = {}
    for relative_path, expected in sources.items():
        source_path = _within_project(project_root, relative_path)
        _ensure(source_path.is_file(), f"missing figure source: {source_path}")
        actual_size = source_path.stat().st_size
        actual_hash = _sha256(source_path)
        _ensure(actual_size == int(expected.get("bytes", -1)),
                f"figure source size changed: {relative_path}")
        _ensure(actual_hash == expected.get("sha256"),
                f"figure source hash changed: {relative_path}")
        source_details[relative_path] = {"bytes": actual_size, "sha256": actual_hash}

    figures = manifest.get("figures")
    _ensure(isinstance(figures, list), "figure manifest has no figure list")
    targets = [str(record.get("target", "")) for record in figures]
    _ensure(len(targets) == len(set(targets)), "duplicate targets in figure manifest")
    _ensure(set(targets) == EXPECTED_TARGETS,
            f"figure target set mismatch: {sorted(set(targets) ^ EXPECTED_TARGETS)}")

    report_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((project_root / "reports").glob("*.md"))
    )
    figure_details = []
    for record in figures:
        relative_path = str(record.get("path", ""))
        figure_path = _within_project(project_root, relative_path)
        _ensure(figure_path.is_file(), f"missing canonical figure: {figure_path}")
        width, height = _png_dimensions(figure_path)
        actual_size = figure_path.stat().st_size
        actual_hash = _sha256(figure_path)
        _ensure(width == int(record.get("width", -1)) and height == int(record.get("height", -1)),
                f"PNG dimensions changed: {relative_path}")
        _ensure(width >= 600 and height >= 300,
                f"canonical PNG is unexpectedly small: {relative_path} ({width}x{height})")
        _ensure(actual_size == int(record.get("bytes", -1)),
                f"PNG size changed: {relative_path}")
        _ensure(actual_hash == record.get("sha256"),
                f"PNG hash changed: {relative_path}")
        referenced_sources = record.get("sources", [])
        _ensure(referenced_sources and set(referenced_sources).issubset(sources),
                f"invalid source references for {relative_path}")
        markdown_reference = f"figures/{figure_path.name}"
        _ensure(markdown_reference in report_text,
                f"canonical figure is not referenced by any report: {relative_path}")
        figure_details.append({
            "target": record["target"],
            "path": relative_path,
            "width": width,
            "height": height,
            "bytes": actual_size,
            "sha256": actual_hash,
        })

    return {
        "manifest": str(manifest_path),
        "generator": str(generator_path.relative_to(project_root)),
        "generator_sha256": generator["sha256"],
        "n_sources": len(source_details),
        "n_figures": len(figure_details),
        "figures": figure_details,
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    details = validate_figure_manifest(args.data_dir)
    print(f"PASS: {details['n_figures']} canonical figures and {details['n_sources']} sources verified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FigureManifestError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
