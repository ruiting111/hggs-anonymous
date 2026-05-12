from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from hggs.compression import CompressedAsset
from hggs.data import Evidence, GaussianModel


def load_gaussians(path: str | Path) -> GaussianModel:
    with np.load(path) as data:
        return GaussianModel(
            position=data["position"],
            scale=data["scale"],
            rotation=data["rotation"],
            opacity=data["opacity"],
            color=data["color"],
        )


def save_gaussians(model: GaussianModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **model.as_dict())


def load_evidence(path: str | Path) -> Evidence:
    with np.load(path) as data:
        mapping = {key: data[key] for key in data.files}
    return Evidence.from_mapping(mapping)


def save_evidence(evidence: Evidence, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **evidence.as_dict())


def save_asset(asset: CompressedAsset, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_gaussians(asset.model, directory / "gaussians_compressed.npz")
    np.savez_compressed(
        directory / "metadata_arrays.npz",
        retained_indices=asset.retained_indices,
        raw_score=asset.raw_score,
        score=asset.score,
        precision_group=asset.precision_group,
        layer=asset.layer,
    )
    metadata = {
        "estimated_size_mb": asset.estimated_size_mb,
        "summary": asset.summary(),
        "metadata": dict(asset.metadata),
    }
    write_json(metadata, directory / "metadata.json")


def load_asset(directory: str | Path) -> CompressedAsset:
    directory = Path(directory)
    model = load_gaussians(directory / "gaussians_compressed.npz")
    with np.load(directory / "metadata_arrays.npz") as arrays:
        retained_indices = arrays["retained_indices"]
        raw_score = arrays["raw_score"]
        score = arrays["score"]
        precision_group = arrays["precision_group"]
        layer = arrays["layer"]
    metadata = read_json(directory / "metadata.json")
    return CompressedAsset(
        model=model,
        retained_indices=retained_indices,
        raw_score=raw_score,
        score=score,
        precision_group=precision_group,
        layer=layer,
        estimated_size_mb=float(metadata["estimated_size_mb"]),
        metadata=metadata.get("metadata", {}),
    )


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

