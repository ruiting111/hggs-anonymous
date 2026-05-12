from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hggs.data import Evidence, GaussianModel


@dataclass(frozen=True)
class LodConfig:
    base_coverage: float = 0.95
    semantic_threshold: float = 0.70


def assign_lod_layers(
    model: GaussianModel,
    evidence: Evidence,
    retained: np.ndarray,
    score: np.ndarray,
    config: LodConfig = LodConfig(),
) -> np.ndarray:
    """Assign retained Gaussians to base, detail, and annotation-detail layers."""

    evidence.require_compatible(model)
    retained = np.asarray(retained, dtype=bool)
    layer = np.full(model.n_gaussians, -1, dtype=np.int32)
    if not np.any(retained):
        return layer

    coverage = evidence.coverage if evidence.coverage is not None else evidence.visibility
    coverage = np.maximum(np.asarray(coverage, dtype=np.float32), 0.0)
    retained_indices = np.flatnonzero(retained)
    order = retained_indices[np.argsort(-coverage[retained_indices])]
    total_coverage = float(np.sum(coverage[retained_indices]) + 1e-8)
    cumulative = np.cumsum(coverage[order]) / total_coverage
    base_count = int(np.searchsorted(cumulative, config.base_coverage, side="left")) + 1
    base = order[: min(base_count, order.size)]
    layer[base] = 0

    remaining = retained & (layer < 0)
    semantic_label = (
        evidence.semantic_label.astype(bool)
        if evidence.semantic_label is not None
        else score >= config.semantic_threshold
    )
    semantic_detail = remaining & semantic_label & (score >= config.semantic_threshold)
    geometry_detail = remaining & ~semantic_detail
    layer[geometry_detail] = 1
    layer[semantic_detail] = 2
    return layer

