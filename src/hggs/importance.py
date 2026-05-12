from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from hggs.data import Evidence, GaussianModel


@dataclass(frozen=True)
class ImportanceConfig:
    lambda_visual: float = 0.25
    lambda_mask: float = 0.50
    lambda_contribution: float = 0.25
    robust_low_percentile: float = 5.0
    robust_high_percentile: float = 95.0
    neighborhood_weight: float = 0.20
    neighborhood_radius_scale: float = 3.0


def robust_normalize(
    values: np.ndarray,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
    eps: float = 1e-8,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.copy()
    lo = float(np.percentile(values, low_percentile))
    hi = float(np.percentile(values, high_percentile))
    return np.clip((values - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)


def compute_semantic_detail_score(
    model: GaussianModel,
    evidence: Evidence,
    config: ImportanceConfig = ImportanceConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw and smoothed semantic-detail scores.

    The score follows the manuscript: visual-detail evidence, annotation-mask
    evidence, and alpha-compositing contribution are robustly normalized and
    combined. The smoothed score is used for pruning, precision assignment, and
    layer allocation.
    """

    evidence.require_compatible(model)
    weight_sum = (
        config.lambda_visual + config.lambda_mask + config.lambda_contribution
    )
    if weight_sum <= 0:
        raise ValueError("importance weights must sum to a positive value")

    visual = robust_normalize(
        evidence.visual_detail,
        config.robust_low_percentile,
        config.robust_high_percentile,
    )
    mask = robust_normalize(
        evidence.mask,
        config.robust_low_percentile,
        config.robust_high_percentile,
    )
    contribution = robust_normalize(
        evidence.contribution,
        config.robust_low_percentile,
        config.robust_high_percentile,
    )
    raw = (
        config.lambda_visual * visual
        + config.lambda_mask * mask
        + config.lambda_contribution * contribution
    ) / weight_sum
    raw = np.clip(raw, 0.0, 1.0).astype(np.float32)
    smoothed = smooth_scores(model, raw, config)
    return raw, smoothed


def smooth_scores(
    model: GaussianModel,
    scores: np.ndarray,
    config: ImportanceConfig = ImportanceConfig(),
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if config.neighborhood_weight <= 0 or model.n_gaussians == 0:
        return scores.copy()

    mean_scale = np.maximum(np.mean(model.scale, axis=1), 1e-6)
    radius = config.neighborhood_radius_scale * mean_scale
    tree = cKDTree(model.position)
    out = np.empty_like(scores)

    for i, point in enumerate(model.position):
        neighbors = tree.query_ball_point(point, float(radius[i]))
        if not neighbors:
            out[i] = scores[i]
            continue
        delta = model.position[neighbors] - point
        dist2 = np.sum(delta * delta, axis=1)
        sigma = max(float(2.0 * mean_scale[i]), 1e-6)
        weights = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
        neighbor_score = float(np.sum(weights * scores[neighbors]) / (np.sum(weights) + 1e-8))
        out[i] = (
            (1.0 - config.neighborhood_weight) * scores[i]
            + config.neighborhood_weight * neighbor_score
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32)

