from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

from hggs.data import Evidence, GaussianModel


@dataclass(frozen=True)
class CompressionConfig:
    semantic_loss_weight: float = 2.0
    semantic_floor_weight: float = 0.15
    overlap_weight: float = 0.30
    invisibility_weight: float = 0.20
    size_tolerance_mb: float = 0.002
    precision_lambda: float = 0.02
    precision_semantic_boost: float = 1.0
    low_bits: int = 8
    medium_bits: int = 12
    high_bits: int = 16


@dataclass(frozen=True)
class CompressedAsset:
    model: GaussianModel
    retained_indices: np.ndarray
    raw_score: np.ndarray
    score: np.ndarray
    precision_group: np.ndarray
    layer: np.ndarray
    estimated_size_mb: float
    metadata: Mapping[str, float]

    def summary(self) -> Dict[str, float]:
        unique, counts = np.unique(self.layer, return_counts=True)
        layer_counts = {f"layer_{int(k)}": int(v) for k, v in zip(unique, counts)}
        out: Dict[str, float] = {
            "gaussians": int(self.model.n_gaussians),
            "estimated_size_mb": float(self.estimated_size_mb),
            "mean_score": float(np.mean(self.score)) if self.score.size else 0.0,
        }
        out.update(layer_counts)
        out.update({k: float(v) for k, v in self.metadata.items()})
        return out


def estimate_attribute_bytes(
    model: GaussianModel,
    bits: np.ndarray,
    include_layer_metadata: bool = True,
) -> np.ndarray:
    """Estimate per-Gaussian serialized bytes for scalar-quantized attributes."""

    bits = np.asarray(bits, dtype=np.float32)
    n_scalars = 3 + 3 + 4 + 1 + model.color.shape[1]
    attribute_bytes = np.ceil(n_scalars * bits / 8.0)
    metadata_bytes = 2.0 if include_layer_metadata else 0.0
    return (attribute_bytes + metadata_bytes).astype(np.float32)


def estimate_size_mb(model: GaussianModel, bits: np.ndarray) -> float:
    return float(np.sum(estimate_attribute_bytes(model, bits)) / (1024.0 * 1024.0))


def prune_to_budget(
    model: GaussianModel,
    evidence: Evidence,
    score: np.ndarray,
    target_mb: float,
    config: CompressionConfig = CompressionConfig(),
) -> np.ndarray:
    """Return a boolean retained mask under the target size budget."""

    evidence.require_compatible(model)
    n = model.n_gaussians
    if n == 0:
        return np.zeros(0, dtype=bool)

    medium_bits = np.full(n, config.medium_bits, dtype=np.float32)
    per_gaussian_bytes = estimate_attribute_bytes(model, medium_bits)
    target_bytes = max(float(target_mb) * 1024.0 * 1024.0, 1.0)

    rec_damage = np.asarray(evidence.contribution, dtype=np.float32)
    sem_damage = np.asarray(evidence.mask, dtype=np.float32)
    visibility = np.asarray(evidence.visibility, dtype=np.float32)
    visibility_norm = visibility / (np.max(visibility) + 1e-8)
    removal_damage = (
        rec_damage
        + config.semantic_loss_weight * sem_damage
        + config.semantic_floor_weight * score * visibility_norm
    )
    saved_rate = per_gaussian_bytes
    overlap = (
        np.asarray(evidence.overlap, dtype=np.float32)
        if evidence.overlap is not None
        else np.zeros(n, dtype=np.float32)
    )
    overlap_norm = overlap / (np.percentile(overlap, 95) + 1e-8)
    keep_priority = (
        removal_damage / (saved_rate + 1e-8)
        - config.overlap_weight * np.clip(overlap_norm, 0.0, 1.0)
        - config.invisibility_weight * (1.0 - visibility_norm)
    )

    # Retain high-priority Gaussians until the estimated medium-precision budget is met.
    order = np.argsort(-keep_priority)
    cumulative = np.cumsum(saved_rate[order])
    keep_count = int(np.searchsorted(cumulative, target_bytes, side="right"))
    keep_count = max(1, min(n, keep_count))
    retained = np.zeros(n, dtype=bool)
    retained[order[:keep_count]] = True
    return retained


def assign_precision_groups(
    model: GaussianModel,
    retained: np.ndarray,
    score: np.ndarray,
    evidence: Evidence,
    target_mb: Optional[float] = None,
    config: CompressionConfig = CompressionConfig(),
) -> np.ndarray:
    """Assign low/medium/high precision groups to retained Gaussians.

    Groups are encoded as 0, 1, and 2. The rule is deterministic and uses the
    semantic-detail score, projected area, and visibility as a proxy for measured
    distortion sensitivity.
    """

    retained = np.asarray(retained, dtype=bool)
    n = model.n_gaussians
    group = np.full(n, -1, dtype=np.int32)
    if not np.any(retained):
        return group

    projected_area = (
        evidence.projected_area
        if evidence.projected_area is not None
        else np.mean(model.scale, axis=1)
    )
    area_norm = projected_area / (np.percentile(projected_area, 95) + 1e-8)
    vis_norm = evidence.visibility / (np.max(evidence.visibility) + 1e-8)
    sensitivity = (
        score * (1.0 + config.precision_semantic_boost)
        + 0.30 * np.clip(area_norm, 0.0, 1.0)
        + 0.20 * vis_norm
    )
    low_cut = float(np.percentile(sensitivity[retained], 40))
    high_cut = float(np.percentile(sensitivity[retained], 78))
    group[retained & (sensitivity < low_cut)] = 0
    group[retained & (sensitivity >= low_cut) & (sensitivity < high_cut)] = 1
    group[retained & (sensitivity >= high_cut)] = 2

    if target_mb is not None:
        group = _relax_precision_to_budget(model, retained, group, sensitivity, target_mb, config)
    return group


def bits_for_groups(groups: np.ndarray, config: CompressionConfig = CompressionConfig()) -> np.ndarray:
    bits = np.zeros(groups.shape, dtype=np.float32)
    bits[groups == 0] = config.low_bits
    bits[groups == 1] = config.medium_bits
    bits[groups == 2] = config.high_bits
    return bits


def quantize_model(
    model: GaussianModel,
    retained: np.ndarray,
    precision_group: np.ndarray,
    config: CompressionConfig = CompressionConfig(),
) -> GaussianModel:
    subset = model.subset(retained)
    groups = precision_group[retained]
    return GaussianModel(
        position=_quantize_array(subset.position, groups, config),
        scale=_quantize_array(subset.scale, groups, config),
        rotation=_renormalize_quaternions(_quantize_array(subset.rotation, groups, config)),
        opacity=_quantize_array(subset.opacity[:, None], groups, config)[:, 0],
        color=_quantize_array(subset.color, groups, config),
    )


def _relax_precision_to_budget(
    model: GaussianModel,
    retained: np.ndarray,
    group: np.ndarray,
    sensitivity: np.ndarray,
    target_mb: float,
    config: CompressionConfig,
) -> np.ndarray:
    out = group.copy()
    retained_indices = np.flatnonzero(retained)
    while True:
        bits = bits_for_groups(out[retained], config)
        current = estimate_size_mb(model.subset(retained), bits)
        if current <= target_mb + config.size_tolerance_mb:
            return out
        candidates = retained_indices[out[retained_indices] > 0]
        if candidates.size == 0:
            return out
        demote = candidates[np.argmin(sensitivity[candidates])]
        out[demote] -= 1


def _quantize_array(
    values: np.ndarray,
    groups: np.ndarray,
    config: CompressionConfig,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    bit_lookup = {0: config.low_bits, 1: config.medium_bits, 2: config.high_bits}
    for group, bits in bit_lookup.items():
        mask = groups == group
        if not np.any(mask):
            continue
        block = values[mask]
        vmin = np.min(block, axis=0, keepdims=True)
        vmax = np.max(block, axis=0, keepdims=True)
        levels = float(2**bits - 1)
        step = (vmax - vmin) / max(levels, 1.0)
        out[mask] = vmin + step * np.round((block - vmin) / (step + 1e-8))
    return out.astype(np.float32)


def _renormalize_quaternions(rotation: np.ndarray) -> np.ndarray:
    rotation = rotation.copy()
    sign = np.where(rotation[:, :1] < 0.0, -1.0, 1.0)
    rotation *= sign
    norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    return (rotation / (norm + 1e-8)).astype(np.float32)
