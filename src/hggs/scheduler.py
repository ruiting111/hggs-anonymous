from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import numpy as np


@dataclass(frozen=True)
class SchedulerConfig:
    target_frame_ms: float = 16.67
    tracking_ms: float = 3.10
    camera_ms: float = 2.40
    compose_ms: float = 0.00
    safety_margin_ms: float = 1.50
    beta_semantic: float = 0.45
    beta_radius: float = 0.25
    beta_label: float = 0.20
    beta_cost: float = 0.10
    hysteresis_weight: float = 0.15
    radius_norm_px: float = 8.0
    activation_radius_px: float = 4.0


def available_renderer_budget_ms(config: SchedulerConfig) -> float:
    return (
        config.target_frame_ms
        - config.tracking_ms
        - config.camera_ms
        - config.compose_ms
        - config.safety_margin_ms
    )


def choose_active_layers(
    layer: np.ndarray,
    score: np.ndarray,
    projected_radius: np.ndarray,
    semantic_label: np.ndarray,
    per_gaussian_cost_ms: np.ndarray,
    resident_memory_mb: Mapping[int, float],
    free_memory_mb: float,
    previously_active: Iterable[int] = (0,),
    config: SchedulerConfig = SchedulerConfig(),
) -> List[int]:
    """Select hierarchical active layers for one AR frame."""

    budget = available_renderer_budget_ms(config)
    if budget <= 0:
        return [0]
    previous = set(previously_active)
    candidates = [[0], [0, 1], [0, 2], [0, 1, 2]]
    best_layers = [0]
    best_utility = -np.inf
    for layers in candidates:
        layers_set = set(layers)
        active = np.isin(layer, layers) & (layer >= 0)
        detail = layer > 0
        active &= (~detail) | (projected_radius >= config.activation_radius_px)
        cost = float(np.sum(per_gaussian_cost_ms[active]))
        memory = float(sum(resident_memory_mb.get(int(l), 0.0) for l in layers))
        if cost > budget or memory > free_memory_mb:
            continue
        utility = _layer_utility(
            active,
            layers_set,
            previous,
            score,
            projected_radius,
            semantic_label,
            per_gaussian_cost_ms,
            config,
        )
        if utility > best_utility:
            best_utility = utility
            best_layers = layers
    return best_layers


def simulate_scheduler(
    layer: np.ndarray,
    score: np.ndarray,
    semantic_label: np.ndarray,
    projected_radius_frames: np.ndarray,
    per_gaussian_cost_ms: np.ndarray,
    resident_memory_mb: Mapping[int, float],
    free_memory_mb: float,
    config: SchedulerConfig = SchedulerConfig(),
) -> Dict[str, np.ndarray]:
    active_layers = []
    active_counts = []
    frame_cost = []
    previous = [0]
    for radii in projected_radius_frames:
        current = choose_active_layers(
            layer=layer,
            score=score,
            projected_radius=radii,
            semantic_label=semantic_label,
            per_gaussian_cost_ms=per_gaussian_cost_ms,
            resident_memory_mb=resident_memory_mb,
            free_memory_mb=free_memory_mb,
            previously_active=previous,
            config=config,
        )
        mask = np.isin(layer, current) & (layer >= 0)
        mask &= (layer == 0) | (radii >= config.activation_radius_px)
        active_layers.append(current)
        active_counts.append(int(np.sum(mask)))
        frame_cost.append(float(np.sum(per_gaussian_cost_ms[mask])))
        previous = current
    return {
        "active_counts": np.asarray(active_counts, dtype=np.int32),
        "frame_cost_ms": np.asarray(frame_cost, dtype=np.float32),
        "active_layers": np.asarray([max(x) for x in active_layers], dtype=np.int32),
    }


def _layer_utility(
    active: np.ndarray,
    layers: set[int],
    previous: set[int],
    score: np.ndarray,
    projected_radius: np.ndarray,
    semantic_label: np.ndarray,
    per_gaussian_cost_ms: np.ndarray,
    config: SchedulerConfig,
) -> float:
    radius_term = np.minimum(1.0, projected_radius / max(config.radius_norm_px, 1e-6))
    priority = (
        config.beta_semantic * score
        + config.beta_radius * radius_term
        + config.beta_label * semantic_label.astype(np.float32)
        - config.beta_cost * per_gaussian_cost_ms
    )
    switch_penalty = config.hysteresis_weight * len(layers.symmetric_difference(previous))
    return float(np.sum(priority[active]) / (np.sum(per_gaussian_cost_ms[active]) + 1e-8) - switch_penalty)
