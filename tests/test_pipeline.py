from __future__ import annotations

import numpy as np

from hggs.compression import CompressionConfig
from hggs.data import Evidence, GaussianModel
from hggs.importance import compute_semantic_detail_score
from hggs.pipeline import HggsConfig, run_compression
from hggs.scheduler import SchedulerConfig, simulate_scheduler


def synthetic_model(n: int = 256) -> tuple[GaussianModel, Evidence]:
    rng = np.random.default_rng(4)
    position = rng.normal(size=(n, 3)).astype(np.float32)
    scale = rng.uniform(0.01, 0.04, size=(n, 3)).astype(np.float32)
    rotation = rng.normal(size=(n, 4)).astype(np.float32)
    rotation /= np.linalg.norm(rotation, axis=1, keepdims=True) + 1e-8
    opacity = rng.uniform(0.05, 0.9, size=n).astype(np.float32)
    color = rng.normal(size=(n, 16)).astype(np.float32)
    semantic = (position[:, 0] > 0.8).astype(np.int32)
    evidence = Evidence(
        visual_detail=(0.2 * rng.random(n) + 0.7 * semantic).astype(np.float32),
        mask=(0.1 * rng.random(n) + semantic).astype(np.float32),
        contribution=(opacity * rng.random(n)).astype(np.float32),
        visibility=rng.integers(1, 20, size=n).astype(np.float32),
        coverage=rng.random(n).astype(np.float32),
        projected_area=rng.uniform(0.5, 12.0, size=n).astype(np.float32),
        semantic_label=semantic,
    )
    model = GaussianModel(position, scale, rotation, opacity, color)
    return model, evidence


def test_importance_prefers_annotated_regions() -> None:
    model, evidence = synthetic_model()
    _, score = compute_semantic_detail_score(model, evidence)
    assert score[evidence.semantic_label == 1].mean() > score[evidence.semantic_label == 0].mean()


def test_pipeline_outputs_layered_asset() -> None:
    model, evidence = synthetic_model()
    config = HggsConfig(compression=CompressionConfig(low_bits=6, medium_bits=8, high_bits=10))
    asset = run_compression(model, evidence, target_mb=0.004, config=config)
    assert asset.model.n_gaussians > 0
    assert asset.model.n_gaussians < model.n_gaussians
    assert set(np.unique(asset.layer)).issubset({0, 1, 2})
    assert asset.estimated_size_mb <= 0.006
    assert np.allclose(np.linalg.norm(asset.model.rotation, axis=1), 1.0, atol=1e-4)


def test_scheduler_returns_hierarchical_layers() -> None:
    model, evidence = synthetic_model()
    asset = run_compression(model, evidence, target_mb=0.005)
    rng = np.random.default_rng(8)
    frames = rng.uniform(1.0, 10.0, size=(4, asset.model.n_gaussians)).astype(np.float32)
    cost = np.full(asset.model.n_gaussians, 0.001, dtype=np.float32)
    semantic = asset.layer == 2
    result = simulate_scheduler(
        layer=asset.layer,
        score=asset.score,
        semantic_label=semantic,
        projected_radius_frames=frames,
        per_gaussian_cost_ms=cost,
        resident_memory_mb={0: 1.0, 1: 1.0, 2: 1.0},
        free_memory_mb=8.0,
        config=SchedulerConfig(),
    )
    assert result["active_counts"].shape == (4,)
    assert np.all(result["active_layers"] >= 0)

