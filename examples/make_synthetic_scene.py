from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_scene(out: Path, n: int = 5000, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    theta = rng.uniform(0, 2 * np.pi, size=n)
    radius = 0.6 + 0.25 * rng.random(n)
    z = rng.normal(0.0, 0.18, size=n)
    position = np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta), z]
    ).astype(np.float32)
    scale = rng.uniform(0.003, 0.025, size=(n, 3)).astype(np.float32)
    rotation = rng.normal(size=(n, 4)).astype(np.float32)
    rotation /= np.linalg.norm(rotation, axis=1, keepdims=True) + 1e-8
    opacity = rng.uniform(0.05, 0.95, size=n).astype(np.float32)
    color = rng.normal(0.0, 0.25, size=(n, 48)).astype(np.float32)

    # Simulate a band of annotated inscriptions and relief ridges.
    inscription = (np.abs(position[:, 2]) < 0.045) & (position[:, 0] > 0)
    relief = (np.sin(theta * 8) > 0.92) & (position[:, 1] < 0.2)
    semantic_label = (inscription | relief).astype(np.int32)
    visual_detail = (
        0.25 * rng.random(n)
        + 0.55 * semantic_label
        + 0.20 * np.clip(np.linalg.norm(position[:, :2], axis=1), 0, 1)
    ).astype(np.float32)
    mask = (0.10 * rng.random(n) + 0.90 * semantic_label).astype(np.float32)
    contribution = (
        opacity * (0.2 + 0.8 * rng.random(n)) * (1.0 + 0.3 * visual_detail)
    ).astype(np.float32)
    visibility = rng.integers(1, 64, size=n).astype(np.float32)
    coverage = (visibility * np.maximum(scale.mean(axis=1), 1e-4)).astype(np.float32)
    projected_area = (rng.uniform(0.5, 20.0, size=n) * (1.0 + semantic_label)).astype(np.float32)

    np.savez_compressed(
        out / "gaussians.npz",
        position=position,
        scale=scale,
        rotation=rotation,
        opacity=opacity,
        color=color,
    )
    np.savez_compressed(
        out / "evidence.npz",
        visual_detail=visual_detail,
        mask=mask,
        contribution=contribution,
        visibility=visibility,
        coverage=coverage,
        projected_area=projected_area,
        semantic_label=semantic_label,
    )
    print(f"Wrote synthetic HG-GS inputs to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("demo_scene"))
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    make_scene(args.out, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()

