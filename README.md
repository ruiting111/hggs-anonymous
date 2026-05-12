# HG-GS: annotation-guided 3D Gaussian compression

HG-GS is a reference implementation of the annotation-guided compression pipeline
described in the manuscript:

> Annotation-guided 3D Gaussian compression improves preservation of heritage
> details for mobile augmented reality.

The code converts a trained 3D Gaussian Splatting asset plus multi-view
heritage-detail annotations into a compact, layered representation. The
implementation focuses on reproducible compression decisions:

- aggregate image-detail, annotation, and alpha-compositing evidence into a
  Gaussian-level semantic-detail score;
- prune redundant Gaussians under a target serialized-size budget;
- assign attribute precision groups with a semantic-detail-aware rule;
- partition retained primitives into base, geometry-detail, and annotation-detail
  layers;
- simulate a lightweight mobile AR scheduler over a recorded trajectory.

This repository is designed as a clean research package. It does not include a
full differentiable 3DGS trainer or a device-specific Metal/ARKit renderer.
Instead, it operates on exported Gaussian attributes and visibility/evidence
arrays that can be produced by an existing 3DGS pipeline.

## Repository layout

```text
src/hggs/
  data.py              Array schemas and validation
  io.py                NPZ/JSON loading and saving
  importance.py        Semantic-detail score estimation
  compression.py       Pruning and precision assignment
  lod.py               Progressive layer assignment
  scheduler.py         Mobile AR layer scheduler
  metrics.py           PSNR, masked PSNR, edge F1, runtime summaries
  pipeline.py          End-to-end compression pipeline
  cli.py               Command-line interface
configs/
  default.yaml         Default paper-style hyperparameters
examples/
  make_synthetic_scene.py
tests/
  test_pipeline.py
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Quick start

Generate a small synthetic exported scene:

```bash
python examples/make_synthetic_scene.py --out demo_scene
```

Run HG-GS compression:

```bash
hggs-compress \
  --gaussians demo_scene/gaussians.npz \
  --evidence demo_scene/evidence.npz \
  --config configs/default.yaml \
  --target-mb 0.12 \
  --out demo_scene/hggs_asset
```

Inspect the output:

```bash
python - <<'PY'
from hggs.io import load_asset
asset = load_asset("demo_scene/hggs_asset")
print(asset.summary())
PY
```

## Expected input format

`gaussians.npz` contains one row per Gaussian:

- `position`: `(N, 3)` float32
- `scale`: `(N, 3)` float32
- `rotation`: `(N, 4)` float32 unit quaternions
- `opacity`: `(N,)` float32
- `color`: `(N, C)` float32 spherical-harmonic or color coefficients

`evidence.npz` contains exported per-Gaussian evidence accumulated from training
views:

- `visual_detail`: `(N,)` normalized image-detail evidence
- `mask`: `(N,)` heritage-detail mask evidence
- `contribution`: `(N,)` alpha-compositing/rendering contribution evidence
- `visibility`: `(N,)` visible-view counts or normalized visibility
- optional `coverage`: `(N,)` foreground coverage contribution
- optional `projected_area`: `(N,)` average projected splat area
- optional `semantic_label`: `(N,)` boolean or integer annotation labels

If your renderer exports per-pixel splat ownership, use it to compute these
arrays before running this package. The package also contains a normalized
aggregation helper for per-Gaussian evidence arrays.

## Notes for Scientific Reports submission

Before submission, add a release archive or DOI and update the manuscript with:

- repository URL and commit hash;
- license;
- data-access conditions for restricted original artifact images;
- exact ethics approval or waiver information for the user study.

