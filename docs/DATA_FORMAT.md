# HG-GS data format

HG-GS expects data exported from a trained 3DGS implementation. The reference
format is compressed NumPy NPZ because it is easy to inspect and deterministic.

## `gaussians.npz`

| Key | Shape | Meaning |
| --- | --- | --- |
| `position` | `(N, 3)` | Gaussian center in object/world coordinates |
| `scale` | `(N, 3)` | Gaussian scale parameters |
| `rotation` | `(N, 4)` | Unit quaternion, canonicalized during compression |
| `opacity` | `(N,)` | Gaussian opacity |
| `color` | `(N, C)` | RGB, SH, or learned color coefficients |

## `evidence.npz`

| Key | Shape | Meaning |
| --- | --- | --- |
| `visual_detail` | `(N,)` | Detail evidence projected from image gradients |
| `mask` | `(N,)` | Confidence-weighted heritage-detail mask evidence |
| `contribution` | `(N,)` | Alpha-compositing or rendering contribution evidence |
| `visibility` | `(N,)` | Visible view count or normalized visibility |
| `coverage` | `(N,)` | Optional foreground coverage contribution for base-layer selection |
| `projected_area` | `(N,)` | Optional average projected splat area |
| `semantic_label` | `(N,)` | Optional annotation label inherited by each Gaussian |

The manuscript computes `visual_detail` and `mask` as contribution-normalized
expectations over projected splat supports. If only per-Gaussian summaries are
available, export them directly with the same semantics.

