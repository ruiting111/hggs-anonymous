from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np


def _as_float32(name: str, value: np.ndarray, ndim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {arr.ndim}")
    return arr


def _as_int32(name: str, value: np.ndarray, ndim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int32)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {arr.ndim}")
    return arr


@dataclass(frozen=True)
class GaussianModel:
    """Exported Gaussian attributes used by HG-GS.

    The class deliberately avoids renderer-specific fields. A 3DGS training
    pipeline can export spherical-harmonic coefficients or compact RGB features
    into the `color` matrix.
    """

    position: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray
    opacity: np.ndarray
    color: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _as_float32("position", self.position, 2))
        object.__setattr__(self, "scale", _as_float32("scale", self.scale, 2))
        object.__setattr__(self, "rotation", _as_float32("rotation", self.rotation, 2))
        object.__setattr__(self, "opacity", _as_float32("opacity", self.opacity, 1))
        object.__setattr__(self, "color", _as_float32("color", self.color, 2))
        n = self.position.shape[0]
        expected = {
            "position": (n, 3),
            "scale": (n, 3),
            "rotation": (n, 4),
            "opacity": (n,),
        }
        for name, shape in expected.items():
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
        if self.color.shape[0] != n:
            raise ValueError("color must have the same first dimension as position")

    @property
    def n_gaussians(self) -> int:
        return int(self.position.shape[0])

    def subset(self, mask: np.ndarray) -> "GaussianModel":
        mask = np.asarray(mask, dtype=bool)
        return GaussianModel(
            position=self.position[mask],
            scale=self.scale[mask],
            rotation=self.rotation[mask],
            opacity=self.opacity[mask],
            color=self.color[mask],
        )

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "position": self.position,
            "scale": self.scale,
            "rotation": self.rotation,
            "opacity": self.opacity,
            "color": self.color,
        }


@dataclass(frozen=True)
class Evidence:
    """Per-Gaussian evidence projected from training views."""

    visual_detail: np.ndarray
    mask: np.ndarray
    contribution: np.ndarray
    visibility: np.ndarray
    coverage: Optional[np.ndarray] = None
    projected_area: Optional[np.ndarray] = None
    semantic_label: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        fields = {
            "visual_detail": _as_float32("visual_detail", self.visual_detail, 1),
            "mask": _as_float32("mask", self.mask, 1),
            "contribution": _as_float32("contribution", self.contribution, 1),
            "visibility": _as_float32("visibility", self.visibility, 1),
        }
        n = fields["visual_detail"].shape[0]
        for name, arr in fields.items():
            if arr.shape[0] != n:
                raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}")
            object.__setattr__(self, name, arr)
        for optional_name in ("coverage", "projected_area"):
            value = getattr(self, optional_name)
            if value is not None:
                arr = _as_float32(optional_name, value, 1)
                if arr.shape[0] != n:
                    raise ValueError(f"{optional_name} must have length {n}")
                object.__setattr__(self, optional_name, arr)
        if self.semantic_label is not None:
            label = _as_int32("semantic_label", self.semantic_label, 1)
            if label.shape[0] != n:
                raise ValueError("semantic_label must have the same length as evidence")
            object.__setattr__(self, "semantic_label", label)

    @property
    def n_gaussians(self) -> int:
        return int(self.visual_detail.shape[0])

    def require_compatible(self, model: GaussianModel) -> None:
        if model.n_gaussians != self.n_gaussians:
            raise ValueError(
                f"model has {model.n_gaussians} Gaussians but evidence has {self.n_gaussians}"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, np.ndarray]) -> "Evidence":
        return cls(
            visual_detail=mapping["visual_detail"],
            mask=mapping["mask"],
            contribution=mapping["contribution"],
            visibility=mapping["visibility"],
            coverage=mapping.get("coverage"),
            projected_area=mapping.get("projected_area"),
            semantic_label=mapping.get("semantic_label"),
        )

    def as_dict(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {
            "visual_detail": self.visual_detail,
            "mask": self.mask,
            "contribution": self.contribution,
            "visibility": self.visibility,
        }
        if self.coverage is not None:
            out["coverage"] = self.coverage
        if self.projected_area is not None:
            out["projected_area"] = self.projected_area
        if self.semantic_label is not None:
            out["semantic_label"] = self.semantic_label
        return out

