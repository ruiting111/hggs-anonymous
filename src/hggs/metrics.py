from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage


def psnr(reference: np.ndarray, rendered: np.ndarray, max_value: float = 255.0) -> float:
    reference = np.asarray(reference, dtype=np.float32)
    rendered = np.asarray(rendered, dtype=np.float32)
    mse = float(np.mean((reference - rendered) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((max_value * max_value) / mse))


def masked_psnr(
    reference: np.ndarray,
    rendered: np.ndarray,
    mask: np.ndarray,
    max_value: float = 255.0,
) -> float:
    reference = np.asarray(reference, dtype=np.float32)
    rendered = np.asarray(rendered, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == reference.ndim - 1:
        mask = mask[..., None]
    if not np.any(mask):
        return float("nan")
    sq = (reference - rendered) ** 2
    mse = float(np.sum(sq * mask) / (np.sum(mask) * reference.shape[-1] + 1e-8))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((max_value * max_value) / mse))


def edge_f1(
    reference_luma: np.ndarray,
    rendered_luma: np.ndarray,
    mask: Optional[np.ndarray] = None,
    tolerance: int = 2,
) -> float:
    ref_edges = _simple_edges(reference_luma)
    ren_edges = _simple_edges(rendered_luma)
    if mask is not None:
        mask_bool = np.asarray(mask, dtype=bool)
        ref_edges &= mask_bool
        ren_edges &= mask_bool
    ref_dilated = ndimage.binary_dilation(ref_edges, iterations=tolerance)
    ren_dilated = ndimage.binary_dilation(ren_edges, iterations=tolerance)
    tp = float(np.sum(ren_edges & ref_dilated))
    fp = float(np.sum(ren_edges & ~ref_dilated))
    fn = float(np.sum(ref_edges & ~ren_dilated))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2.0 * precision * recall / (precision + recall + 1e-8))


def runtime_summary(frame_times_ms: np.ndarray) -> Dict[str, float]:
    frame_times_ms = np.asarray(frame_times_ms, dtype=np.float32)
    fps = 1000.0 / np.maximum(frame_times_ms, 1e-6)
    return {
        "mean_fps": float(np.mean(fps)),
        "p05_fps": float(np.percentile(fps, 5)),
        "p95_frame_ms": float(np.percentile(frame_times_ms, 95)),
        "spike_rate_33ms": float(np.mean(frame_times_ms > 33.3)),
    }


def _simple_edges(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    gx = ndimage.sobel(image, axis=1)
    gy = ndimage.sobel(image, axis=0)
    magnitude = np.hypot(gx, gy)
    threshold = float(np.percentile(magnitude, 85))
    return magnitude >= threshold

