from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
import yaml

from hggs.compression import (
    CompressedAsset,
    CompressionConfig,
    assign_precision_groups,
    bits_for_groups,
    estimate_size_mb,
    prune_to_budget,
    quantize_model,
)
from hggs.data import Evidence, GaussianModel
from hggs.importance import ImportanceConfig, compute_semantic_detail_score
from hggs.lod import LodConfig, assign_lod_layers


@dataclass(frozen=True)
class HggsConfig:
    importance: ImportanceConfig = ImportanceConfig()
    compression: CompressionConfig = CompressionConfig()
    lod: LodConfig = LodConfig()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HggsConfig":
        precision = data.get("compression", {}).get("precision_groups", {})
        compression_data = dict(data.get("compression", {}))
        compression_data.pop("precision_groups", None)
        compression_data.update(
            {
                "low_bits": precision.get("low_bits", CompressionConfig.low_bits),
                "medium_bits": precision.get("medium_bits", CompressionConfig.medium_bits),
                "high_bits": precision.get("high_bits", CompressionConfig.high_bits),
            }
        )
        return cls(
            importance=ImportanceConfig(**data.get("importance", {})),
            compression=CompressionConfig(**compression_data),
            lod=LodConfig(**data.get("lod", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "HggsConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)


def run_compression(
    model: GaussianModel,
    evidence: Evidence,
    target_mb: float,
    config: HggsConfig = HggsConfig(),
) -> CompressedAsset:
    evidence.require_compatible(model)
    raw_score, score = compute_semantic_detail_score(model, evidence, config.importance)
    retained = prune_to_budget(model, evidence, score, target_mb, config.compression)
    precision_group = assign_precision_groups(
        model=model,
        retained=retained,
        score=score,
        evidence=evidence,
        target_mb=target_mb,
        config=config.compression,
    )
    layer = assign_lod_layers(model, evidence, retained, score, config.lod)
    compressed_model = quantize_model(model, retained, precision_group, config.compression)
    retained_indices = np.flatnonzero(retained).astype(np.int32)
    retained_bits = bits_for_groups(precision_group[retained], config.compression)
    estimated_size_mb = estimate_size_mb(compressed_model, retained_bits)
    metadata: Dict[str, float] = {
        "target_mb": float(target_mb),
        "retention_ratio": float(np.mean(retained)),
        "mean_raw_score": float(np.mean(raw_score)),
        "mean_smoothed_score": float(np.mean(score)),
    }
    return CompressedAsset(
        model=compressed_model,
        retained_indices=retained_indices,
        raw_score=raw_score[retained],
        score=score[retained],
        precision_group=precision_group[retained],
        layer=layer[retained],
        estimated_size_mb=estimated_size_mb,
        metadata=metadata,
    )

