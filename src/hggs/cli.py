from __future__ import annotations

import argparse

from hggs.io import load_evidence, load_gaussians, save_asset, write_json
from hggs.pipeline import HggsConfig, run_compression


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HG-GS compression.")
    parser.add_argument("--gaussians", required=True, help="Path to gaussians.npz")
    parser.add_argument("--evidence", required=True, help="Path to evidence.npz")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config")
    parser.add_argument("--target-mb", type=float, required=True, help="Target asset size in MB")
    parser.add_argument("--out", required=True, help="Output asset directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = load_gaussians(args.gaussians)
    evidence = load_evidence(args.evidence)
    config = HggsConfig.from_yaml(args.config)
    asset = run_compression(model, evidence, args.target_mb, config)
    save_asset(asset, args.out)
    write_json(asset.summary(), f"{args.out}/summary.json")
    print(asset.summary())


if __name__ == "__main__":
    main()

