#!/usr/bin/env python3
"""
Pre-download GemmaScope SAE weights for all layers.

Downloads one params.npz per layer (width_16k, L0 closest to target)
into the local HuggingFace cache so all other scripts run offline.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


def resolve_files(repo_id: str, width: str, target_l0: int) -> dict[int, str]:
    """Return {layer_idx: filename} mapping, picking L0 nearest to target_l0."""
    all_files = list(list_repo_files(repo_id))
    by_layer: dict[int, list[tuple[int, str]]] = {}

    for f in all_files:
        parts = f.split("/")
        if (
            len(parts) == 4
            and parts[0].startswith("layer_")
            and parts[1] == width
            and parts[2].startswith("average_l0_")
            and parts[3] == "params.npz"
        ):
            layer_idx = int(parts[0].split("_")[1])
            l0 = int(parts[2].split("_")[-1])
            by_layer.setdefault(layer_idx, []).append((l0, f))

    chosen = {}
    for layer_idx, options in sorted(by_layer.items()):
        best_l0, best_file = min(options, key=lambda x: abs(x[0] - target_l0))
        chosen[layer_idx] = best_file
        print(f"  layer {layer_idx:2d} -> {best_file}  (L0={best_l0})")

    return chosen


def main():
    parser = argparse.ArgumentParser(
        description="Download GemmaScope SAE weights for all layers"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="google/gemma-scope-2b-pt-res",
        help="HuggingFace repo ID",
    )
    parser.add_argument(
        "--width",
        type=str,
        default="width_16k",
        choices=["width_16k", "width_65k"],
        help="Feature width variant",
    )
    parser.add_argument(
        "--target_l0",
        type=int,
        default=40,
        help="Target L0 sparsity; nearest available is selected per layer",
    )
    args = parser.parse_args()

    print(f"Resolving files from {args.repo_id} ({args.width}, target L0~{args.target_l0})...")
    chosen = resolve_files(args.repo_id, args.width, args.target_l0)

    print(f"\nDownloading {len(chosen)} files...")
    for layer_idx, filename in tqdm(sorted(chosen.items()), desc="Downloading"):
        hf_hub_download(repo_id=args.repo_id, filename=filename)

    print(f"\nDone. {len(chosen)} SAE weight files cached.")


if __name__ == "__main__":
    main()
