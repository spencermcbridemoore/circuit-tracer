#!/usr/bin/env python3
"""
Compute differences between two feature extraction runs.

This script loads features from two runs (e.g., correct vs misconception),
groups them by pair_id, and computes the top differing features per layer.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import safe_json_dump, setup_logging

logger = logging.getLogger(__name__)


def load_features(features_file: Path) -> dict:
    """
    Load features from JSONL file.

    Returns:
        Dict mapping (pair_id, label, layer) -> feature data
    """
    features = {}
    
    with open(features_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            key = (data["pair_id"], data["label"], data["layer"])
            features[key] = {
                "id": data["id"],
                "prompt": data["prompt"],
                "feature_ids": data["topk_feature_ids"],
                "feature_activations": data["topk_feature_activations"],
            }
    
    logger.info(f"Loaded {len(features)} feature records from {features_file}")
    return features


def compute_diff_for_pair(
    features_a: dict,
    features_b: dict,
    pair_id: str,
    layer: int,
    topk: int,
) -> dict | None:
    """
    Compute feature differences for a specific pair and layer.

    Args:
        features_a: Features from run A
        features_b: Features from run B
        pair_id: Pair ID
        layer: Layer index
        topk: Number of top differing features to return

    Returns:
        Dict with diff data or None if data missing
    """
    # Get feature data for correct and misconception variants
    key_correct = (pair_id, "correct", layer)
    key_misconception = (pair_id, "misconception", layer)
    
    if key_correct not in features_a or key_misconception not in features_b:
        return None
    
    feat_correct = features_a[key_correct]
    feat_misconception = features_b[key_misconception]
    
    # Build activation dictionaries
    act_correct = {
        fid: val
        for fid, val in zip(
            feat_correct["feature_ids"],
            feat_correct["feature_activations"]
        )
    }
    act_misconception = {
        fid: val
        for fid, val in zip(
            feat_misconception["feature_ids"],
            feat_misconception["feature_activations"]
        )
    }
    
    # Get all features from both sets
    all_features = set(act_correct.keys()) | set(act_misconception.keys())
    
    # Compute differences
    diffs = {}
    for fid in all_features:
        val_correct = act_correct.get(fid, 0.0)
        val_misconception = act_misconception.get(fid, 0.0)
        diffs[fid] = abs(val_correct - val_misconception)
    
    # Sort by difference magnitude
    sorted_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
    
    # Take top-k
    top_diffs = sorted_diffs[:topk]
    
    return {
        "pair_id": pair_id,
        "layer": layer,
        "top_diff_features": [fid for fid, _ in top_diffs],
        "diff_magnitudes": [diff for _, diff in top_diffs],
        "correct_activations": [act_correct.get(fid, 0.0) for fid, _ in top_diffs],
        "misconception_activations": [
            act_misconception.get(fid, 0.0) for fid, _ in top_diffs
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute feature differences between two runs"
    )
    parser.add_argument(
        "--run_dir_a",
        type=str,
        required=True,
        help="Path to first run directory (e.g., correct prompts)",
    )
    parser.add_argument(
        "--run_dir_b",
        type=str,
        required=True,
        help="Path to second run directory (e.g., misconception prompts)",
    )
    parser.add_argument(
        "--topk_features",
        type=int,
        default=20,
        help="Number of top differing features to report per layer",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for diff report JSON (default: run_dir_a/diff_report.json)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("Computing feature differences")
    logger.info("=" * 60)
    
    # Load features from both runs
    run_dir_a = Path(args.run_dir_a)
    run_dir_b = Path(args.run_dir_b)
    
    features_a = load_features(run_dir_a / "features.jsonl")
    features_b = load_features(run_dir_b / "features.jsonl")
    
    # Get all pair IDs and layers
    pairs_a = {(pid, layer) for pid, label, layer in features_a.keys()}
    pairs_b = {(pid, layer) for pid, label, layer in features_b.keys()}
    common_pairs = pairs_a & pairs_b
    
    logger.info(f"Found {len(common_pairs)} common (pair_id, layer) combinations")
    
    # Compute diffs
    all_diffs = []
    
    for pair_id, layer in sorted(common_pairs):
        diff = compute_diff_for_pair(
            features_a,
            features_b,
            pair_id,
            layer,
            args.topk_features,
        )
        
        if diff is not None:
            all_diffs.append(diff)
    
    # Group by layer for summary
    by_layer = defaultdict(list)
    for diff in all_diffs:
        by_layer[diff["layer"]].append(diff)
    
    # Create report
    report = {
        "run_a": str(run_dir_a),
        "run_b": str(run_dir_b),
        "topk_features": args.topk_features,
        "num_pairs": len(set(d["pair_id"] for d in all_diffs)),
        "layers": sorted(by_layer.keys()),
        "diffs_by_layer": {
            layer: diffs for layer, diffs in sorted(by_layer.items())
        },
        "all_diffs": all_diffs,
    }
    
    # Determine output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = run_dir_a / "diff_report.json"
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(safe_json_dump(report), f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Completed! Diff report saved to {output_path}")
    
    # Print summary
    logger.info("\nSummary by layer:")
    for layer in sorted(by_layer.keys()):
        layer_diffs = by_layer[layer]
        avg_top_diff = np.mean([d["diff_magnitudes"][0] for d in layer_diffs])
        logger.info(f"  Layer {layer}: {len(layer_diffs)} pairs, avg top diff = {avg_top_diff:.4f}")


if __name__ == "__main__":
    main()
