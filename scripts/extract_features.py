#!/usr/bin/env python3
"""
Extract sparse features from model activations.

This script hooks into transformer layers, captures residual stream activations,
encodes them using SAE, and saves the sparse features to disk.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    add_common_args,
    add_model_args,
    add_output_args,
    create_timestamped_dir,
    get_torch_dtype,
    safe_json_dump,
    set_seed,
    setup_logging,
)
from src.hooks import ResidualHookManager
from src.model_loader import ModelLoader
from src.sae_adapter import create_sae_adapter
from src.feature_utils import get_topk_features

logger = logging.getLogger(__name__)


def load_prompts(prompts_file: str) -> list[dict]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    return prompts


def extract_features_for_prompt(
    model,
    tokenizer,
    hook_manager: ResidualHookManager,
    sae_adapter,
    prompt_text: str,
    layers_to_hook: list[int],
    topk_features: int,
) -> dict:
    """
    Extract features for a single prompt.

    Args:
        model: The language model
        tokenizer: Tokenizer
        hook_manager: Hook manager
        sae_adapter: SAE adapter
        prompt_text: Prompt text
        layers_to_hook: Layers to extract from
        topk_features: Number of top features to record per layer

    Returns:
        Dict with per-layer feature data
    """
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=False)
    input_ids = inputs["input_ids"].to(model.device)
    
    # Clear previous activations
    hook_manager.clear_activations()
    
    # Forward pass with hooks
    with torch.no_grad():
        _ = model(input_ids)
    
    # Extract and encode features for each layer
    layer_features = {}
    
    for layer_idx in layers_to_hook:
        # Get activations for this layer
        activations = hook_manager.get_activations(layer_idx)
        
        if activations is None:
            logger.warning(f"No activations captured for layer {layer_idx}")
            continue
        
        # Encode to sparse features
        features = sae_adapter.encode(activations, layer_idx)
        
        # Get top-k features
        topk_indices, topk_values = get_topk_features(features, k=topk_features)
        
        layer_features[layer_idx] = {
            "topk_feature_ids": topk_indices.tolist(),
            "topk_feature_activations": topk_values.tolist(),
        }
    
    return layer_features


def main():
    parser = argparse.ArgumentParser(
        description="Extract sparse features from model activations"
    )
    add_common_args(parser)
    add_model_args(parser)
    add_output_args(parser)
    
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="data/prompts_physics.jsonl",
        help="Path to prompts JSONL file",
    )
    parser.add_argument(
        "--layers_to_hook",
        type=str,
        default="0,6,12,18,24",
        help="Comma-separated layer indices to hook (e.g., '0,6,12')",
    )
    parser.add_argument(
        "--sae_repo",
        type=str,
        default=None,
        help="HuggingFace repo for SAE weights (None for mock mode)",
    )
    parser.add_argument(
        "--topk_features",
        type=int,
        default=20,
        help="Number of top features to record per layer",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock SAE mode (random features)",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("Extracting features script")
    logger.info("=" * 60)
    
    # Parse layers
    layers_to_hook = [int(x.strip()) for x in args.layers_to_hook.split(",")]
    logger.info(f"Hooking layers: {layers_to_hook}")
    
    # Load prompts
    prompts = load_prompts(args.prompts_file)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Load model
    torch_dtype = get_torch_dtype(args.dtype)
    loader = ModelLoader(
        model_name=args.model_name,
        dtype=torch_dtype,
        device_map="auto",
    )
    model, tokenizer = loader.load()
    
    # Create SAE adapter
    sae_adapter = create_sae_adapter(
        sae_repo=args.sae_repo,
        hidden_dim=loader.hidden_size,
        num_mock_features=16384,
        device=model.device if hasattr(model, 'device') else 'cuda',
        dtype=torch_dtype,
        force_mock=args.mock,
        seed=args.seed,
    )
    
    # Create hook manager
    hook_manager = ResidualHookManager(model, layers_to_hook)
    
    # Create output directory
    output_dir = create_timestamped_dir(args.output_dir)
    features_file = output_dir / "features.jsonl"
    
    logger.info(f"Saving features to {features_file}")
    
    # Extract features
    with hook_manager:  # Context manager registers/removes hooks
        with open(features_file, "w", encoding="utf-8") as f:
            for prompt_data in tqdm(prompts, desc="Extracting features"):
                try:
                    layer_features = extract_features_for_prompt(
                        model,
                        tokenizer,
                        hook_manager,
                        sae_adapter,
                        prompt_data["prompt"],
                        layers_to_hook,
                        args.topk_features,
                    )
                    
                    # Write per-layer features
                    for layer_idx, features in layer_features.items():
                        output = {
                            "id": prompt_data["id"],
                            "pair_id": prompt_data["pair_id"],
                            "label": prompt_data["label"],
                            "prompt": prompt_data["prompt"],
                            "layer": layer_idx,
                            **features,
                        }
                        
                        f.write(json.dumps(safe_json_dump(output), ensure_ascii=False) + "\n")
                    
                    f.flush()
                    
                except Exception as e:
                    logger.error(f"Error processing prompt {prompt_data['id']}: {e}")
                    continue
    
    logger.info(f"✓ Completed! Features saved to {features_file}")
    logger.info(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
