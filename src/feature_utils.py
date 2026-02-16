"""Utilities for sparse feature operations and comparison."""

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_topk_features(
    feature_activations: torch.Tensor,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get top-k activated features.

    Args:
        feature_activations: Feature activation tensor [batch, seq_len, num_features]
        k: Number of top features to return

    Returns:
        Tuple of (feature_indices, feature_values) as numpy arrays
    """
    # Average over batch and sequence dimensions
    mean_activations = feature_activations.mean(dim=(0, 1))
    
    # Get top-k
    topk_values, topk_indices = torch.topk(mean_activations, k=min(k, len(mean_activations)))
    
    return topk_indices.cpu().numpy(), topk_values.cpu().numpy()


def get_topk_features_per_position(
    feature_activations: torch.Tensor,
    k: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Get top-k activated features for each sequence position.

    Args:
        feature_activations: Feature activation tensor [batch, seq_len, num_features]
        k: Number of top features per position

    Returns:
        List of (feature_indices, feature_values) tuples, one per position
    """
    batch_size, seq_len, num_features = feature_activations.shape
    
    # Average over batch dimension
    mean_per_pos = feature_activations.mean(dim=0)  # [seq_len, num_features]
    
    results = []
    for pos in range(seq_len):
        topk_values, topk_indices = torch.topk(
            mean_per_pos[pos], k=min(k, num_features)
        )
        results.append((
            topk_indices.cpu().numpy(),
            topk_values.cpu().numpy()
        ))
    
    return results


def compute_feature_diff(
    features_a: dict[int, torch.Tensor],
    features_b: dict[int, torch.Tensor],
    topk: int = 20,
) -> dict[int, dict[str, Any]]:
    """
    Compute top differing features between two feature sets.

    Args:
        features_a: Dict mapping layer_idx -> feature activations
        features_b: Dict mapping layer_idx -> feature activations
        topk: Number of top differing features to return per layer

    Returns:
        Dict mapping layer_idx -> {
            'top_diff_features': feature indices,
            'diff_magnitudes': absolute differences,
            'mean_a': mean activations in set A,
            'mean_b': mean activations in set B,
        }
    """
    results = {}
    
    # Get common layers
    common_layers = set(features_a.keys()) & set(features_b.keys())
    
    for layer_idx in sorted(common_layers):
        feat_a = features_a[layer_idx]
        feat_b = features_b[layer_idx]
        
        # Average over batch and sequence
        mean_a = feat_a.mean(dim=(0, 1))  # [num_features]
        mean_b = feat_b.mean(dim=(0, 1))  # [num_features]
        
        # Compute absolute difference
        diff = torch.abs(mean_a - mean_b)
        
        # Get top-k differing features
        topk_diffs, topk_indices = torch.topk(diff, k=min(topk, len(diff)))
        
        results[layer_idx] = {
            'top_diff_features': topk_indices.cpu().numpy().tolist(),
            'diff_magnitudes': topk_diffs.cpu().numpy().tolist(),
            'mean_a': mean_a[topk_indices].cpu().numpy().tolist(),
            'mean_b': mean_b[topk_indices].cpu().numpy().tolist(),
        }
        
        logger.info(
            f"Layer {layer_idx}: Top diff feature {topk_indices[0].item()} "
            f"with magnitude {topk_diffs[0].item():.4f}"
        )
    
    return results


def sparsify_features(
    feature_activations: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert dense feature activations to sparse format.

    Args:
        feature_activations: Dense features [batch, seq_len, num_features]
        threshold: Activation threshold for sparsity

    Returns:
        Tuple of (sparse_indices, sparse_values)
    """
    # Apply threshold
    mask = feature_activations > threshold
    
    # Get non-zero indices and values
    sparse_indices = torch.nonzero(mask, as_tuple=False)
    sparse_values = feature_activations[mask]
    
    return sparse_indices, sparse_values


def compute_activation_statistics(
    feature_activations: torch.Tensor,
) -> dict[str, float]:
    """
    Compute statistics about feature activations.

    Args:
        feature_activations: Feature tensor [batch, seq_len, num_features]

    Returns:
        Dict with statistics (mean, std, sparsity, max, etc.)
    """
    flat = feature_activations.flatten()
    
    # Sparsity: fraction of exactly zero activations
    sparsity = (flat == 0).float().mean().item()
    
    # L0 norm (average number of non-zero features)
    l0_norm = (feature_activations > 0).float().sum(dim=-1).mean().item()
    
    return {
        'mean': flat.mean().item(),
        'std': flat.std().item(),
        'max': flat.max().item(),
        'min': flat.min().item(),
        'sparsity': sparsity,
        'l0_norm': l0_norm,
    }
