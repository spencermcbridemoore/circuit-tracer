"""Feature ablation utilities via forward hooks."""

import logging
from typing import Any

import torch
import torch.nn as nn

from src.sae_adapter import SAEAdapter

logger = logging.getLogger(__name__)


class AblationHook:
    """
    Hook for ablating specific features during forward pass.
    
    This hook intercepts the residual stream at a layer, encodes to features,
    zeros the target feature, then decodes back to the residual stream.
    """

    def __init__(
        self,
        model: Any,
        sae_adapter: SAEAdapter,
        layer_idx: int,
        feature_id: int,
    ):
        """
        Initialize ablation hook.

        Args:
            model: The transformer model
            sae_adapter: SAE adapter for encoding/decoding
            layer_idx: Target layer index
            feature_id: Target feature ID to ablate
        """
        self.model = model
        self.sae_adapter = sae_adapter
        self.layer_idx = layer_idx
        self.feature_id = feature_id
        self.hook_handle = None
        
        logger.info(
            f"Initialized ablation for layer {layer_idx}, feature {feature_id}"
        )

    def _ablation_hook(
        self, module: nn.Module, input: tuple[torch.Tensor], output: torch.Tensor
    ) -> torch.Tensor:
        """
        Hook function that ablates the target feature.

        Args:
            module: The module being hooked (MLP)
            input: Input tuple (residual stream before MLP)
            output: Output from MLP

        Returns:
            Modified output with feature ablated
        """
        # Get the residual stream (input to MLP)
        residual = input[0]
        
        # Encode to features
        features = self.sae_adapter.encode(residual, self.layer_idx)
        
        # Zero out the target feature
        features[..., self.feature_id] = 0.0
        
        # Decode back to residual space
        reconstructed = self.sae_adapter.decode(features, self.layer_idx)
        
        # Compute the intervention: replace MLP output with reconstructed version
        # The original computation was: residual -> MLP -> output
        # We want: residual (with feature ablated) -> MLP
        # But we've already intercepted at MLP input, so we need to adjust output
        
        # Actually, we need to modify the residual before it goes through MLP
        # This is a bit tricky with output hooks. Let's use a different approach:
        # We'll compute what the MLP *would* have output with ablated input
        
        # For simplicity, we'll just compute the delta and apply it to output
        delta = reconstructed - residual
        
        # The MLP operates on the ablated residual, so we need to recompute
        # However, we can't easily do that here. Instead, we'll modify the
        # residual stream directly by hooking the input side.
        
        # Actually, let's use a pre-hook instead (see register_hook below)
        return output

    def _input_ablation_hook(
        self, module: nn.Module, input: tuple[torch.Tensor]
    ) -> tuple[torch.Tensor]:
        """
        Pre-hook that ablates features in the input to the MLP.

        Args:
            module: The module (MLP)
            input: Input tuple

        Returns:
            Modified input tuple with feature ablated
        """
        residual = input[0]
        
        # Encode to features
        features = self.sae_adapter.encode(residual, self.layer_idx)
        
        # Zero out target feature
        features[..., self.feature_id] = 0.0
        
        # Decode back
        ablated_residual = self.sae_adapter.decode(features, self.layer_idx)
        
        # Return modified input
        return (ablated_residual,) + input[1:]

    def register(self) -> None:
        """Register the ablation hook on the target layer."""
        # Get the MLP module for the target layer
        layer = self.model.model.layers[self.layer_idx]
        mlp = layer.mlp
        
        # Register pre-hook to modify input
        self.hook_handle = mlp.register_forward_pre_hook(self._input_ablation_hook)
        
        logger.debug(f"Registered ablation hook on layer {self.layer_idx}")

    def remove(self) -> None:
        """Remove the ablation hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.debug("Removed ablation hook")

    def __enter__(self):
        """Context manager entry."""
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove()
        return False


class MultiFeatureAblationHook:
    """Hook for ablating multiple features simultaneously."""

    def __init__(
        self,
        model: Any,
        sae_adapter: SAEAdapter,
        ablations: list[tuple[int, int]],  # List of (layer_idx, feature_id)
    ):
        """
        Initialize multi-feature ablation.

        Args:
            model: The transformer model
            sae_adapter: SAE adapter
            ablations: List of (layer_idx, feature_id) tuples to ablate
        """
        self.model = model
        self.sae_adapter = sae_adapter
        self.ablations = ablations
        self.hooks = []
        
        # Group ablations by layer
        self.ablations_by_layer = {}
        for layer_idx, feature_id in ablations:
            if layer_idx not in self.ablations_by_layer:
                self.ablations_by_layer[layer_idx] = []
            self.ablations_by_layer[layer_idx].append(feature_id)
        
        logger.info(f"Initialized multi-feature ablation for {len(ablations)} features")

    def _make_hook(self, layer_idx: int):
        """Create ablation hook for a specific layer."""
        feature_ids = self.ablations_by_layer[layer_idx]
        
        def hook_fn(module: nn.Module, input: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
            residual = input[0]
            
            # Encode
            features = self.sae_adapter.encode(residual, layer_idx)
            
            # Zero out all target features
            for feature_id in feature_ids:
                features[..., feature_id] = 0.0
            
            # Decode
            ablated_residual = self.sae_adapter.decode(features, layer_idx)
            
            return (ablated_residual,) + input[1:]
        
        return hook_fn

    def register(self) -> None:
        """Register all ablation hooks."""
        for layer_idx in self.ablations_by_layer.keys():
            layer = self.model.model.layers[layer_idx]
            mlp = layer.mlp
            
            hook = mlp.register_forward_pre_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)
        
        logger.debug(f"Registered {len(self.hooks)} ablation hooks")

    def remove(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        """Context manager entry."""
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove()
        return False
