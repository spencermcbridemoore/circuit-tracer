"""Forward hook utilities for capturing residual stream activations."""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ResidualHookManager:
    """Manages forward hooks on transformer layer residual streams."""

    def __init__(self, model: Any, layers_to_hook: list[int]):
        """
        Initialize hook manager.

        Args:
            model: The transformer model (e.g., Gemma 2)
            layers_to_hook: List of layer indices to hook
        """
        self.model = model
        self.layers_to_hook = sorted(layers_to_hook)
        self.hooks = []
        self.activations = {}
        
        logger.info(f"Initializing hooks on layers: {self.layers_to_hook}")

    def _make_hook(self, layer_idx: int):
        """
        Create a hook function for a specific layer.

        The hook captures the input to the MLP (i.e., post-attention residual).
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Hook function
        """
        def hook_fn(module: nn.Module, input: tuple[torch.Tensor], output: torch.Tensor):
            """
            Hook function that captures MLP input (post-attention residual).
            
            For Gemma 2, the layer structure is:
            - input -> self_attn -> + residual -> [MLP INPUT] -> mlp -> + residual -> output
            
            We want to capture the input to the MLP, which is the post-attention residual.
            The MLP module receives this as its input.
            """
            # input[0] is the first positional argument to the MLP forward
            # This is the post-attention residual stream
            activation = input[0].detach()
            
            # Store with layer index
            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            self.activations[layer_idx].append(activation)
            
        return hook_fn

    def register_hooks(self) -> None:
        """Register forward hooks on specified layers."""
        self.activations.clear()
        
        for layer_idx in self.layers_to_hook:
            # Get the MLP module for this layer
            layer = self.model.model.layers[layer_idx]
            mlp_module = layer.mlp
            
            # Register hook on MLP to capture its input (post-attention residual)
            hook = mlp_module.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)
            
        logger.info(f"Registered {len(self.hooks)} hooks")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.debug("Removed all hooks")

    def get_activations(self, layer_idx: int) -> torch.Tensor | None:
        """
        Get activations for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Stacked activations tensor or None if no activations captured
        """
        if layer_idx not in self.activations:
            return None
        
        # Stack all activations for this layer
        # activations[layer_idx] is a list of tensors from multiple forward passes
        return torch.cat(self.activations[layer_idx], dim=0)

    def clear_activations(self) -> None:
        """Clear all stored activations."""
        self.activations.clear()

    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()
        return False
