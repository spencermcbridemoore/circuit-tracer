"""Model loading utilities for Gemma 2 2B via HuggingFace Transformers."""

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages Gemma 2 2B model and tokenizer."""

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        """
        Initialize model loader.

        Args:
            model_name: HuggingFace model identifier
            dtype: Model data type
            device_map: Device placement strategy for accelerate
        """
        self.model_name = model_name
        self.dtype = dtype
        self.device_map = device_map
        self.model = None
        self.tokenizer = None

    def load(self) -> tuple[Any, Any]:
        """
        Load model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model {self.model_name} with dtype {self.dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with accelerate device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        
        self.model.eval()  # Set to evaluation mode
        
        logger.info(f"Model loaded successfully on device(s): {self.device_map}")
        logger.info(f"Model has {self.model.config.num_hidden_layers} layers")
        
        return self.model, self.tokenizer

    @property
    def num_layers(self) -> int:
        """Get number of transformer layers."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        """Get hidden dimension size."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.config.hidden_size

    def get_layer_module(self, layer_idx: int) -> Any:
        """
        Get a specific transformer layer module.

        Args:
            layer_idx: Layer index (0-indexed)

        Returns:
            The layer module
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range [0, {self.num_layers})"
            )
        
        # For Gemma 2, layers are at model.model.layers
        return self.model.model.layers[layer_idx]
