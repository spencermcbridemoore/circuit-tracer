"""SAE (Sparse Autoencoder) adapter with GemmaScope and mock implementations."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files

logger = logging.getLogger(__name__)


class SAEAdapter(ABC):
    """Abstract base class for SAE adapters."""

    @abstractmethod
    def encode(self, activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Encode activations to sparse features.

        Args:
            activations: Input activations [batch, seq_len, hidden_dim]
            layer_idx: Layer index

        Returns:
            Sparse feature activations [batch, seq_len, num_features]
        """
        pass

    @abstractmethod
    def decode(self, features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Decode sparse features back to activations.

        Args:
            features: Sparse feature activations [batch, seq_len, num_features]
            layer_idx: Layer index

        Returns:
            Reconstructed activations [batch, seq_len, hidden_dim]
        """
        pass

    @abstractmethod
    def get_num_features(self, layer_idx: int) -> int:
        """Get number of features for a layer."""
        pass


class GemmaScopeAdapter(SAEAdapter):
    """
    Adapter for GemmaScope SAE weights from HuggingFace.

    Actual repo structure:
        layer_{idx}/width_16k/average_l0_{l0}/params.npz
    Keys in each npz:
        W_enc: [hidden_dim, num_features]
        W_dec: [num_features, hidden_dim]
        b_enc: [num_features]
        b_dec: [hidden_dim]
        threshold: [num_features]  (JumpReLU threshold)
    """

    def __init__(
        self,
        repo_id: str = "google/gemma-scope-2b-pt-res",
        width: str = "width_16k",
        target_l0: int = 40,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize GemmaScope adapter.

        Args:
            repo_id: HuggingFace repo ID
            width: Feature width variant, e.g. "width_16k" or "width_65k"
            target_l0: Target average L0 sparsity; nearest available is chosen per layer
            device: Device to load weights on
            dtype: Data type for weights
        """
        self.repo_id = repo_id
        self.width = width
        self.target_l0 = target_l0
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.encoders = {}    # layer_idx -> W_enc tensor
        self.decoders = {}    # layer_idx -> W_dec tensor
        self.biases = {}      # layer_idx -> b_enc tensor
        self.thresholds = {}  # layer_idx -> threshold tensor

        # Build a map of available files once
        self._file_map: dict[int, str] = {}
        logger.info(f"Initialized GemmaScopeAdapter for {repo_id} ({width}, target L0~{target_l0})")

    def _resolve_filename(self, layer_idx: int) -> str:
        """
        Find the params.npz path for a layer, picking the L0 closest to target_l0.

        Args:
            layer_idx: Layer index

        Returns:
            HuggingFace filename string (e.g. 'layer_12/width_16k/average_l0_41/params.npz')
        """
        if layer_idx in self._file_map:
            return self._file_map[layer_idx]

        prefix = f"layer_{layer_idx}/{self.width}/"
        all_files = list(list_repo_files(self.repo_id))
        candidates = [
            f for f in all_files
            if f.startswith(prefix) and f.endswith("/params.npz")
        ]

        if not candidates:
            raise RuntimeError(
                f"No {self.width} SAE files found for layer {layer_idx} in {self.repo_id}"
            )

        # Pick the one whose l0 is closest to target_l0
        def _l0(path: str) -> int:
            # path format: layer_X/width_Yk/average_l0_ZZZ/params.npz
            return int(path.split("/")[2].split("_")[-1])

        best = min(candidates, key=lambda p: abs(_l0(p) - self.target_l0))
        logger.info(f"Layer {layer_idx}: selected {best} (L0={_l0(best)})")
        self._file_map[layer_idx] = best
        return best

    def load_layer(self, layer_idx: int) -> None:
        """
        Load SAE weights for a specific layer from the HuggingFace cache.

        Args:
            layer_idx: Layer index
        """
        if layer_idx in self.encoders:
            logger.debug(f"Layer {layer_idx} already loaded")
            return

        try:
            filename = self._resolve_filename(layer_idx)
            logger.info(f"Loading {filename} from {self.repo_id}")

            weights_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
            )

            # Load npz (numpy format)
            data = np.load(weights_path)

            def _to_tensor(key: str) -> torch.Tensor:
                return torch.from_numpy(data[key]).to(device=self.device, dtype=self.dtype)

            self.encoders[layer_idx] = _to_tensor("W_enc")   # [hidden_dim, num_features]
            self.decoders[layer_idx] = _to_tensor("W_dec")   # [num_features, hidden_dim]
            self.biases[layer_idx] = _to_tensor("b_enc")     # [num_features]
            self.thresholds[layer_idx] = _to_tensor("threshold")  # [num_features]

            logger.info(
                f"Loaded layer {layer_idx}: "
                f"{self.encoders[layer_idx].shape[0]}D -> "
                f"{self.encoders[layer_idx].shape[1]} features"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load SAE weights for layer {layer_idx}: {e}"
            ) from e

    def encode(self, activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Encode activations using JumpReLU SAE.

        Formula: ReLU(x @ W_enc + b_enc - threshold)

        Args:
            activations: [batch, seq_len, hidden_dim]
            layer_idx: Layer index

        Returns:
            Sparse features [batch, seq_len, num_features]
        """
        self.load_layer(layer_idx)
        
        # x @ W_enc + b_enc
        pre_activation = torch.matmul(
            activations, self.encoders[layer_idx]
        ) + self.biases[layer_idx]
        
        # JumpReLU: ReLU(pre_activation - threshold)
        features = torch.nn.functional.relu(
            pre_activation - self.thresholds[layer_idx]
        )
        
        return features

    def decode(self, features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Decode sparse features back to activation space.

        Formula: features @ W_dec

        Args:
            features: [batch, seq_len, num_features]
            layer_idx: Layer index

        Returns:
            Reconstructed activations [batch, seq_len, hidden_dim]
        """
        self.load_layer(layer_idx)
        return torch.matmul(features, self.decoders[layer_idx])

    def get_num_features(self, layer_idx: int) -> int:
        """Get number of features for a layer."""
        self.load_layer(layer_idx)
        return self.encoders[layer_idx].shape[1]


class MockSAEAdapter(SAEAdapter):
    """Mock SAE adapter using random orthogonal directions."""

    def __init__(
        self,
        hidden_dim: int,
        num_features: int = 16384,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        """
        Initialize mock adapter with random orthogonal features.

        Args:
            hidden_dim: Model hidden dimension
            num_features: Number of mock features per layer
            device: Device
            dtype: Data type
            seed: Random seed
        """
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.seed = seed
        
        # Generate random orthogonal encoder/decoder
        torch.manual_seed(seed)
        
        # Create random matrix and orthogonalize via QR decomposition
        # W_enc: [hidden_dim, num_features]
        random_matrix = torch.randn(
            hidden_dim, num_features, device=self.device, dtype=self.dtype
        )
        q, _ = torch.linalg.qr(random_matrix)
        self.W_enc = q
        
        # For mock, decoder is just transpose (orthogonal)
        self.W_dec = self.W_enc.T
        
        logger.warning(
            f"Using MockSAEAdapter with {num_features} random orthogonal features. "
            "This is for testing only - not real SAE features!"
        )

    def encode(self, activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Encode using random projection + ReLU.

        Args:
            activations: [batch, seq_len, hidden_dim]
            layer_idx: Layer index (ignored in mock)

        Returns:
            Mock features [batch, seq_len, num_features]
        """
        # Simple projection + ReLU
        features = torch.matmul(activations, self.W_enc)
        features = torch.nn.functional.relu(features)
        return features

    def decode(self, features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Decode using transpose projection.

        Args:
            features: [batch, seq_len, num_features]
            layer_idx: Layer index (ignored in mock)

        Returns:
            Reconstructed activations [batch, seq_len, hidden_dim]
        """
        return torch.matmul(features, self.W_dec)

    def get_num_features(self, layer_idx: int) -> int:
        """Get number of features."""
        return self.num_features


def create_sae_adapter(
    sae_repo: str | None = None,
    hidden_dim: int = 2304,  # Gemma 2 2B hidden dim
    num_mock_features: int = 16384,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    force_mock: bool = False,
    seed: int = 42,
    width: str = "width_16k",
    target_l0: int = 40,
) -> SAEAdapter:
    """
    Create an SAE adapter, falling back to mock if loading fails.

    Args:
        sae_repo: HuggingFace repo ID for SAE weights (None to force mock)
        hidden_dim: Model hidden dimension
        num_mock_features: Number of features for mock adapter
        device: Device
        dtype: Data type
        force_mock: Force mock mode even if repo provided
        seed: Random seed for mock
        width: GemmaScope width variant (e.g. "width_16k", "width_65k")
        target_l0: Target L0 sparsity; nearest available is selected per layer

    Returns:
        SAEAdapter instance
    """
    if force_mock or sae_repo is None:
        logger.warning("Using mock SAE adapter")
        return MockSAEAdapter(
            hidden_dim=hidden_dim,
            num_features=num_mock_features,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    try:
        adapter = GemmaScopeAdapter(
            repo_id=sae_repo,
            width=width,
            target_l0=target_l0,
            device=device,
            dtype=dtype,
        )
        # Try loading layer 0 to validate connectivity and weight format
        adapter.load_layer(0)
        logger.info(f"Successfully loaded GemmaScope adapter from {sae_repo}")
        return adapter

    except Exception as e:
        logger.warning(
            f"Failed to load SAE from {sae_repo}: {e}. Falling back to mock adapter."
        )
        return MockSAEAdapter(
            hidden_dim=hidden_dim,
            num_features=num_mock_features,
            device=device,
            dtype=dtype,
            seed=seed,
        )
