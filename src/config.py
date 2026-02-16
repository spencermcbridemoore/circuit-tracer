"""Shared configuration utilities for the PoC scripts."""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Additional determinism settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the appropriate compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common CLI arguments to a parser."""
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model-related CLI arguments."""
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b",
        help="HuggingFace model name (default: google/gemma-2-2b)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)",
    )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """Add output directory CLI arguments."""
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs",
        help="Base output directory (default: runs)",
    )


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def create_timestamped_dir(base_dir: str | Path) -> Path:
    """Create a timestamped output directory."""
    from datetime import datetime
    
    base_path = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_path / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def safe_json_dump(obj: Any) -> Any:
    """Convert tensors and numpy arrays to JSON-serializable types."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json_dump(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_dump(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj
