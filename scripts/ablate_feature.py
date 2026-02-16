#!/usr/bin/env python3
"""
Ablate a specific feature and measure impact on answer probabilities.

This script loads a model, registers an ablation hook that zeros a target
feature, and reruns prompts to measure the change in probability for the
expected correct answers.
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

from src.ablation import AblationHook
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
from src.model_loader import ModelLoader
from src.sae_adapter import create_sae_adapter

logger = logging.getLogger(__name__)


def load_prompts(prompts_file: str) -> list[dict]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    return prompts


def get_answer_probability(
    model,
    tokenizer,
    prompt: str,
    answer: str,
) -> float:
    """
    Compute probability of a specific answer continuation.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        answer: Expected answer string

    Returns:
        Log probability of the answer tokens
    """
    # Tokenize prompt and answer
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_text = prompt + " " + answer
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)
    
    # Answer tokens are the difference
    answer_ids = full_ids[len(prompt_ids):]
    
    if len(answer_ids) == 0:
        logger.warning(f"No answer tokens for: {answer}")
        return float('-inf')
    
    # Convert to tensors
    input_ids = torch.tensor([full_ids]).to(model.device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
    
    # Compute log probabilities for answer tokens
    log_probs = []
    for i, token_id in enumerate(answer_ids):
        # Position in logits is len(prompt_ids) + i - 1 (since logits are shifted)
        pos = len(prompt_ids) + i - 1
        if pos < len(logits):
            token_logits = logits[pos]
            token_log_probs = torch.log_softmax(token_logits, dim=-1)
            log_probs.append(token_log_probs[token_id].item())
    
    # Return mean log probability
    return sum(log_probs) / len(log_probs) if log_probs else float('-inf')


def run_ablation_for_prompt(
    model,
    tokenizer,
    sae_adapter,
    prompt_data: dict,
    layer_idx: int,
    feature_id: int,
) -> dict:
    """
    Run ablation for a single prompt.

    Args:
        model: Language model
        tokenizer: Tokenizer
        sae_adapter: SAE adapter
        prompt_data: Prompt data dict
        layer_idx: Layer to ablate
        feature_id: Feature to ablate

    Returns:
        Dict with baseline and ablated probabilities
    """
    prompt = prompt_data["prompt"]
    answer = prompt_data["expected_correct_answer"]
    
    # Baseline (no ablation)
    baseline_prob = get_answer_probability(model, tokenizer, prompt, answer)
    
    # Ablated
    with AblationHook(model, sae_adapter, layer_idx, feature_id):
        ablated_prob = get_answer_probability(model, tokenizer, prompt, answer)
    
    return {
        "id": prompt_data["id"],
        "pair_id": prompt_data["pair_id"],
        "label": prompt_data["label"],
        "prompt": prompt,
        "expected_answer": answer,
        "baseline_log_prob": baseline_prob,
        "ablated_log_prob": ablated_prob,
        "delta_log_prob": ablated_prob - baseline_prob,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ablate a feature and measure impact on answer probability"
    )
    add_common_args(parser)
    add_model_args(parser)
    add_output_args(parser)
    
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index to ablate",
    )
    parser.add_argument(
        "--feature_id",
        type=int,
        required=True,
        help="Feature ID to ablate",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="data/prompts_physics.jsonl",
        help="Path to prompts JSONL file",
    )
    parser.add_argument(
        "--sae_repo",
        type=str,
        default=None,
        help="HuggingFace repo for SAE weights (None for mock mode)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock SAE mode",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info(f"Ablating feature {args.feature_id} at layer {args.layer}")
    logger.info("=" * 60)
    
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
    
    # Create output directory
    output_dir = create_timestamped_dir(args.output_dir)
    report_file = output_dir / "ablation_report.json"
    
    logger.info(f"Saving report to {report_file}")
    
    # Run ablation experiments
    results = []
    
    for prompt_data in tqdm(prompts, desc="Running ablation"):
        try:
            result = run_ablation_for_prompt(
                model,
                tokenizer,
                sae_adapter,
                prompt_data,
                args.layer,
                args.feature_id,
            )
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing prompt {prompt_data['id']}: {e}")
            continue
    
    # Create report
    report = {
        "layer": args.layer,
        "feature_id": args.feature_id,
        "model_name": args.model_name,
        "sae_repo": args.sae_repo,
        "mock_mode": args.mock,
        "num_prompts": len(results),
        "results": results,
        "summary": {
            "mean_delta_log_prob": sum(r["delta_log_prob"] for r in results) / len(results) if results else 0.0,
            "num_decreased": sum(1 for r in results if r["delta_log_prob"] < 0),
            "num_increased": sum(1 for r in results if r["delta_log_prob"] > 0),
            "num_unchanged": sum(1 for r in results if r["delta_log_prob"] == 0),
        }
    }
    
    # Save report
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(safe_json_dump(report), f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Completed! Ablation report saved to {report_file}")
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"  Mean delta log prob: {report['summary']['mean_delta_log_prob']:.4f}")
    logger.info(f"  Decreased: {report['summary']['num_decreased']}")
    logger.info(f"  Increased: {report['summary']['num_increased']}")
    logger.info(f"  Unchanged: {report['summary']['num_unchanged']}")


if __name__ == "__main__":
    main()
