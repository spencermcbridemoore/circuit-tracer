#!/usr/bin/env python3
"""
Run prompts through Gemma 2 2B and save results.

This script loads prompts from data/prompts_physics.jsonl, runs them through
the model, and saves token IDs, top-k logits, and decoded answers.
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
from src.model_loader import ModelLoader

logger = logging.getLogger(__name__)


def load_prompts(prompts_file: str) -> list[dict]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
    return prompts


def run_single_prompt(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    topk_logits: int = 50,
) -> dict:
    """
    Run a single prompt and return results.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum tokens to generate
        topk_logits: Number of top logit entries to save

    Returns:
        Dict with tokens, logits, and decoded answer
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = inputs["input_ids"].to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Extract generated tokens
    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
    generated_tokens = generated_ids.tolist()
    
    # Extract logits for generated tokens
    # outputs.scores is a tuple of tensors, one per generated step
    logits_list = []
    for step_idx, step_logits in enumerate(outputs.scores):
        # step_logits: [batch_size, vocab_size]
        step_logits_cpu = step_logits[0].cpu()  # First (only) batch item
        
        # Get top-k
        topk_values, topk_indices = torch.topk(step_logits_cpu, k=topk_logits)
        
        logits_list.append({
            "step": step_idx,
            "top_tokens": topk_indices.tolist(),
            "top_logits": topk_values.tolist(),
        })
    
    # Decode generated text
    decoded_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return {
        "tokens": generated_tokens,
        "logits_topk": logits_list,
        "decoded_answer": decoded_answer,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run prompts through model and save results"
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
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum new tokens to generate per prompt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (currently only 1 supported)",
    )
    parser.add_argument(
        "--topk_logits",
        type=int,
        default=50,
        help="Number of top logit entries to save per generation step",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("Running prompts script")
    logger.info("=" * 60)
    
    # Load prompts
    prompts = load_prompts(args.prompts_file)
    
    # Load model
    torch_dtype = get_torch_dtype(args.dtype)
    loader = ModelLoader(
        model_name=args.model_name,
        dtype=torch_dtype,
        device_map="auto",
    )
    model, tokenizer = loader.load()
    
    # Create output directory
    output_dir = create_timestamped_dir(args.output_dir)
    results_file = output_dir / "results.jsonl"
    
    logger.info(f"Saving results to {results_file}")
    
    # Run prompts
    with open(results_file, "w", encoding="utf-8") as f:
        for prompt_data in tqdm(prompts, desc="Running prompts"):
            try:
                result = run_single_prompt(
                    model,
                    tokenizer,
                    prompt_data["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    topk_logits=args.topk_logits,
                )
                
                # Combine with original prompt data
                output = {
                    "id": prompt_data["id"],
                    "pair_id": prompt_data["pair_id"],
                    "label": prompt_data["label"],
                    "prompt": prompt_data["prompt"],
                    "expected_correct_answer": prompt_data["expected_correct_answer"],
                    **result,
                }
                
                # Write to JSONL
                f.write(json.dumps(safe_json_dump(output), ensure_ascii=False) + "\n")
                f.flush()
                
            except Exception as e:
                logger.error(f"Error processing prompt {prompt_data['id']}: {e}")
                continue
    
    logger.info(f"✓ Completed! Results saved to {results_file}")
    logger.info(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
