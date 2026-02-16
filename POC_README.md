# Physics Misconception Mechanistic Interpretability PoC

A proof-of-concept for identifying features in Gemma 2 2B that distinguish between correct physics reasoning and common misconceptions about velocity vs acceleration.

## Overview

This PoC uses Sparse Autoencoders (SAEs) to decompose model activations into interpretable features, then identifies which features are most active when the model processes physics misconceptions versus correct reasoning.

**Key capabilities:**
- Run prompts through Gemma 2 2B and capture logits
- Extract sparse features from residual stream activations using GemmaScope SAEs
- Compare features between correct and misconception prompt pairs
- Ablate specific features and measure impact on answer probabilities

## Repository Structure

```
.
├── data/
│   └── prompts_physics.jsonl      # 22 minimal-pair prompts (velocity vs acceleration)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Shared CLI args, logging, seeds
│   ├── model_loader.py            # Gemma 2 2B loading via HuggingFace
│   ├── hooks.py                   # Forward hooks for residual stream
│   ├── sae_adapter.py             # SAE loading (GemmaScope + mock fallback)
│   ├── feature_utils.py           # Sparse feature operations
│   └── ablation.py                # Feature ablation hooks
├── scripts/
│   ├── run_prompts.py             # Step 1: Run prompts, save results
│   ├── extract_features.py        # Step 2: Extract sparse features
│   ├── diff_features.py           # Step 3: Compare feature sets
│   └── ablate_feature.py          # Step 4: Ablate features, measure impact
├── runs/                          # Output directory (timestamped subdirs)
├── requirements.txt               # PoC dependencies
└── POC_README.md                  # This file
```

## Setup

### Requirements

- Python 3.11+
- CUDA-capable GPU (recommended: 16GB+ VRAM for Gemma 2 2B)
- ~8GB disk space for model weights

### Installation

1. Navigate to the repository:
```bash
cd circuit-tracer
```

2. Install PoC dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Log in to HuggingFace to access Gemma models:
```bash
huggingface-cli login
```

## Usage

### End-to-End Workflow

The PoC consists of 4 scripts that run sequentially:

#### Step 1: Run Prompts

Run all prompts through the model and save logits and decoded answers:

```bash
python scripts/run_prompts.py \
  --model_name google/gemma-2-2b \
  --max_new_tokens 10 \
  --output_dir runs \
  --seed 42
```

**Outputs:** `runs/<timestamp>/results.jsonl`

Each line contains:
- `id`, `pair_id`, `label`: Prompt metadata
- `prompt`: Input prompt text
- `tokens`: Generated token IDs
- `logits_topk`: Top-50 logits for each generation step
- `decoded_answer`: Generated text

#### Step 2: Extract Features

Hook into transformer layers and extract sparse features:

```bash
python scripts/extract_features.py \
  --model_name google/gemma-2-2b \
  --layers_to_hook 0,6,12,18,24 \
  --sae_repo google/gemma-scope-2b-pt-res \
  --topk_features 20 \
  --output_dir runs \
  --seed 42
```

**Mock mode** (when SAE weights unavailable):
```bash
python scripts/extract_features.py --mock --layers_to_hook 0,6,12
```

**Outputs:** `runs/<timestamp>/features.jsonl`

Each line contains:
- `id`, `pair_id`, `label`, `layer`: Prompt and layer metadata
- `topk_feature_ids`: Top-k activated feature indices
- `topk_feature_activations`: Corresponding activation values

#### Step 3: Compare Features

Compute top differing features between two runs (e.g., correct vs misconception):

```bash
python scripts/diff_features.py \
  --run_dir_a runs/20260215_120000 \
  --run_dir_b runs/20260215_120100 \
  --topk_features 20 \
  --output_path runs/20260215_120000/diff_report.json
```

**Outputs:** `diff_report.json`

Contains:
- `diffs_by_layer`: Per-layer top differing features
- `all_diffs`: List of all (pair_id, layer) diffs
- Per-diff: `top_diff_features`, `diff_magnitudes`, activations for both sets

#### Step 4: Ablate Feature

Zero a specific feature and measure impact on answer probability:

```bash
python scripts/ablate_feature.py \
  --model_name google/gemma-2-2b \
  --layer 12 \
  --feature_id 4567 \
  --sae_repo google/gemma-scope-2b-pt-res \
  --output_dir runs \
  --seed 42
```

**Outputs:** `runs/<timestamp>/ablation_report.json`

Contains:
- `layer`, `feature_id`: Ablation target
- `results`: Per-prompt baseline vs ablated log probabilities
- `summary`: Mean delta, count of decreased/increased/unchanged

## Mock Mode

When GemmaScope SAE weights are unavailable (e.g., no internet, model not released), the PoC automatically falls back to **mock mode**:

- Uses random orthogonal directions as features
- Useful for testing the pipeline without real SAEs
- Add `--mock` flag to force mock mode
- Logs a warning when mock mode is active

## Data Format

### Prompts (`data/prompts_physics.jsonl`)

Each line is a JSON object with:
- `id`: Unique prompt ID (e.g., `vel_accel_01a`)
- `pair_id`: Pair identifier (e.g., `vel_accel_01`)
- `label`: `"correct"` or `"misconception"`
- `prompt`: Prompt text (incomplete sentence)
- `expected_correct_answer`: The factually correct completion

**Minimal pairs:** Each `pair_id` has two variants with the same prompt but different expected answers (correct physics vs common misconception).

## Tips

### GPU Memory

- Gemma 2 2B in bfloat16 requires ~5GB VRAM
- SAE encoding adds ~2-4GB depending on layer width and batch size
- Use `--dtype float16` if bfloat16 unsupported
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if OOM errors occur

### Layers to Hook

For Gemma 2 2B (26 layers):
- Early layers: 0-5
- Middle layers: 10-15
- Late layers: 20-25

Suggested: `--layers_to_hook 0,6,12,18,24`

### Reproducibility

All scripts accept `--seed` (default: 42). Set the same seed across runs for reproducibility.

## Troubleshooting

**"Failed to load SAE weights"**
- Check internet connection
- Ensure HuggingFace token is valid (if repo requires authentication)
- Use `--mock` to bypass SAE loading

**"CUDA out of memory"**
- Reduce number of layers (`--layers_to_hook 12`)
- Use smaller batch size (currently fixed at 1)
- Try `device_map="cpu"` (very slow)

**"Import error: No module named 'src'"**
- Ensure you're running scripts from the repo root: `python scripts/run_prompts.py ...`
- The scripts add `src/` to Python path automatically

## Citation

This PoC was built for the `circuit-tracer` project. For questions, see the main repository README.

## License

MIT License (see repository root for full text)
