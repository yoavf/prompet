# prompet

Training vision-language models to generate cute pet vector art using supervised fine-tuning (SFT) and reinforcement learning (GRPO).

## Results

- **Final model**: https://huggingface.co/yoavf/prompet-cute-pet
- **Training dataset**: https://huggingface.co/datasets/yoavf/svg-animal-illustrations

## How It Works

Two-stage pipeline:

1. **SFT (Supervised Fine-Tuning)**: Train base model on prompt→SVG pairs
2. **GRPO (Reinforcement Learning)**: Refine outputs using a learned critique model as reward

Two independent model lineages:
- **Critique** (`critique/v1`): Scores image quality
- **Generator** (`generator/v1` → `v2`): Creates SVG illustrations

## Hardware Requirements

- **SFT training**: 1x RTX 3090 (24GB VRAM)
- **RL training**: 2x RTX 3090 or 1x 48GB card

Training uses 4-bit quantization + LoRA adapters for memory efficiency.

## Quick Start

### Setup

```bash
git clone https://github.com/yoavf/prompet
cd prompet

# Install dependencies
pip install torch transformers trl peft bitsandbytes accelerate datasets pillow wandb
```

### Training on RunPod

This project uses RunPod for GPU training. [Sign up here](https://runpod.io?ref=YOUR_REF_CODE) (referral link - you get $5 credit, I get a small commission).

1. **Launch a RunPod instance**:
   - Template: PyTorch 2.x
   - GPU: RTX 3090 (24GB) for SFT, 2x RTX 3090 or A6000 (48GB) for RL
   - Enable SSH access

2. **Train critique model** (SFT):
```bash
python scripts/deploy_sft_critique_runpod.py --version critique/v1
```

3. **Train generator** (SFT):
```bash
python scripts/deploy_sft_generator_runpod.py --version generator/v1
```

4. **Train generator** (RL):
```bash
python scripts/deploy_rl_generator_runpod.py --version generator/v2
```

The deploy scripts handle:
- SSH connection setup
- File uploads
- Dependency installation
- Training execution with tmux monitoring
- Result downloads (models, samples, logs)

### Local Generation

```bash
python scripts/generate_samples.py \
  --model versions/generator/v2/models/checkpoints/checkpoint-40 \
  --count 10 \
  --output samples/
```

## Project Structure

```
prompet/
├── scripts/
│   ├── train_*.py              # Training scripts
│   ├── deploy_*_runpod.py      # RunPod deployment automation
│   ├── generate_samples.py     # Generate SVGs from trained models
│   ├── review_samples.py       # Score SVGs with critique model
│   └── runpod_utils.py         # Shared deployment utilities
└── versions/
    ├── critique/v1/            # Critique model (scores images)
    └── generator/
        ├── v1/                 # SFT baseline
        └── v2/                 # RL refinement
```

## Configuration

Training parameters are in `versions/{lineage}/v{N}/config.json`:

**SFT** (`sft_training`):
- `num_epochs`: Training epochs
- `learning_rate`: 2e-5 typical
- `batch_size`: 1 (with gradient accumulation)
- `max_seq_length`: 2048 for SVG generation

**RL** (`grpo_training`):
- `num_generations`: SVGs generated per step (4-8)
- `max_steps`: Training iterations
- `learning_rate`: 1e-6 (lower than SFT)
- `temperature`: Generation diversity (0.7-1.0)

## Notes

- Models are saved as LoRA checkpoints (~100MB) not full merged models (~6GB)
- RL training requires a trained critique model (auto-detected from `critique/v1/models/final`)
- Dataset preparation: Download from HuggingFace, place as `dataset.jsonl` in version directory
- WandB tracking: Set `WANDB_API_KEY` environment variable (optional)

## License

MIT
