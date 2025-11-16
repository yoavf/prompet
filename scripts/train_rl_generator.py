#!/usr/bin/env python3
"""GRPO training script for generator model - teaching VLM to generate cute pet SVGs."""

import json
import sys
import torch
import gc
import cairosvg
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
from PIL import Image
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from peft import get_peft_model
import wandb

# Import TRL+PEFT utilities
sys.path.append(str(Path(__file__).parent))
from model_utils import load_model_4bit
from training_utils import setup_lora_config

# Import shared training utilities
from training_core import (
    log_memory_usage,
    load_config,
    save_metadata,
    detect_version_dir,
    detect_gpu_config,
    resolve_model_path,
)


def extract_svg_from_completion(completion: str) -> str:
    """
    Extract SVG code from model completion.

    The model may respond with explanatory text and SVG in markdown blocks.
    We need to extract just the SVG code.
    """
    import re

    # Try to find SVG in markdown code block
    svg_block_match = re.search(r'```(?:svg|xml)?\s*\n?(.*?)\n?```', completion, re.DOTALL | re.IGNORECASE)
    if svg_block_match:
        return svg_block_match.group(1).strip()

    # Try to find raw SVG tags
    svg_match = re.search(r'(<svg\s+.*?</svg>)', completion, re.DOTALL | re.IGNORECASE)
    if svg_match:
        return svg_match.group(1).strip()

    # Fallback: return as-is (will likely fail rendering and get 0.0 score)
    return completion.strip()


def render_svg_to_png(svg_str, size=512):
    """
    Render SVG string to PIL Image.

    Args:
        svg_str: SVG code as string
        size: Output image size (square)

    Returns:
        PIL Image
    """
    try:
        # Extract SVG from completion if needed
        svg_code = extract_svg_from_completion(svg_str)

        png_bytes = cairosvg.svg2png(
            bytestring=svg_code.encode('utf-8'),
            output_width=size,
            output_height=size
        )
        return Image.open(BytesIO(png_bytes)).convert('RGB')
    except Exception as e:
        print(f"SVG rendering failed: {e}")
        # Return blank image on error
        return Image.new('RGB', (size, size), color='white')


def main():
    """Main training function."""
    # Force line buffering for real-time logging
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # Set PyTorch memory allocator to avoid fragmentation (recommended by PyTorch)
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    if len(sys.argv) < 2:
        print("Usage: python train_rl_generator.py <lineage/version>")
        print("Example: python train_rl_generator.py critique/v1")
        print("Example: python train_rl_generator.py generator/v2")
        sys.exit(1)

    version = sys.argv[1]
    version_dir = detect_version_dir(version)

    if not version_dir.exists():
        print(f"Error: Version directory not found: {version_dir}")
        print(f"Run 'python scripts/bump.py' to create a new version")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"GRPO Training - {version}")
    print(f"Version directory: {version_dir.absolute()}")
    print(f"{'='*60}\n")

    # Load config
    config_path = version_dir / "config.json"
    config = load_config(config_path)

    # Check for GPU
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPUs detected. This training requires GPU.")
        sys.exit(1)

    # Detect GPU configuration
    num_gpus, device_map_config, is_multi_gpu = detect_gpu_config()

    print(f"Configuration:")
    print(f"  Base model: {config['base_model']}")
    print(f"  Num generations: {config['grpo_training']['num_generations']}")
    print(f"  Max steps: {config['grpo_training']['max_steps']}")
    print(f"  Learning rate: {config['grpo_training']['learning_rate']}")
    print()

    log_memory_usage("start")

    # Resolve model path
    model_path = resolve_model_path(config['base_model'], version_dir)

    # Load generator model with 4-bit quantization
    print(f"Loading base model: {model_path}")
    model, processor = load_model_4bit(
        model_path,
        device_map="balanced" if device_map_config else "auto"
    )

    # Prepare quantized model for LoRA training
    print("\nPreparing quantized model for k-bit training...")
    from peft import prepare_model_for_kbit_training, PeftModel
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # Enable input gradients for PEFT (required for backpropagation through frozen base model)
    # Must be called BEFORE wrapping with LoRA
    print("Enabling input gradients for PEFT...")
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        # Fallback for models without this method
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Load SFT LoRA adapters if continuing from SFT checkpoint
    sft_checkpoint = config.get("sft_lora_checkpoint")
    if sft_checkpoint:
        # Resolve relative path
        if sft_checkpoint.startswith("../"):
            sft_checkpoint_path = version_dir.parent / sft_checkpoint[3:]
        else:
            sft_checkpoint_path = Path(sft_checkpoint)

        print(f"\nLoading SFT LoRA adapters from: {sft_checkpoint_path}")
        model = PeftModel.from_pretrained(model, str(sft_checkpoint_path), is_trainable=True)
        print("✓ SFT LoRA adapters loaded - continuing training")
    else:
        # Apply new LoRA adapters for RL fine-tuning from scratch
        print("\nApplying new LoRA configuration...")
        lora_config = setup_lora_config(config.get("lora", {}))
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    print("✓ LoRA applied")
    log_memory_usage("after_lora")

    # Create training_samples directory for saving actual GRPO samples
    # Ensure we're using the version directory correctly (not nested)
    training_samples_dir = version_dir / "training_samples"

    # Clean up any existing nested directory structure
    if training_samples_dir.exists() and (training_samples_dir / "training_samples").exists():
        print(f"Warning: Found nested training_samples directory, cleaning up...")
        import shutil
        # Move samples up one level
        nested_dir = training_samples_dir / "training_samples"
        for item in nested_dir.iterdir():
            shutil.move(str(item), str(training_samples_dir / item.name))
        nested_dir.rmdir()

    training_samples_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training samples directory: {training_samples_dir.absolute()}")

    # Use self-critique: training model evaluates its own outputs (zero extra memory)
    print(f"\nUsing self-critique: generator model will evaluate its own outputs")

    # Use self-critique pattern (no separate critique model - uses training model)
    from training_utils import create_reward_function_self_critique
    base_reward_func = create_reward_function_self_critique(
        training_model=model,
        processor=processor,
        render_fn=render_svg_to_png
    )

    # Wrap reward function to save actual GRPO samples (no extra generation!)
    current_step = {'value': 0}

    def reward_func_with_saving(prompts, completions, **kwargs):
        """Wrapper that saves actual GRPO samples, then calls reward function."""
        step_num = current_step['value']

        # Log memory before reward function
        if torch.cuda.is_available():
            mem_start = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n{'='*60}", flush=True)
            print(f"STEP {step_num}: Starting reward scoring", flush=True)
            print(f"[MEMORY] Allocated: {mem_start:.2f}GB, Reserved: {mem_reserved:.2f}GB", flush=True)
            print(f"{'='*60}", flush=True)

        # Call actual reward function
        scores = base_reward_func(prompts, completions, **kwargs)

        # Log memory after reward function
        if torch.cuda.is_available():
            mem_end = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[MEMORY] After reward function: {mem_end:.2f}GB (delta: {mem_end - mem_start:+.2f}GB)", flush=True)

        # Save ALL samples from this step (all generations with rewards)
        try:
            if step_num > 0:  # Skip step 0
                step_dir = training_samples_dir / f"step_{step_num:04d}"
                step_dir.mkdir(exist_ok=True)

                # Debug: Print prompts array info
                print(f"  [DEBUG] Prompts array length: {len(prompts)}, Completions: {len(completions)}, Scores: {len(scores)}", flush=True)

                # Save ALL samples (all actual GRPO generations!)
                num_to_save = len(completions)
                for i in range(num_to_save):
                    svg_code = extract_svg_from_completion(completions[i])

                    # Save SVG
                    svg_path = step_dir / f"sample_{i+1}.svg"
                    svg_path.write_text(svg_code)

                    # Get prompt - handle both cases:
                    # - prompts[i] if each completion has its own prompt
                    # - prompts[0] if all completions share one prompt (typical with batch_size=1)
                    if i < len(prompts):
                        prompt_text = prompts[i]
                    elif len(prompts) > 0:
                        prompt_text = prompts[0]  # All share the same prompt
                    else:
                        prompt_text = "N/A"

                    # Save review with prompt and score
                    review_path = step_dir / f"sample_{i+1}.txt"
                    review_path.write_text(
                        f"Step: {step_num}\n"
                        f"Sample: {i+1}/{num_to_save}\n"
                        f"Prompt: {prompt_text}\n"
                        f"Reward Score: {scores[i]:.4f}\n\n"
                        f"SVG Code (first 500 chars):\n{svg_code[:500]}...\n"
                    )

                print(f"  ✓ Saved {num_to_save} actual GRPO samples for step {step_num}")
        except Exception as e:
            print(f"  Warning: Failed to save samples: {e}")

        # Log final memory state
        if torch.cuda.is_available():
            mem_final = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved_final = torch.cuda.memory_reserved(0) / 1024**3
            print(f"[MEMORY] Step {step_num} complete - Allocated: {mem_final:.2f}GB, Reserved: {mem_reserved_final:.2f}GB", flush=True)
            print(f"{'='*60}\n", flush=True)

        # Increment step counter
        current_step['value'] += 1

        return scores

    reward_func = reward_func_with_saving

    # Load diverse prompts from dataset to prevent reward hacking
    print("\nLoading diverse prompts from dataset...")
    dataset_path = version_dir / "dataset.jsonl"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Creating fallback dataset with single prompt (WARNING: high risk of reward hacking)")
        prompt_text = config.get('prompt', 'Generate an svg illustration of a pet - output svg code only')
        prompts = [prompt_text] * 100
    else:
        # Load all prompts from dataset
        import json
        prompts = []
        with open(dataset_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                prompts.append(entry['prompt'])

        print(f"Loaded {len(prompts)} diverse prompts from dataset")

        # Use only 100 diverse prompts (same as before when it worked)
        import random
        prompts = random.sample(prompts, min(100, len(prompts)))
        print(f"Using {len(prompts)} diverse prompts for training")

    # Format all prompts in chat format
    formatted_prompts = []
    for prompt_text in prompts:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}]
        }]
        formatted_prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    # Create dataset with diverse prompts
    dataset = Dataset.from_dict({
        "prompt": formatted_prompts
    })

    print(f"✓ Dataset created with {len(dataset)} diverse prompts (prevents reward hacking)")

    # Initialize Weights & Biases
    print("\nInitializing Weights & Biases...")
    wandb.init(
        project="pet-auto-rl",
        name=f"{version}-grpo-generator",
        config={
            "base_model": config['base_model'],
            "training_mode": "grpo-generator",
            "version": version,
            **config['grpo_training']
        }
    )

    # Configure GRPO with TRL
    print("\nConfiguring GRPO trainer with TRL...")
    log_memory_usage("before_grpo_config")

    grpo_cfg = config['grpo_training']
    num_generations = grpo_cfg['num_generations']

    grpo_config = GRPOConfig(
        output_dir=str(version_dir / "models" / "checkpoints"),
        learning_rate=grpo_cfg['learning_rate'],
        per_device_train_batch_size=grpo_cfg['batch_size'],
        gradient_accumulation_steps=grpo_cfg.get('gradient_accumulation_steps', 1),
        num_generations=num_generations,
        generation_batch_size=grpo_cfg.get('generation_batch_size', num_generations),  # Generate in smaller batches to save memory
        max_steps=grpo_cfg['max_steps'],
        lr_scheduler_type=grpo_cfg.get('lr_scheduler_type', 'linear'),
        max_completion_length=grpo_cfg['max_new_tokens'],
        temperature=grpo_cfg['temperature'],
        logging_steps=1,
        save_steps=grpo_cfg.get('save_steps', 5),
        gradient_checkpointing=False,  # Disabled - incompatible with GRPO generation phase
        dataloader_num_workers=0,
        fp16=False,  # Disabled - gradient scaler conflicts with GRPO generation
        bf16=False,  # Disabled - model already mixed precision from PEFT
        optim="paged_adamw_32bit",  # Memory-efficient paged optimizer
        report_to="wandb",
        run_name=f"{version}-grpo",
        logging_first_step=True,
    )

    # Initialize GRPO trainer
    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
    )

    # Simple callback for logging rewards to wandb
    from transformers import TrainerCallback

    class RewardLoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                # Log reward metrics to wandb
                wandb_logs = {}
                for key, value in logs.items():
                    if 'reward' in key.lower() or 'kl' in key.lower():
                        wandb_logs[key] = value
                if wandb_logs:
                    wandb.log(wandb_logs)

    grpo_trainer.add_callback(RewardLoggingCallback())

    print("✓ GRPO trainer configured with TRL")
    log_memory_usage("after_grpo_config")

    # Train with GRPO
    print(f"\n{'='*60}")
    print("Starting GRPO training...")
    print(f"{'='*60}\n")

    # Always start fresh training (using sft_lora_checkpoint from config if specified)
    # Auto-resume disabled - to resume manually, use trainer's resume_from_checkpoint parameter
    print("Starting training from scratch (auto-resume disabled)\n")

    start_time = datetime.now(timezone.utc)

    try:
        # Run training from scratch
        grpo_trainer.train()

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Save final model (checkpoint only - no merge to save disk space)
        print(f"\n{'='*60}")
        print("Training complete!")

        # Note: Final checkpoint is already saved by trainer
        # Merging LoRA into base model would require ~6GB extra disk space
        # For inference, use the checkpoint directly or merge later
        checkpoint_dir = version_dir / "models" / "checkpoints"
        final_checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
        if final_checkpoints:
            print(f"✓ Final checkpoint saved: {final_checkpoints[-1].name}")
        print("  (LoRA checkpoint only - merge for inference if needed)")

        # Save metadata
        metadata_path = version_dir / "metadata.json"
        save_metadata(metadata_path, config, "completed")

        print(f"✓ Metadata saved")
        print(f"\n{'='*60}")
        print(f"GRPO training complete!")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f}m)")
        print(f"{'='*60}")

        # Finish WandB run
        wandb.finish()

        # Cleanup resources
        print("\nCleaning up resources...")
        del grpo_trainer
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print("✓ Cleanup complete")

        # Exit successfully
        sys.exit(0)

    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        print(f"\n{'='*60}")
        print(f"ERROR: Training failed!")
        print(f"{'='*60}")
        print(f"Exception: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print(f"Duration before failure: {duration:.1f}s")

        # Save error metadata
        metadata_path = version_dir / "metadata.json"
        save_metadata(metadata_path, config, "failed", str(e))

        print(f"✓ Error logged to metadata.json")
        print(f"{'='*60}\n")

        # Finish WandB run with failed status
        try:
            wandb.finish(exit_code=1)
        except Exception:
            pass

        # Cleanup resources
        print("Cleaning up resources...")
        try:
            del grpo_trainer
            del model
            del processor
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print("✓ Cleanup complete")

        # Exit with error code
        sys.exit(1)


if __name__ == "__main__":
    main()
