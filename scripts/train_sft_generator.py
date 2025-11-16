#!/usr/bin/env python3
"""SFT training script for generator model - teaching VLM to generate cute pet SVGs."""

import json
import sys
import torch
import gc
from pathlib import Path
from datetime import datetime, timezone
from trl import SFTConfig, SFTTrainer
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


def load_sft_dataset(dataset_path: Path, validation_split: float = 0.2) -> tuple[Dataset, Dataset]:
    """
    Load JSONL dataset for SFT training with train/validation split.

    Args:
        dataset_path: Path to dataset.jsonl file
        validation_split: Fraction of data to use for validation (default 0.2 = 20%)

    Returns:
        Tuple of (train_dataset, eval_dataset) with prompt/completion pairs
    """
    print(f"Loading dataset from: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load JSONL - use "completion" field name for TRL compatibility
    data = {"prompt": [], "completion": []}
    invalid_count = 0

    with open(dataset_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)

                # Validate required fields
                if "prompt" not in entry or "svg" not in entry:
                    print(f"Warning: Line {line_num} missing required fields, skipping")
                    invalid_count += 1
                    continue

                if not entry["svg"].strip():
                    print(f"Warning: Line {line_num} has empty SVG, skipping")
                    invalid_count += 1
                    continue

                # Store with "completion" field name for TRL compatibility
                data["prompt"].append(entry["prompt"])
                data["completion"].append(entry["svg"])

            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} invalid JSON: {e}, skipping")
                invalid_count += 1

    total_entries = len(data["prompt"])

    if total_entries == 0:
        raise ValueError("Dataset is empty or all entries are invalid")

    if invalid_count > 0 and invalid_count / (total_entries + invalid_count) > 0.9:
        raise ValueError(f"Too many invalid entries: {invalid_count}/{total_entries + invalid_count}")

    print(f"✓ Loaded {total_entries} training examples")
    if invalid_count > 0:
        print(f"  Skipped {invalid_count} invalid entries")

    full_dataset = Dataset.from_dict(data)

    # Split into train/validation
    if validation_split > 0 and len(full_dataset) > 1:
        split = full_dataset.train_test_split(test_size=validation_split, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
        print(f"  Split: {len(train_dataset)} train, {len(eval_dataset)} validation")
        return train_dataset, eval_dataset
    else:
        print(f"  No validation split (set validation_split=0)")
        return full_dataset, None


def main():
    """Main SFT training function."""
    # Force line buffering for real-time logging
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # Set PyTorch memory allocator to avoid fragmentation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    if len(sys.argv) < 2:
        print("Usage: python train_sft_generator.py <lineage/version>")
        print("Example: python train_sft_generator.py generator/v1")
        sys.exit(1)

    version = sys.argv[1]
    version_dir = detect_version_dir(version)

    if not version_dir.exists():
        print(f"Error: Version directory not found: {version_dir}")
        print(f"Run 'python scripts/bump.py' to create a new version")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"SFT Training - {version}")
    print(f"{'='*60}\n")

    # Load config
    config_path = version_dir / "config.json"
    config = load_config(config_path)

    # Validate training mode
    if config.get('training_mode') != 'sft':
        print(f"ERROR: Config training_mode is '{config.get('training_mode')}', expected 'sft'")
        sys.exit(1)

    # Check for GPU
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPUs detected. This training requires GPU.")
        sys.exit(1)

    # Detect GPU configuration
    num_gpus, device_map_config, is_multi_gpu = detect_gpu_config()

    print(f"Configuration:")
    print(f"  Base model: {config['base_model']}")
    print(f"  Training mode: SFT")
    print(f"  Num epochs: {config['sft_training']['num_epochs']}")
    print(f"  Learning rate: {config['sft_training']['learning_rate']}")
    print()

    log_memory_usage("start")

    # Resolve model path
    model_path = resolve_model_path(config['base_model'], version_dir)

    # Load dataset with train/validation split
    dataset_path = version_dir / "dataset.jsonl"
    validation_split = config.get('validation_split', 0.2)  # 20% validation by default
    train_dataset, eval_dataset = load_sft_dataset(dataset_path, validation_split=validation_split)

    # Load model with 4-bit quantization
    print(f"Loading model from: {model_path}")
    model, processor = load_model_4bit(
        model_path,
        device_map="balanced" if device_map_config else "auto"
    )

    # Apply LoRA adapters
    print("\nApplying LoRA configuration...")
    lora_config = setup_lora_config(config.get("lora", {}))
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("✓ LoRA applied")
    log_memory_usage("after_lora")

    # Initialize Weights & Biases
    print("\nInitializing Weights & Biases...")
    wandb.init(
        project="pet-auto-rl",
        name=f"{version}-sft-generator",
        config={
            "base_model": config['base_model'],
            "training_mode": "sft-generator",
            "version": version,
            **config['sft_training']
        }
    )

    # Configure SFT training
    print("\nConfiguring SFT trainer with TRL...")
    log_memory_usage("before_sft_config")

    sft_cfg = config['sft_training']

    sft_config = SFTConfig(
        output_dir=str(version_dir / "models" / "checkpoints"),
        num_train_epochs=sft_cfg['num_epochs'],
        per_device_train_batch_size=sft_cfg['batch_size'],
        gradient_accumulation_steps=sft_cfg.get('gradient_accumulation_steps', 1),
        learning_rate=sft_cfg['learning_rate'],
        lr_scheduler_type=sft_cfg.get('lr_scheduler_type', 'cosine'),
        warmup_steps=sft_cfg.get('warmup_steps', 10),
        max_length=None,  # Disable truncation for vision models to preserve image tokens
        logging_steps=10,  # Reduce logging frequency to save memory
        save_steps=sft_cfg.get('save_steps', 100),
        save_total_limit=2,  # Keep fewer checkpoints to save disk/memory
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        fp16=True if torch.cuda.is_available() else False,
        optim="adamw_8bit",
        report_to="wandb",
        run_name=f"{version}-sft-generator",
        logging_first_step=True,
        dataset_text_field="",  # Required for vision models (empty since we use messages format)
        packing=False,  # No sequence packing for vision models
        # Keep unused columns for vision data
        remove_unused_columns=False,
        # Evaluation configuration
        eval_strategy="epoch" if eval_dataset else "no",  # Eval only at end of epoch to save memory
        per_device_eval_batch_size=1,  # Always use batch size 1 for eval to save memory
        eval_accumulation_steps=8,  # Accumulate more steps during eval to reduce memory
        load_best_model_at_end=False,  # Disable to save memory (can't load best model during training)
        save_only_model=True,  # Don't save optimizer states to save disk space
        # Memory optimization: disable logging history accumulation
        logging_nan_inf_filter=False,  # Skip NaN/Inf checks to save memory
        skip_memory_metrics=True,  # Don't log memory metrics (saves history tracking)
        # Critical: Don't accumulate predictions during eval (saves huge memory for vision models)
        prediction_loss_only=True,  # Only compute loss, don't store predictions/labels
    )

    # Initialize SFT trainer
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,  # Use tokenizer for text-only training (no images)
    )

    print("✓ SFT trainer configured with TRL")
    log_memory_usage("after_sft_config")

    # Train with SFT
    print(f"\n{'='*60}")
    print("Starting SFT training...")
    print(f"{'='*60}\n")

    start_time = datetime.now(timezone.utc)

    try:
        # Run training
        sft_trainer.train()

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Save final model
        print(f"\n{'='*60}")
        print("Training complete! Saving model...")
        final_model_dir = version_dir / "models" / "final"
        final_model_dir.mkdir(parents=True, exist_ok=True)

        # Merge LoRA adapters and save
        print("Merging LoRA adapters into base model...")
        model = model.merge_and_unload()
        model.save_pretrained(str(final_model_dir))
        processor.save_pretrained(str(final_model_dir))

        print(f"✓ Model saved to {final_model_dir}")

        # Save metadata
        metadata_path = version_dir / "metadata.json"
        save_metadata(metadata_path, config, "completed")
        print(f"✓ Metadata saved")

        print(f"\n{'='*60}")
        print(f"SFT training complete!")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f}m)")
        print(f"{'='*60}")

        # Finish WandB run
        wandb.finish()

        # Cleanup resources
        print("\nCleaning up resources...")
        del sft_trainer
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print("✓ Cleanup complete")

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
            del sft_trainer
            del model
            del processor
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print("✓ Cleanup complete")

        sys.exit(1)


if __name__ == "__main__":
    main()
