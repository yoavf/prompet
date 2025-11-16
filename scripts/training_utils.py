#!/usr/bin/env python3
"""
Utility functions for SFT and RL training with TRL.
"""

import torch
import gc
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig


def setup_lora_config(config):
    """
    Create LoRA configuration from config dict.

    Args:
        config: Dict with lora parameters (r, alpha, dropout, target_modules)

    Returns:
        LoraConfig instance
    """
    return LoraConfig(
        r=config.get("r", 16),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.0),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )


def setup_sft_trainer(model, processor, dataset, output_dir, training_config, lora_config):
    """
    Set up SFT trainer with LoRA.

    Args:
        model: Base model
        processor: Processor/tokenizer
        dataset: Training dataset
        output_dir: Where to save checkpoints
        training_config: Dict with training parameters
        lora_config: LoraConfig instance

    Returns:
        SFTTrainer instance
    """
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 2e-5),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 100),
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
    )

    return trainer


def create_reward_function_self_critique(training_model, processor, render_fn):
    """
    Create reward function that uses the training model itself as critic (zero extra memory).

    The model critiques its own outputs - "self-critique" approach.
    This eliminates the need to load a separate critique model.

    Args:
        training_model: The model being trained (already loaded)
        processor: Processor for the training model
        render_fn: Function to render SVG to PNG (svg_str -> PIL.Image)

    Returns:
        Reward function for GRPO
    """
    def reward_fn(prompts, completions, **kwargs):
        """Score generated SVGs using the training model as critic."""
        from vlm_reward import Qwen3VLRewardModel

        print(f"Scoring {len(completions)} completions with self-critique (training model)...")

        # Use the training model as critic (no loading needed - already in memory!)
        reward_model = Qwen3VLRewardModel(
            model=training_model,
            processor=processor,
            image_size=512,
            shared_model=True  # We're sharing the training model
        )

        scores = []
        for idx, svg in enumerate(completions):
            try:
                image = render_fn(svg)
                score = reward_model.score_image(image)
                scores.append(score)

                # CRITICAL: Clean up image tensors after EACH scoring (prevent accumulation)
                del image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error scoring SVG {idx+1}/{len(completions)}: {e}")
                scores.append(0.0)

        # Light cleanup (no model deletion - it's the training model!)
        del reward_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return scores

    return reward_fn


def create_reward_function_load_unload(critique_model_path, render_fn):
    """
    Create reward function that loads/unloads critique model each step.

    Uses PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent fragmentation.

    Args:
        critique_model_path: Path to trained critique model
        render_fn: Function to render SVG to PNG (svg_str -> PIL.Image)

    Returns:
        Reward function for GRPO
    """
    def reward_fn(prompts, completions, **kwargs):
        """Score generated SVGs by loading critique, scoring, then unloading."""
        from model_utils import load_vlm_8bit
        from vlm_reward import Qwen3VLRewardModel

        # Memory before loading
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated(0) / 1024**3
            print(f"\n[MEMORY] Before loading critique: {mem_before:.2f}GB", flush=True)

        # Aggressive cleanup before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Load critique model with 8-bit quantization (better quality than 4-bit)
        print(f"Loading critique model for scoring {len(completions)} completions...")
        model, processor = load_vlm_8bit(critique_model_path)

        if torch.cuda.is_available():
            mem_after_load = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[MEMORY] After loading critique: {mem_after_load:.2f}GB (+{mem_after_load - mem_before:.2f}GB)", flush=True)
        reward_model = Qwen3VLRewardModel(
            model=model,
            processor=processor,
            image_size=512,
            shared_model=False
        )

        scores = []
        for svg in completions:
            try:
                image = render_fn(svg)
                score = reward_model.score_image(image)
                scores.append(score)
            except Exception as e:
                print(f"Error scoring SVG: {e}")
                scores.append(0.0)

        if torch.cuda.is_available():
            mem_after_scoring = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[MEMORY] After scoring: {mem_after_scoring:.2f}GB", flush=True)

        # VERY aggressive cleanup - critical for preventing OOM
        del reward_model
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_after_cleanup = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[MEMORY] After unloading critique: {mem_after_cleanup:.2f}GB (-{mem_after_scoring - mem_after_cleanup:.2f}GB)\n", flush=True)

        return scores

    return reward_fn


def create_reward_function_preloaded(critique_reward_model, render_fn):
    """
    Create reward function using pre-loaded critique model.

    Args:
        critique_reward_model: Pre-loaded Qwen3VLRewardModel instance
        render_fn: Function to render SVG to PNG (svg_str -> PIL.Image)

    Returns:
        Reward function for GRPO
    """
    def reward_fn(prompts, completions, **kwargs):
        """
        Score generated SVGs using pre-loaded critique model.

        Args:
            prompts: List of prompts (not used, but required by GRPO)
            completions: List of generated text completions (SVG code)
            **kwargs: Additional arguments from GRPO (e.g., completion_ids) - ignored

        Returns:
            List of scores (0.0-1.0) for each completion
        """
        # Use pre-loaded critique model - no loading/unloading
        print(f"Scoring {len(completions)} completions with pre-loaded critique model...")

        scores = []
        for svg in completions:
            try:
                # Render SVG to PNG
                image = render_fn(svg)

                # Score with critique model
                score = critique_reward_model.score_image(image)
                scores.append(score)
            except Exception as e:
                print(f"Error scoring SVG: {e}")
                scores.append(0.0)

        # Light cleanup after scoring (no model deletion)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return scores

    return reward_fn


def setup_grpo_trainer(model, processor, reward_fn, prompt_dataset, output_dir, grpo_config):
    """
    Set up GRPO trainer.

    Args:
        model: Generator model (with LoRA applied)
        processor: Processor/tokenizer
        reward_fn: Reward function
        prompt_dataset: Dataset of prompts
        output_dir: Where to save checkpoints
        grpo_config: Dict with GRPO parameters

    Returns:
        GRPOTrainer instance
    """
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=grpo_config.get("num_epochs", 1),
        per_device_train_batch_size=grpo_config.get("batch_size", 1),
        learning_rate=grpo_config.get("learning_rate", 1e-6),
        num_generations=grpo_config.get("num_generations", 8),
        logging_steps=grpo_config.get("logging_steps", 1),
        save_steps=grpo_config.get("save_steps", 8),
        max_new_tokens=grpo_config.get("max_new_tokens", 2048),
        temperature=grpo_config.get("temperature", 0.8),
    )

    trainer = GRPOTrainer(
        model=model,
        config=training_args,
        reward_fn=reward_fn,
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
        train_dataset=prompt_dataset,
    )

    return trainer
