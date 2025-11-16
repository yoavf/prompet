#!/usr/bin/env python3
"""Generate sample SVGs from a trained model for visual inspection."""

import argparse
import torch
import json
import random
from pathlib import Path
from model_utils import load_model_4bit
import gc

def generate_samples(model_path: str, count: int, output_dir: Path, temperature: float, dataset_path: Path = None):
    """
    Generate sample SVGs from a trained model.

    Args:
        model_path: Path to trained model directory
        count: Number of samples to generate
        output_dir: Directory to save SVG files
        temperature: Sampling temperature
    """
    print(f"Loading model from: {model_path}")

    # Detect number of available GPUs for multi-GPU support
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        print(f"Detected {num_gpus} GPUs - using multi-GPU mode with device_map='balanced'")
        device_map = "balanced"
    elif num_gpus == 1:
        print(f"Detected 1 GPU - using single GPU mode")
        device_map = "auto"
    else:
        device_map = "cpu"

    # Load model with 4-bit quantization (handles both regular and LoRA models)
    model, processor = load_model_4bit(model_path, device_map=device_map)

    print(f"✓ Model loaded")
    print(f"\nGenerating {count} samples...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts from dataset if provided
    if dataset_path and dataset_path.exists():
        print(f"Loading diverse prompts from: {dataset_path}")
        prompts_text = []
        with open(dataset_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                prompts_text.append(entry['prompt'])

        # Randomly sample count prompts (with replacement if needed)
        if len(prompts_text) >= count:
            selected_prompts = random.sample(prompts_text, count)
        else:
            selected_prompts = random.choices(prompts_text, k=count)

        print(f"✓ Loaded {len(prompts_text)} prompts, using {count} random samples")
    else:
        # Fallback to single generic prompt
        print("Using default generic prompt (no dataset provided)")
        selected_prompts = ["Generate an svg illustration of a pet - output svg code only"] * count

    # Convert prompts to chat format
    formatted_prompts = []
    for prompt_text in selected_prompts:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}]
        }]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(prompt)

    # Batch generation settings
    batch_size = 4  # Generate 4 samples at a time
    num_batches = (count + batch_size - 1) // batch_size

    sample_idx = 0
    for batch_num in range(num_batches):
        batch_count = min(batch_size, count - sample_idx)
        print(f"\nGenerating batch {batch_num + 1}/{num_batches} ({batch_count} samples)...")

        # Get prompts for this batch
        batch_prompts = formatted_prompts[sample_idx:sample_idx + batch_count]
        inputs = processor(
            text=batch_prompts,
            return_tensors="pt",
            padding=True
        )

        # Move to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        # Process each generated sample in the batch
        for i in range(batch_count):
            # Decode
            generated_text = processor.tokenizer.decode(
                outputs[i][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Extract just the SVG part
            if "<svg" in generated_text and "</svg>" in generated_text:
                svg_start = generated_text.index("<svg")
                svg_end = generated_text.index("</svg>") + len("</svg>")
                generated_text = generated_text[svg_start:svg_end]
            else:
                print(f"  ⚠️  Warning: SVG tags not found in sample {sample_idx + 1}")

            # Save SVG file
            output_path = output_dir / f"sample_{sample_idx + 1}.svg"
            with open(output_path, 'w') as f:
                f.write(generated_text)

            # Save prompt info file
            info_path = output_dir / f"sample_{sample_idx + 1}_prompt.txt"
            with open(info_path, 'w') as f:
                f.write(f"Prompt: {selected_prompts[sample_idx]}\n")
                f.write(f"Temperature: {temperature}\n")

            # Get file size for feedback
            size_kb = output_path.stat().st_size / 1024
            prompt_preview = selected_prompts[sample_idx][:50] + "..." if len(selected_prompts[sample_idx]) > 50 else selected_prompts[sample_idx]
            print(f"  ✓ Sample {sample_idx + 1}: {output_path.name} ({size_kb:.1f} KB) - \"{prompt_preview}\"")

            sample_idx += 1

    print(f"\n{'='*60}")
    print(f"Generated {count} samples in {output_dir}")
    print(f"{'='*60}")

    # Cleanup GPU memory
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate sample SVGs from a trained model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory (e.g., /workspace/work/v1/models/final/)"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of samples to generate (default: 10)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for SVG files (default: samples/ in model dir)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset.jsonl with diverse prompts (optional)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default to samples/ in the version directory
        model_path = Path(args.model)
        # Navigate up from models/final or models/checkpoints/checkpoint-N
        if "checkpoints" in str(model_path):
            version_dir = model_path.parent.parent.parent
        else:
            version_dir = model_path.parent.parent
        output_dir = version_dir / "samples"

    # Determine dataset path
    dataset_path = None
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        # Auto-detect dataset.jsonl in version directory
        model_path = Path(args.model)
        if "checkpoints" in str(model_path):
            version_dir = model_path.parent.parent.parent
        else:
            version_dir = model_path.parent.parent

        auto_dataset = version_dir / "dataset.jsonl"
        if auto_dataset.exists():
            dataset_path = auto_dataset
            print(f"Auto-detected dataset: {dataset_path}")

    # Generate samples
    generate_samples(
        model_path=args.model,
        count=args.count,
        output_dir=output_dir,
        temperature=args.temperature,
        dataset_path=dataset_path
    )


if __name__ == "__main__":
    main()
