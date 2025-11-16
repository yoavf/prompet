#!/usr/bin/env python3
"""Review generated SVG samples using the VLM reward model."""

import argparse
import sys
import torch
from pathlib import Path
from model_utils import load_vlm_8bit
import gc

# Import VLM reward model
sys.path.append(str(Path(__file__).parent))
from vlm_reward import Qwen3VLRewardModel


def review_samples(model_path: str, samples_dir: Path):
    """
    Review all SVG samples in directory using VLM reward model.

    Args:
        model_path: Path to trained model directory
        samples_dir: Directory containing SVG files to review
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

    # Load model with 8-bit quantization (better quality than 4-bit for inference)
    model, processor = load_vlm_8bit(model_path, device_map=device_map)

    print(f"✓ Model loaded")

    # Initialize reward model
    print(f"Initializing VLM reward model...")
    reward_model = Qwen3VLRewardModel(
        model=model,
        processor=processor,
        image_size=512,
        shared_model=False  # Not shared since we're only doing inference
    )

    # Find all SVG files
    svg_files = sorted(samples_dir.glob("*.svg"))

    if not svg_files:
        print(f"No SVG files found in {samples_dir}")
        return

    print(f"\nFound {len(svg_files)} SVG files to review")
    print(f"{'='*60}\n")

    # Review each SVG
    for i, svg_file in enumerate(svg_files):
        print(f"Reviewing {svg_file.name} ({i+1}/{len(svg_files)})...")

        # Read SVG content
        svg_content = svg_file.read_text()

        # Get detailed review
        try:
            # Render SVG to image
            import io
            from PIL import Image
            png_bytes = reward_model.render_svg_to_png(svg_content)
            image = Image.open(io.BytesIO(png_bytes)).convert('RGB')

            # Get VLM evaluation with detailed response
            review_text = get_detailed_review(reward_model, image)

            # Save review to .txt file
            review_file = svg_file.with_suffix('.txt')
            review_file.write_text(review_text)

            print(f"  ✓ Review saved to {review_file.name}")

        except Exception as e:
            # SVG rendering failed
            error_text = f"SVG RENDERING FAILED\n\nError: {e}\n\nThis SVG is structurally invalid and cannot be rendered."
            review_file = svg_file.with_suffix('.txt')
            review_file.write_text(error_text)

            print(f"  ⚠️  Invalid SVG - error saved to {review_file.name}")

        print()

    print(f"{'='*60}")
    print(f"✓ Review complete! Reviewed {len(svg_files)} files")
    print(f"{'='*60}\n")

    # Cleanup GPU memory
    reward_model.cleanup()
    gc.collect()


def get_detailed_review(reward_model, image):
    """
    Get detailed review from VLM for a single image.

    Args:
        reward_model: Qwen3VLRewardModel instance
        image: PIL Image to evaluate

    Returns:
        Detailed review text with scores
    """
    # Prepare messages in Qwen3-VL format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": reward_model.score_prompt}
        ]
    }]

    # Apply chat template
    text = reward_model.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Prepare inputs
    inputs = reward_model.processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(reward_model.model.device)

    # Generate detailed review with low temperature
    with torch.no_grad():
        outputs = reward_model.model.generate(
            **inputs,
            max_new_tokens=200,  # More tokens for detailed review
            do_sample=False
        )

    # Decode response
    response = reward_model.processor.tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    # Clean up
    del inputs
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return response.strip()


def main():
    parser = argparse.ArgumentParser(description='Review generated SVG samples')
    parser.add_argument('--model', required=True, help='Path to trained model directory')
    parser.add_argument('--samples', required=True, help='Directory containing SVG samples')

    args = parser.parse_args()

    model_path = args.model
    samples_dir = Path(args.samples)

    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        sys.exit(1)

    review_samples(model_path, samples_dir)


if __name__ == "__main__":
    main()
