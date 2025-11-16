#!/usr/bin/env python3
"""VLM-based reward model for evaluating generated pet SVGs using Qwen3-VL."""

import io
import re
import torch
from typing import List
from PIL import Image

try:
    import cairosvg
except ImportError:
    raise ImportError(
        "cairosvg is required for SVG rendering. "
        "Install with: pip install cairosvg"
    )


class Qwen3VLRewardModel:
    """
    Reward model using Qwen3-VL to score generated pet SVG quality.

    Evaluates based on:
    - Clear identification of pet type
    - Correct anatomy and body part placement
    - Large expressive eyes and friendly expression
    - Rounded, appealing shapes
    """

    def __init__(self, model, processor, image_size=512, shared_model=False):
        """
        Initialize VLM reward model.

        Args:
            model: Qwen3VL model instance (frozen or shared with trainer)
            processor: Qwen3VL processor for formatting inputs
            image_size: Size to render SVG images (width and height)
            shared_model: If True, model is shared with trainer (enable gradient disabling)
        """
        self.model = model
        self.processor = processor
        self.image_size = image_size
        self.shared_model = shared_model

        # Scoring prompt - matches critique model training format
        self.score_prompt = """We had a program attempt to generate a vector illustration of a pet. Critique this generation on these 4 criteria:

Species clarity: Can you identify what pet this is?
0/10 = completely unrecognizable
2/10 = not an animal at all
5/10 = unclear which animal
7/10 = likely identifiable as a specific pet
10/10 = immediately obvious which pet it is

Anatomy: Are body parts correct for this animal?
0/10 = no identifiable parts
3/10 = wrong/missing major parts (name the parts)
5/10 = 1 missing or badly placed part
7/10 = mostly correct with issues
10/10 = anatomically sound

Cuteness: Expression and appeal
0/10 = unidentifiable
3/10 = ugly or scary
5/10 = no expression at all
7/10 = has cute features
10/10 = very cute and charming

Technical: How well is the illustration rendered?
Focus on what you can SEE: overall coherence, color choices, visible shape problems
0/10 = severely broken (major parts invisible, completely incoherent)
3/10 = obvious visual problems (jarring colors, shapes disconnected/overlapping badly, missing fills)
5/10 = noticeable visual issues (some awkward colors, minor shape problems, uneven appearance)
7/10 = visually clean (colors work together, shapes look intentional, cohesive appearance)
10/10 = polished look (harmonious colors, all shapes appear complete and deliberate)

IMPORTANT: Focus on obvious visual problems you can clearly see, not subtle technical details.

Respond with exactly 4 lines in this format:
Species: X/10 - [explanation]
Anatomy: X/10 - [explanation]
Cuteness: X/10 - [explanation]
Technical: X/10 - [explanation]"""

        print("✓ Qwen3VLRewardModel initialized")

    def evaluate_batch(self, svg_contents: List[str], save_dir=None, step=None) -> List[float]:
        """
        Evaluate batch of complete SVG documents and return scores.

        Args:
            svg_contents: List of complete SVG document strings
            save_dir: Optional directory to save SVGs and reviews during training
            step: Optional training step number for naming files

        Returns:
            List of scores (0.0 to 1.0)
        """
        scores = []
        valid_count = 0
        invalid_count = 0
        total = len(svg_contents)

        print(f"  Evaluating {total} SVG generations...")

        # Create save directory if specified
        if save_dir:
            from pathlib import Path
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        for i, svg_content in enumerate(svg_contents):
            # Debug: Show last 200 chars of generated content (to see if complete)
            preview = svg_content[-200:].replace('\n', ' ')
            print(f"    Sample {i+1}/{total} preview (end): ...{preview}")

            # Generate filename if saving
            if save_dir:
                from pathlib import Path
                if step is not None:
                    filename = f"step{step}_sample{i+1}"
                else:
                    filename = f"sample{i+1}"
                svg_file = Path(save_dir) / f"{filename}.svg"
                review_file = Path(save_dir) / f"{filename}.txt"

            # Two-stage reward: structural validity + aesthetics
            review_text = None
            try:
                # Stage 1: Check if SVG is structurally valid (can render)
                png_bytes = self.render_svg_to_png(svg_content)
                image = Image.open(io.BytesIO(png_bytes)).convert('RGB')

                # SVG is valid! Now evaluate aesthetics
                aesthetic_score = self.score_image(image)

                # Stage 2: Map aesthetic score to 0.5-1.0 range
                # Valid SVGs get 0.5 base + (0.5 * aesthetic_quality)
                final_score = 0.5 + (0.5 * aesthetic_score)

                print(f"    Sample {i+1}/{total}: VALID (aesthetic={aesthetic_score:.2f}, final={final_score:.2f})")
                scores.append(final_score)
                valid_count += 1

                # Prepare review text
                if save_dir:
                    review_text = f"VALID SVG\n\nAesthetic Score: {aesthetic_score:.2f}\nFinal Score: {final_score:.2f}\n\n"
                    review_text += f"This SVG successfully rendered and was evaluated by the VLM reward model.\n"

            except Exception as e:
                # Stage 1 failed: SVG is broken/invalid
                # Always give 0.0 for structural failures
                print(f"    Sample {i+1}/{total}: INVALID ({e})")
                print(f"      Score: 0.0")
                scores.append(0.0)
                invalid_count += 1

                # Prepare review text for invalid SVG
                if save_dir:
                    review_text = f"INVALID SVG\n\nError: {e}\n\nScore: 0.0\n\n"
                    review_text += f"This SVG is structurally invalid and could not be rendered.\n"

            # Save SVG and review if save_dir is specified
            if save_dir:
                # Save SVG
                svg_file.write_text(svg_content)
                # Save review
                review_file.write_text(review_text)

        # Print summary statistics
        print(f"\n  {'='*60}")
        print(f"  Reward Summary:")
        print(f"  {'='*60}")
        print(f"  Total samples: {total}")
        print(f"  Valid SVGs: {valid_count} ({100*valid_count/total:.1f}%)")
        print(f"  Invalid SVGs: {invalid_count} ({100*invalid_count/total:.1f}%)")

        if valid_count > 0:
            import numpy as np
            valid_scores = [s for s in scores if s > 0.0]
            print(f"\n  Valid SVG Scores:")
            print(f"    Mean: {np.mean(valid_scores):.3f}")
            print(f"    Std:  {np.std(valid_scores):.3f}")
            print(f"    Min:  {np.min(valid_scores):.3f}")
            print(f"    Max:  {np.max(valid_scores):.3f}")

        print(f"\n  Overall Score Statistics:")
        import numpy as np
        print(f"    Mean (all): {np.mean(scores):.3f}")
        print(f"    Std (all):  {np.std(scores):.3f}")
        print(f"  {'='*60}\n")

        return scores

    def score_image(self, image: Image.Image) -> float:
        """
        Score a single image with Qwen3-VL.

        Args:
            image: PIL Image to evaluate

        Returns:
            Score between 0.0 and 1.0
        """
        import gc

        # If using shared model, switch to evaluation mode and disable gradients
        if self.shared_model:
            was_training = self.model.training
            self.model.train(mode=False)  # Switch to evaluation mode

        try:
            # Prepare messages in Qwen3-VL format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.score_prompt}
                ]
            }]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

            # Generate score with low temperature for consistency
            # Note: do_sample=False means temperature is ignored, avoiding warnings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,  # Increased for detailed scoring format
                    do_sample=False
                )

            # Decode response
            response = self.processor.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Aggressively clear memory
            del inputs
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Parse float from response
            score = self.parse_score(response)
            return score

        finally:
            # If using shared model, restore training mode
            if self.shared_model and was_training:
                self.model.train(mode=True)

    def parse_score(self, response: str) -> float:
        """
        Extract float score from model response.

        Args:
            response: Raw model output text (expects "X/10 - explanation" format for 4 criteria)

        Returns:
            Parsed score averaged from all criteria and normalized to [0, 1], or 0.0 if unparseable
        """
        # Parse X/10 scores from each criterion line
        # Expected format:
        # Species: 8/10 - explanation
        # Anatomy: 7/10 - explanation
        # Cuteness: 9/10 - explanation
        # Technical: 6/10 - explanation

        scores = []

        # Try to find all X/10 patterns
        matches = re.findall(r'(\d+)/10', response)

        if matches:
            for match in matches:
                try:
                    score_out_of_10 = float(match)
                    # Normalize to 0.0-1.0
                    normalized = score_out_of_10 / 10.0
                    scores.append(normalized)
                except ValueError:
                    continue

        if scores:
            # Average all scores
            avg_score = sum(scores) / len(scores)
            # Clamp to valid range
            return max(0.0, min(1.0, avg_score))

        print(f"    Warning: Could not parse X/10 scores from: '{response}' - defaulting to 0.0")
        return 0.0

    def render_svg_to_png(self, svg_content: str) -> bytes:
        """
        Render SVG to PNG using CairoSVG.

        Args:
            svg_content: Complete SVG document as string

        Returns:
            PNG image as bytes
        """
        return cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            output_width=self.image_size,
            output_height=self.image_size
        )

    def cleanup(self):
        """Clean up GPU memory."""
        import gc
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def test_reward_model():
    """Test the VLM reward model with a sample SVG."""
    from model_utils import load_model_4bit

    # Sample cute cat SVG
    sample_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <!-- Cat face -->
  <circle cx="200" cy="200" r="100" fill="#ff9933"/>
  <!-- Eyes -->
  <circle cx="170" cy="180" r="20" fill="#000"/>
  <circle cx="230" cy="180" r="20" fill="#000"/>
  <circle cx="175" cy="175" r="8" fill="#fff"/>
  <circle cx="235" cy="175" r="8" fill="#fff"/>
  <!-- Nose -->
  <circle cx="200" cy="210" r="8" fill="#ff6666"/>
  <!-- Mouth -->
  <path d="M 200 210 Q 180 230 170 220" stroke="#000" stroke-width="2" fill="none"/>
  <path d="M 200 210 Q 220 230 230 220" stroke="#000" stroke-width="2" fill="none"/>
  <!-- Ears -->
  <path d="M 150 150 L 130 100 L 170 140 Z" fill="#ff9933"/>
  <path d="M 250 150 L 270 100 L 230 140 Z" fill="#ff9933"/>
</svg>'''

    print("Loading Pixtral-12B model for testing...")
    model, processor = load_model_4bit()

    print("Initializing reward model...")
    reward_model = Qwen3VLRewardModel(model, processor)

    print("\nTesting with sample cute cat SVG...")
    scores = reward_model.evaluate_batch([sample_svg])
    print(f"\nResult: {scores[0]:.2f}")


if __name__ == "__main__":
    test_reward_model()
