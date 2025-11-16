#!/usr/bin/env python3
"""
Utility functions for loading vision-language models with 4-bit quantization.
Supports Pixtral-12B, Qwen2-VL, and other VLMs.
"""

import torch
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


def load_vlm_4bit(model_path, device_map="auto", base_model_for_processor=None):
    """
    Load vision-language model with 4-bit quantization.

    Handles both full models and PEFT checkpoints (LoRA adapters).

    Args:
        model_path: Path to model (HF hub ID, local full model, or PEFT checkpoint)
        device_map: Device mapping strategy ("auto", "balanced", etc.)
        base_model_for_processor: Base model ID for processor (only needed for checkpoints)

    Returns:
        (model, processor) tuple

    Examples:
        # Load Pixtral-12B
        model, processor = load_vlm_4bit("mistral-community/pixtral-12b")

        # Load Qwen2-VL-8B
        model, processor = load_vlm_4bit("Qwen/Qwen2-VL-8B-Instruct")

        # Load from PEFT checkpoint (auto-detects base model)
        model, processor = load_vlm_4bit("/path/to/checkpoint-16")
    """
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # Check if this is a PEFT checkpoint (has adapter_config.json)
    model_path_obj = Path(model_path)
    adapter_config_path = model_path_obj / "adapter_config.json"
    is_peft_checkpoint = adapter_config_path.exists()

    if is_peft_checkpoint:
        print(f"  Detected PEFT checkpoint - loading base model + adapters...")

        # Load PEFT config to get base model path
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path

        print(f"  Base model: {base_model_path}")

        # Load base model with 4-bit quantization
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map=device_map
        )

        # Load PEFT adapters on top
        model = PeftModel.from_pretrained(model, model_path)
        print(f"  ✓ PEFT adapters loaded")

        # Processor should come from base model
        processor_path = base_model_for_processor if base_model_for_processor else base_model_path
    else:
        # Load full model directly
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map
        )

        # Processor from model path
        processor_path = base_model_for_processor if base_model_for_processor else model_path

    # Load processor
    processor = AutoProcessor.from_pretrained(processor_path)

    # Set pad token if not present
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


def load_vlm_8bit(model_path, device_map="auto", base_model_for_processor=None):
    """
    Load vision-language model with 8-bit quantization (better quality than 4-bit).

    Handles both full models and PEFT checkpoints (LoRA adapters).

    Args:
        model_path: Path to model (HF hub ID, local full model, or PEFT checkpoint)
        device_map: Device mapping strategy ("auto", "balanced", etc.)
        base_model_for_processor: Base model ID for processor (only needed for checkpoints)

    Returns:
        (model, processor) tuple
    """
    # Configure 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    # Check if this is a PEFT checkpoint (has adapter_config.json)
    model_path_obj = Path(model_path)
    adapter_config_path = model_path_obj / "adapter_config.json"
    is_peft_checkpoint = adapter_config_path.exists()

    if is_peft_checkpoint:
        print(f"  Detected PEFT checkpoint - loading base model + adapters...")

        # Load PEFT config to get base model path
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path

        print(f"  Base model: {base_model_path}")

        # Load base model with 8-bit quantization
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map=device_map
        )

        # Load PEFT adapters on top
        model = PeftModel.from_pretrained(model, model_path)
        print(f"  ✓ PEFT adapters loaded")

        # Processor should come from base model
        processor_path = base_model_for_processor if base_model_for_processor else base_model_path
    else:
        # Load full model directly
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map
        )

        # Processor from model path
        processor_path = base_model_for_processor if base_model_for_processor else model_path

    # Load processor
    processor = AutoProcessor.from_pretrained(processor_path)

    # Set pad token if not present
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


# Backwards compatibility alias
def load_model_4bit(model_path="mistral-community/pixtral-12b", device_map="auto"):
    """
    Legacy function name - use load_vlm_4bit instead.
    Kept for backwards compatibility.
    """
    return load_vlm_4bit(model_path, device_map)


def prepare_vision_inputs(processor, image, text):
    """
    Prepare inputs for vision-language model.

    Args:
        processor: AutoProcessor instance
        image: PIL Image
        text: Text prompt

    Returns:
        Inputs dict ready for model.generate()
    """
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    # Move to device and cast pixel_values to float16
    inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    return inputs
