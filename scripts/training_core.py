"""
Shared training utilities for RL and SFT training.

This module contains common functions for configuration loading, metadata saving,
version detection, and GPU configuration.
"""

import json
import torch
from pathlib import Path
from datetime import datetime, timezone


def log_memory_usage(step=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
        gpu_max = torch.cuda.max_memory_allocated(0) / 1024**3
        gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"[MEMORY {step}] GPU 0: {gpu_mem:.2f}GB (max: {gpu_max:.2f}GB, reserved: {gpu_reserved:.2f}GB)", flush=True)


def load_config(config_path: Path) -> dict:
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_metadata(metadata_path: Path, config: dict, status: str, error: str = None):
    """Save or update training metadata."""
    metadata = {
        'version': config['version'],
        'base_model': config['base_model'],
        'training_mode': config.get('training_mode', 'rl'),
        'trained_at': datetime.now(timezone.utc).isoformat() + 'Z',
        'status': status
    }

    if error:
        metadata['error'] = error

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def detect_version_dir(version: str) -> Path:
    """
    Detect version directory based on environment (RunPod vs local).

    Args:
        version: Version identifier (e.g., "v1", "critique/v1", or "generator/v1")

    Returns:
        Path to version directory
    """
    script_dir = Path(__file__).parent

    # Detect if running on RunPod or locally
    if script_dir == Path("/workspace/work/scripts") or script_dir == Path("/workspace/scripts"):
        # RunPod: /workspace/work/critique/v1/ or /workspace/critique/v1/
        if Path("/workspace/work").exists():
            version_dir = Path("/workspace/work") / version
        else:
            version_dir = Path("/workspace") / version
    else:
        # Local: pet-auto-rl/versions/critique/v1/
        version_dir = script_dir.parent / "versions" / version

    return version_dir


def detect_gpu_config():
    """
    Detect GPU configuration and return appropriate device_map settings.

    Returns:
        Tuple of (num_gpus, device_map_config, is_multi_gpu)
    """
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus > 1:
        device_map_config = "balanced"
        is_multi_gpu = True
        print(f"GPU Setup: Detected {num_gpus} GPUs")
        print(f"  Multi-GPU mode will be enabled with device_map='balanced'")
    elif num_gpus == 1:
        device_map_config = None
        is_multi_gpu = False
        print(f"GPU Setup: Single GPU mode")
    else:
        device_map_config = None
        is_multi_gpu = False
        print("WARNING: No GPUs detected")

    print()
    return num_gpus, device_map_config, is_multi_gpu


def resolve_model_path(base_model: str, version_dir: Path) -> str:
    """
    Resolve model path from config base_model field.

    Args:
        base_model: Model identifier from config (e.g., "../v1/models/final" or "mistral-community/...")
        version_dir: Current version directory

    Returns:
        Resolved model path (absolute path for local models, HF ID for hub models)
    """
    if base_model.startswith('../'):
        # Relative path from version directory
        model_path = (version_dir / base_model).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        return str(model_path)
    else:
        # HuggingFace Hub model ID - return as-is
        return base_model


