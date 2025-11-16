#!/usr/bin/env python3
"""Create a new version directory for training with model lineage support."""

import json
from pathlib import Path


def get_latest_version(lineage_dir: Path) -> tuple:
    """
    Find the latest version number within a lineage directory.

    Args:
        lineage_dir: Path to lineage directory (e.g., versions/critique/ or versions/generator/)

    Returns:
        Tuple of (version_name, version_number) or (None, 0) if no versions exist
    """
    version_dirs = sorted([d for d in lineage_dir.glob("v*") if d.is_dir()])

    if not version_dirs:
        return None, 0

    latest = version_dirs[-1]
    version_num = int(latest.name[1:])  # Strip 'v' prefix

    return latest.name, version_num


def create_version(version_name: str, base_model: str, training_mode: str, lineage: str, lineage_dir: Path):
    """
    Create a new version directory with config.

    Args:
        version_name: Version directory name (e.g., 'v1')
        base_model: Base model path or HuggingFace ID
        training_mode: Training mode ('rl' or 'sft')
        lineage: Model lineage ('critique' or 'generator')
        lineage_dir: Path to lineage directory (e.g., versions/critique/)
    """
    version_dir = lineage_dir / version_name

    # Create directory structure
    version_dir.mkdir(exist_ok=False)
    (version_dir / "models" / "checkpoints").mkdir(parents=True)

    print(f"✓ Created directory: {version_dir}")

    # Create config.json based on training mode
    config = {
        "base_model": base_model,
        "version": version_name,
        "model_lineage": lineage,
        "training_mode": training_mode,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    }

    if training_mode == "sft":
        config["sft_training"] = {
            "num_epochs": 3,
            "learning_rate": 2e-5,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 2048,
            "save_steps": 100,
            "warmup_steps": 10,
            "lr_scheduler_type": "cosine"
        }
    else:  # rl mode
        config["grpo_training"] = {
            "num_generations": 8,
            "max_steps": 16,
            "learning_rate": 1e-6,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_new_tokens": 2048,
            "temperature": 1.1,
            "save_steps": 8,
            "image_size": 512,
            "lr_scheduler_type": "constant"
        }
        config["scoring"] = {
            "temperature": 0.2
        }
        config["prompt"] = "Generate an svg illustration of a pet - output svg code only"

        # For generator RL training, critique model will be auto-detected
        # (finds latest trained critique model in versions/critique/)
        if lineage == "generator":
            config["critique_model_path"] = "auto"

    config_path = version_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Created config: {config_path}")

    # Create placeholder metadata
    metadata = {
        "version": version_name,
        "model_lineage": lineage,
        "base_model": base_model,
        "training_mode": training_mode,
        "status": "pending",
        "notes": f"{lineage}/{version_name} - {training_mode.upper()} training not yet started"
    }

    metadata_path = version_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Created metadata: {metadata_path}")

    # For SFT mode, create empty dataset.jsonl with instructions
    if training_mode == "sft":
        dataset_path = version_dir / "dataset.jsonl"
        with open(dataset_path, 'w') as f:
            f.write('# Add your training data here in JSONL format\n')
            f.write('# Each line should be: {"prompt": "...", "svg": "..."}\n')
            f.write('# Example:\n')
            f.write('# {"prompt": "A cute corgi puppy with big eyes", "svg": "<svg>...</svg>"}\n')

        print(f"✓ Created dataset template: {dataset_path}")


def main():
    """Main function to create next version with lineage support."""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    versions_dir = project_root / "versions"

    # Ensure versions directory and lineage subdirectories exist
    versions_dir.mkdir(exist_ok=True)
    (versions_dir / "critique").mkdir(exist_ok=True)
    (versions_dir / "generator").mkdir(exist_ok=True)

    print(f"{'='*60}")
    print("Create New Training Version")
    print(f"{'='*60}\n")

    # Step 1: Select model lineage
    print("Select model lineage:")
    print("  1. Critique (reward model for scoring SVGs)")
    print("  2. Generator (model that creates SVGs)")
    lineage_choice = input("Enter choice [1/2]: ").strip()

    if lineage_choice == '1':
        lineage = 'critique'
        lineage_dir = versions_dir / "critique"
    else:
        lineage = 'generator'
        lineage_dir = versions_dir / "generator"

    print(f"Selected lineage: {lineage}")
    print()

    # Step 2: Find latest version in this lineage
    latest_name, latest_num = get_latest_version(lineage_dir)

    if latest_name is None:
        # First version in this lineage
        new_version_name = "v1"
        new_version_num = 1

        # Ask user to select base model
        print("\nSelect base model:")
        print("  1. Pixtral-12B (mistral-community/pixtral-12b)")
        print("  2. Qwen2-VL-8B (Qwen/Qwen2-VL-8B-Instruct)")
        print("  3. Custom (enter HuggingFace model ID)")
        model_choice = input("Enter choice [1/2/3]: ").strip()

        if model_choice == '2':
            base_model = "Qwen/Qwen2-VL-8B-Instruct"
        elif model_choice == '3':
            base_model = input("Enter HuggingFace model ID: ").strip()
        else:
            base_model = "mistral-community/pixtral-12b"

        print(f"Creating first version in {lineage} lineage: {new_version_name}")
        print(f"Base model: {base_model}")
    else:
        # Next version in this lineage
        new_version_num = latest_num + 1
        new_version_name = f"v{new_version_num}"

        # Check if previous version has trained model
        prev_model_dir = lineage_dir / latest_name / "models" / "final"

        if not prev_model_dir.exists():
            print(f"⚠️  Warning: Previous version ({lineage}/{latest_name}) has no trained model")
            print(f"   You should train {lineage}/{latest_name} before creating {lineage}/{new_version_name}")
            print()
            response = input(f"Create {lineage}/{new_version_name} anyway using base model? [y/N]: ")

            if response.lower() != 'y':
                print("Cancelled.")
                return

            # Ask user to select base model
            print("\nSelect base model:")
            print("  1. Pixtral-12B (mistral-community/pixtral-12b)")
            print("  2. Qwen2-VL-8B (Qwen/Qwen2-VL-8B-Instruct)")
            print("  3. Custom (enter HuggingFace model ID)")
            model_choice = input("Enter choice [1/2/3]: ").strip()

            if model_choice == '2':
                base_model = "Qwen/Qwen2-VL-8B-Instruct"
            elif model_choice == '3':
                base_model = input("Enter HuggingFace model ID: ").strip()
            else:
                base_model = "mistral-community/pixtral-12b"
        else:
            # Use relative path to previous version's trained model within same lineage
            base_model = f"../{latest_name}/models/final"

        print(f"Creating version: {lineage}/{new_version_name}")
        print(f"Previous version: {lineage}/{latest_name}")
        print(f"Base model: {base_model}")

    # Step 3: Select training mode
    print()
    print("Select training mode:")
    print("  1. RL (GRPO reinforcement learning)")
    print("  2. SFT (Supervised fine-tuning)")
    mode_choice = input("Enter choice [1/2]: ").strip()

    if mode_choice == '2':
        training_mode = 'sft'
        print(f"Training mode: SFT")
    else:
        training_mode = 'rl'
        print(f"Training mode: RL")

    print()

    # Create version
    create_version(new_version_name, base_model, training_mode, lineage, lineage_dir)

    print()
    print(f"{'='*60}")
    print(f"Version {lineage}/{new_version_name} created successfully!")
    print(f"{'='*60}")
    print()
    print(f"Next steps:")

    version_path = f"{lineage}/{new_version_name}"

    if training_mode == 'sft':
        print(f"  1. Add training data to: versions/{version_path}/dataset.jsonl")
        print(f"  2. (Optional) Edit config: versions/{version_path}/config.json")
        if lineage == 'generator':
            print(f"  3. Run training: python scripts/train_sft_generator.py {version_path}")
            print(f"  4. Or deploy to RunPod: python scripts/deploy_sft_generator_runpod.py --version {version_path}")
        else:  # critique
            print(f"  3. Run training: python scripts/train_sft_critique.py {version_path}")
            print(f"  4. Or deploy to RunPod: python scripts/deploy_sft_critique_runpod.py --version {version_path}")
    else:
        print(f"  1. (Optional) Edit config: versions/{version_path}/config.json")
        if lineage == 'generator':
            print(f"  2. Critique model will auto-detect latest (critique_model_path: 'auto')")
            print(f"  3. Run training: python scripts/train_rl_generator.py {version_path}")
            print(f"  4. Or deploy to RunPod: python scripts/deploy_rl_generator_runpod.py --version {version_path}")
        else:  # critique - RL training for critique is not typical but supported
            print(f"  2. Run training: python scripts/train_rl_generator.py {version_path}")
            print(f"  3. Or deploy to RunPod: python scripts/deploy_rl_generator_runpod.py --version {version_path}")

    print()


if __name__ == "__main__":
    main()
