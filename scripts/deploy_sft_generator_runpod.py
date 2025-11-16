#!/usr/bin/env python3
"""Deploy SFT generator training job to RunPod with full SSH access."""

import sys
import json
import subprocess
import time
from pathlib import Path
import runpod_utils as ru


def validate_version_for_sft_generator(version):
    """Validate version directory for SFT generator training."""
    version_dir = Path("versions") / version

    if not version_dir.exists():
        print(f"Error: {version_dir} does not exist")
        return False

    # Check required files
    config_path = version_dir / "config.json"
    metadata_path = version_dir / "metadata.json"
    dataset_path = version_dir / "dataset.jsonl"
    dataset_example_path = version_dir / "dataset_example.jsonl"

    if not config_path.exists():
        print(f"Error: Required file missing: {config_path}")
        return False

    if not metadata_path.exists():
        print(f"Error: Required file missing: {metadata_path}")
        return False

    # Check for dataset (either full or example) - we'll download from HF if needed
    if not dataset_path.exists() and not dataset_example_path.exists():
        print(f"Error: No dataset found (not even dataset_example.jsonl)")
        print(f"   The script will attempt to download from HuggingFace")
        print(f"   If that fails, create dataset_example.jsonl in {version_dir}")
        return False

    # Validate training_mode in config
    with open(config_path, 'r') as f:
        config = json.load(f)

    if config.get('training_mode') != 'sft':
        print(f"Error: training_mode is '{config.get('training_mode')}', expected 'sft'")
        return False

    print(f"✓ Version {version} validated for SFT training")
    return True


def download_dataset_on_pod(ssh_config, version):
    """Download dataset from HuggingFace on the pod if needed."""
    print(f"\n{'='*60}")
    print(f"Downloading dataset from HuggingFace on pod...")
    print(f"{'='*60}\n")

    download_script = f"""
import json
from datasets import load_dataset
from pathlib import Path
import shutil

dataset_path = Path("/workspace/{version}/dataset.jsonl")
dataset_example_path = Path("/workspace/{version}/dataset_example.jsonl")

# Check if already exists
if dataset_path.exists():
    print(f"Dataset already exists at {{dataset_path}}")
    exit(0)

# Try to download from HuggingFace
try:
    print("Loading from HuggingFace: yoavf/svg-animal-illustrations")
    dataset = load_dataset("yoavf/svg-animal-illustrations", split="train")

    print(f"Saving {{len(dataset)}} examples to {{dataset_path}}...")
    with open(dataset_path, 'w') as f:
        for example in dataset:
            json.dump(example, f)
            f.write('\\n')

    print(f"✓ Dataset downloaded ({{len(dataset)}} examples)")
except Exception as e:
    print(f"Failed to download from HuggingFace: {{e}}")
    # Fallback to example dataset
    if dataset_example_path.exists():
        print(f"Copying dataset_example.jsonl to dataset.jsonl as fallback...")
        shutil.copy(dataset_example_path, dataset_path)
        print(f"⚠️  Using example dataset (3 samples only)")
    else:
        print(f"ERROR: No fallback dataset available!")
        exit(1)
"""

    # Write script to temp file on pod and execute
    script_path = f"/tmp/download_dataset_{version.replace('/', '_')}.py"

    # Upload script
    write_script_cmd = ru.get_ssh_cmd(ssh_config) + [
        f"cat > {script_path} << 'EOFSCRIPT'\n{download_script}\nEOFSCRIPT"
    ]
    subprocess.run(write_script_cmd)

    # Execute script
    cmd = ru.get_ssh_cmd(ssh_config) + [
        f"cd /workspace && /workspace/venv/bin/python {script_path}"
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"✓ Dataset ready on pod")
        return True
    else:
        print(f"⚠️  Dataset setup failed - check logs above")
        return False


def sync_files(ssh_config, version):
    """Upload all required files to pod."""
    print(f"\nUploading files to pod...")
    version_dir = Path("versions") / version

    # Check if we have dataset locally
    dataset_path = version_dir / "dataset.jsonl"
    dataset_example_path = version_dir / "dataset_example.jsonl"

    # Upload example dataset first (as fallback)
    upload_dataset = dataset_example_path if dataset_example_path.exists() else None
    dataset_name = "Dataset (example - 3 samples, will download full from HF)"

    if dataset_path.exists():
        # User has full dataset locally
        upload_dataset = dataset_path
        dataset_name = "Dataset (full - from local file)"

    files = [
        (version_dir / "config.json", f"/workspace/{version}/config.json", "Config"),
        (version_dir / "metadata.json", f"/workspace/{version}/metadata.json", "Metadata"),
        (Path("scripts/training_core.py"), "/workspace/scripts/training_core.py", "Training core library"),
        (Path("scripts/model_utils.py"), "/workspace/scripts/model_utils.py", "Model utilities (TRL+PEFT)"),
        (Path("scripts/training_utils.py"), "/workspace/scripts/training_utils.py", "Training utilities (TRL+PEFT)"),
        (Path("scripts/train_sft_generator.py"), "/workspace/scripts/train_sft_generator.py", "Generator SFT training script"),
        (Path("scripts/vlm_reward.py"), "/workspace/scripts/vlm_reward.py", "VLM reward module"),
        (Path("scripts/generate_samples.py"), "/workspace/scripts/generate_samples.py", "Sample generation script"),
    ]

    # Add dataset if we have one to upload
    if upload_dataset:
        files.insert(2, (upload_dataset, f"/workspace/{version}/dataset_example.jsonl", dataset_name))

    for local_path, remote_path, name in files:
        print(f"  Uploading {name}...")
        ru.upload_file(ssh_config, local_path, remote_path)

    print(f"✓ All files uploaded")

    # Download full dataset from HF on the pod if we only uploaded example
    if not dataset_path.exists():
        download_dataset_on_pod(ssh_config, version)


def run_training(ssh_config, version, wandb_key=None):
    """Run training on pod with streaming output."""
    print(f"\n{'='*60}")
    print(f"Starting SFT training for {version}")
    print(f"{'='*60}\n")

    session_name = f"training_{version.replace('/', '_')}"

    # Always kill any existing session to ensure clean start
    print(f"Cleaning up any existing training sessions...")
    kill_cmd = ru.get_ssh_cmd(ssh_config) + [f"tmux kill-session -t {session_name} 2>/dev/null || true"]
    subprocess.run(kill_cmd, capture_output=True)

    # Also delete old log file to prevent tail from reading stale output
    log_file = f"/workspace/{version}_training.log"
    rm_log_cmd = ru.get_ssh_cmd(ssh_config) + [f"rm -f {log_file}"]
    subprocess.run(rm_log_cmd, capture_output=True)

    # Build environment variables for training
    env_vars = f"HF_HOME=/workspace/hf-cache"
    if wandb_key:
        env_vars += f" WANDB_API_KEY={wandb_key}"

    # Start training in tmux
    start_cmd = ru.get_ssh_cmd(ssh_config) + [
        f"cd /workspace && "
        f"tmux new-session -d -s {session_name} "
        f"'"
        f"{env_vars} "
        f"/workspace/venv/bin/python -u scripts/train_sft_generator.py {version}'"
    ]

    print("Starting training session in tmux...")
    result = subprocess.run(start_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Failed to start training session")
        print(f"Error: {result.stderr}")
        return 1

    print(f"✓ Training session started in tmux: {session_name}")

    # Enable tmux logging to capture output
    logging_cmd = ru.get_ssh_cmd(ssh_config) + [
        f"tmux pipe-pane -t {session_name} -o 'cat >> {log_file}'"
    ]
    subprocess.run(logging_cmd)

    print(f"Monitoring training progress (streaming output)...\n")
    time.sleep(1)

    # Stream the log file while tmux session is alive
    monitor_cmd = ru.get_ssh_cmd(ssh_config) + [
        f"tail -f --retry {log_file} & "
        f"TAIL_PID=$!; "
        f"while tmux has-session -t {session_name} 2>/dev/null; do sleep 2; done; "
        f"sleep 3; "
        f"kill $TAIL_PID 2>/dev/null || true; "
        f"sleep 1; "
        f"cat {log_file} | tail -n 20"
    ]

    result = subprocess.run(monitor_cmd)
    print(f"\n✓ Training completed")
    return 0


def generate_samples(ssh_config, version):
    """Generate sample SVGs from trained model on RunPod."""
    print(f"\n{'='*60}")
    print(f"Generating 10 sample SVGs from trained model...")
    print(f"{'='*60}\n")

    generate_cmd = (
        f"cd /workspace && "
        f"HF_HOME=/workspace/hf-cache "
        f"/workspace/venv/bin/python scripts/generate_samples.py "
        f"--model /workspace/{version}/models/final "
        f"--count 10 "
        f"--output /workspace/{version}/samples"
    )

    cmd = ru.get_ssh_cmd(ssh_config) + [generate_cmd]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\n✓ Sample generation complete")
    else:
        print(f"\n⚠️  Warning: Sample generation failed with exit code {result.returncode}")

    return result.returncode


def download_results(ssh_config, version):
    """Download trained model and samples from pod."""
    print(f"\nDownloading results...")
    version_dir = Path("versions") / version

    # Create local directories
    (version_dir / "models" / "final").mkdir(parents=True, exist_ok=True)
    (version_dir / "samples").mkdir(parents=True, exist_ok=True)

    # Download final model
    print(f"  Downloading final model...")
    cmd = ru.get_scp_cmd(ssh_config) + [
        "-r",
        f"root@{ssh_config['SSH_HOST']}:/workspace/{version}/models/final/*",
        str(version_dir / "models" / "final/")
    ]
    final_result = subprocess.run(cmd, capture_output=True, text=True)

    if final_result.returncode == 0:
        print(f"  ✓ Final model downloaded")
    else:
        print(f"  ⚠️  Final model not found (training may have failed)")

    # Download checkpoints
    print(f"  Downloading checkpoints...")
    (version_dir / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
    cmd = ru.get_scp_cmd(ssh_config) + [
        "-r",
        f"root@{ssh_config['SSH_HOST']}:/workspace/{version}/models/checkpoints/*",
        str(version_dir / "models" / "checkpoints/")
    ]
    checkpoint_result = subprocess.run(cmd, capture_output=True, text=True)

    if checkpoint_result.returncode == 0:
        checkpoint_dirs = list((version_dir / "models" / "checkpoints").glob("checkpoint-*"))
        print(f"  ✓ Downloaded {len(checkpoint_dirs)} checkpoint(s)")
    else:
        print(f"  ⚠️  No checkpoints found")

    if final_result.returncode != 0 and checkpoint_result.returncode != 0:
        print(f"  ⚠️  No model files downloaded - training likely failed")

    # Download samples
    print(f"  Downloading samples...")
    cmd = ru.get_scp_cmd(ssh_config) + [
        "-r",
        f"root@{ssh_config['SSH_HOST']}:/workspace/{version}/samples/*",
        str(version_dir / "samples/")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        svg_files = list((version_dir / "samples").glob("*.svg"))
        prompt_files = list((version_dir / "samples").glob("*_prompt.txt"))
        print(f"  ✓ Downloaded {len(svg_files)} SVG files")
        print(f"  ✓ Downloaded {len(prompt_files)} prompt files")
    else:
        print(f"  ⚠️  Could not download samples")

    # Download metadata
    ru.download_metadata(ssh_config, version)

    # Download training log
    print(f"  Downloading training log...")
    cmd = ru.get_scp_cmd(ssh_config) + [
        f"root@{ssh_config['SSH_HOST']}:/workspace/{version}_training.log",
        str(version_dir / "training.log")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Training log downloaded")

    print(f"✓ Download complete")


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "--version":
        print("Usage: python scripts/deploy_sft_generator_runpod.py --version <version>")
        print("Example: python scripts/deploy_sft_generator_runpod.py --version generator/v1")
        sys.exit(1)

    version = sys.argv[2]

    # Prompt for SSH configuration
    ssh_config = ru.prompt_ssh_config()

    print(f"\n{'='*60}")
    print(f"Deploying {version} to RunPod")
    print(f"Training type: SFT with TRL+PEFT")
    print(f"{'='*60}\n")

    # Validate version
    if not validate_version_for_sft_generator(version):
        sys.exit(1)

    # Check SSH connection
    if not ru.check_ssh_connection(ssh_config):
        print(f"\nPlease check SSH connection details in {ru.CONFIG_FILE}")
        sys.exit(1)

    # Create directories
    ru.create_directories(ssh_config, version)

    # Upload files
    sync_files(ssh_config, version)

    # Install dependencies
    ru.install_dependencies(ssh_config)

    # Get Weights & Biases API key
    wandb_key = ru.get_wandb_key(ssh_config)

    # Run training
    training_result = run_training(ssh_config, version, wandb_key)

    if training_result != 0:
        print(f"\n⚠️  Training exited with non-zero code. Checking if it completed...")

    # Generate samples from trained model
    generate_result = generate_samples(ssh_config, version)

    if generate_result != 0:
        print(f"\n⚠️  Sample generation failed but continuing with download...")

    # Download results (model + samples)
    download_results(ssh_config, version)

    # Prompt for shutdown with timeout
    should_stop = ru.prompt_shutdown_with_timeout()

    pod_id = ssh_config.get('POD_ID', '')
    if should_stop and pod_id:
        ru.stop_pod(pod_id, pod_id)
    elif not pod_id:
        print(f"\n✓ Training complete. Pod still running.")
        print(f"   Stop manually via RunPod web interface")
    else:
        print(f"\n✓ Training complete. Pod still running.")
        print(f"   Stop manually with: runpodctl stop pod {pod_id}")

    print(f"\n{'='*60}")
    print(f"✓ {version} training complete!")
    print(f"Final model: versions/{version}/models/final/")
    print(f"Checkpoints: versions/{version}/models/checkpoints/")
    print(f"Samples: versions/{version}/samples/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
