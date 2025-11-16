#!/usr/bin/env python3
"""Deploy SFT critique training job to RunPod with full SSH access."""

import sys
import json
import subprocess
import time
import tarfile
from pathlib import Path
import runpod_utils as ru


def validate_version_for_sft_critique(version):
    """Validate version directory for SFT critique training."""
    version_dir = Path("versions") / version

    if not version_dir.exists():
        print(f"Error: {version_dir} does not exist")
        return False

    required_files = [
        version_dir / "config.json",
        version_dir / "metadata.json",
        version_dir / "dataset.jsonl"
    ]

    for file_path in required_files:
        if not file_path.exists():
            print(f"Error: Required file missing: {file_path}")
            return False

    # Validate training_mode in config
    config_path = version_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    if config.get('training_mode') != 'sft':
        print(f"Error: training_mode is '{config.get('training_mode')}', expected 'sft'")
        return False

    print(f"✓ Version {version} validated for SFT training")
    return True


def sync_files(ssh_config, version):
    """Upload all required files to pod."""
    print(f"\nUploading files to pod...")
    version_dir = Path("versions") / version

    files = [
        (version_dir / "config.json", f"/workspace/{version}/config.json", "Config"),
        (version_dir / "metadata.json", f"/workspace/{version}/metadata.json", "Metadata"),
        (version_dir / "dataset.jsonl", f"/workspace/{version}/dataset.jsonl", "Dataset"),
        (Path("scripts/training_core.py"), "/workspace/scripts/training_core.py", "Training core library"),
        (Path("scripts/model_utils.py"), "/workspace/scripts/model_utils.py", "Model utilities (TRL+PEFT)"),
        (Path("scripts/training_utils.py"), "/workspace/scripts/training_utils.py", "Training utilities (TRL+PEFT)"),
        (Path("scripts/train_sft_critique.py"), "/workspace/scripts/train_sft_critique.py", "Critique SFT training script"),
        (Path("scripts/critique_validation.py"), "/workspace/scripts/critique_validation.py", "Critique validation script"),
        (Path("scripts/vlm_reward.py"), "/workspace/scripts/vlm_reward.py", "VLM reward module"),
        (Path("scripts/generate_samples.py"), "/workspace/scripts/generate_samples.py", "Sample generation script"),
        (Path("scripts/review_samples.py"), "/workspace/scripts/review_samples.py", "Review samples script"),
    ]

    for local_path, remote_path, name in files:
        print(f"  Uploading {name}...")
        ru.upload_file(ssh_config, local_path, remote_path)

    # Upload images directory if it exists (for critique datasets)
    images_dir = version_dir / "images"
    if images_dir.exists() and images_dir.is_dir():
        num_images = len(list(images_dir.glob('*.png')))
        print(f"  Creating archive of {num_images} images...")

        # Create tar.gz archive
        archive_path = version_dir / "images.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(images_dir, arcname="images")

        archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"  Archive created: {archive_size_mb:.1f} MB")

        # Upload archive
        print(f"  Uploading archive...")
        ru.upload_file(ssh_config, archive_path, f"/workspace/{version}/images.tar.gz")

        # Extract on remote
        print(f"  Extracting archive on pod...")
        extract_cmd = ru.get_ssh_cmd(ssh_config) + [
            f"cd /workspace/{version} && tar -xzf images.tar.gz --no-same-owner && rm images.tar.gz"
        ]
        subprocess.run(extract_cmd, check=True)

        # Clean up local archive
        archive_path.unlink()

        print(f"  ✓ Uploaded and extracted {num_images} PNG images")

    # Upload test_images directory if it exists (for post-training critique)
    test_images_dir = version_dir / "test_images"
    if test_images_dir.exists() and test_images_dir.is_dir():
        num_test_images = len(list(test_images_dir.glob('*.png')))
        if num_test_images > 0:
            print(f"  Creating archive of {num_test_images} test images...")

            # Create tar.gz archive
            archive_path = version_dir / "test_images.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(test_images_dir, arcname="test_images")

            archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
            print(f"  Archive created: {archive_size_mb:.1f} MB")

            # Upload archive
            print(f"  Uploading test images archive...")
            ru.upload_file(ssh_config, archive_path, f"/workspace/{version}/test_images.tar.gz")

            # Extract on remote
            print(f"  Extracting test images on pod...")
            extract_cmd = ru.get_ssh_cmd(ssh_config) + [
                f"cd /workspace/{version} && tar -xzf test_images.tar.gz --no-same-owner && rm test_images.tar.gz"
            ]
            subprocess.run(extract_cmd, check=True)

            # Clean up local archive
            archive_path.unlink()

            print(f"  ✓ Uploaded and extracted {num_test_images} test PNG images")

    print(f"✓ All files uploaded")


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
        f"/workspace/venv/bin/python -u scripts/train_sft_critique.py {version}'"
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


def critique_test_images(ssh_config, version):
    """Run trained critique model on test images."""
    version_dir = Path("versions") / version
    test_images_dir = version_dir / "test_images"

    # Check if test_images directory exists locally (indicates it was uploaded)
    if not test_images_dir.exists() or not list(test_images_dir.glob("*.png")):
        print(f"\n⊘ No test images found, skipping critique")
        return 0

    print(f"\n{'='*60}")
    print(f"Running critique on test images...")
    print(f"{'='*60}\n")

    critique_cmd = (
        f"cd /workspace && "
        f"HF_HOME=/workspace/hf-cache "
        f"/workspace/venv/bin/python scripts/critique_validation.py {version}"
    )

    cmd = ru.get_ssh_cmd(ssh_config) + [critique_cmd]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Test image critique complete")
    else:
        print(f"\n⚠️  Warning: Test image critique failed with exit code {result.returncode}")

    return result.returncode


def download_results(ssh_config, version):
    """Download trained model and samples from pod."""
    print(f"\nDownloading results...")
    version_dir = Path("versions") / version

    # Create local directories
    (version_dir / "models" / "final").mkdir(parents=True, exist_ok=True)

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

    # Download test reviews if they exist (as archive for speed)
    print(f"  Downloading test image reviews...")
    check_cmd = ru.get_ssh_cmd(ssh_config) + [f"test -d /workspace/{version}/test_reviews && echo 'exists' || echo 'missing'"]
    check_result = subprocess.run(check_cmd, capture_output=True, text=True)

    if check_result.returncode == 0 and 'exists' in check_result.stdout:
        # Create archive on remote
        archive_cmd = ru.get_ssh_cmd(ssh_config) + [
            f"cd /workspace/{version} && tar -czf test_reviews.tar.gz test_reviews/"
        ]
        subprocess.run(archive_cmd, capture_output=True)

        # Download archive
        (version_dir / "test_reviews").mkdir(parents=True, exist_ok=True)
        archive_local = version_dir / "test_reviews.tar.gz"
        cmd = ru.get_scp_cmd(ssh_config) + [
            f"root@{ssh_config['SSH_HOST']}:/workspace/{version}/test_reviews.tar.gz",
            str(archive_local)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and archive_local.exists():
            # Extract locally
            with tarfile.open(archive_local, "r:gz") as tar:
                tar.extractall(version_dir)

            # Count and cleanup
            txt_files = list((version_dir / "test_reviews").glob("*.txt"))
            archive_local.unlink()

            if txt_files:
                print(f"  ✓ Downloaded {len(txt_files)} test review files")
            else:
                print(f"  ⊘ No test reviews (no test images provided)")
        else:
            print(f"  ⊘ Failed to download test reviews archive")
    else:
        print(f"  ⊘ No test reviews found")

    # Download training samples (generated during training with reviews)
    print(f"  Downloading training samples and reviews...")
    (version_dir / "training_samples").mkdir(parents=True, exist_ok=True)
    cmd = ru.get_scp_cmd(ssh_config) + [
        "-r",
        f"root@{ssh_config['SSH_HOST']}:/workspace/{version}/training_samples/*",
        str(version_dir / "training_samples/")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        training_svg_files = list((version_dir / "training_samples").glob("*.svg"))
        training_txt_files = list((version_dir / "training_samples").glob("*.txt"))
        print(f"  ✓ Downloaded {len(training_svg_files)} training SVG files")
        print(f"  ✓ Downloaded {len(training_txt_files)} training review files")
    else:
        print(f"  ⚠️  Could not download training samples (may not exist)")

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
        print("Usage: python scripts/deploy_sft_critique_runpod.py --version <version>")
        print("Example: python scripts/deploy_sft_critique_runpod.py --version critique/v1")
        sys.exit(1)

    version = sys.argv[2]

    # Prompt for SSH configuration
    ssh_config = ru.prompt_ssh_config()

    print(f"\n{'='*60}")
    print(f"Deploying {version} to RunPod")
    print(f"Training type: SFT with TRL+PEFT")
    print(f"{'='*60}\n")

    # Validate version
    if not validate_version_for_sft_critique(version):
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

    # Run critique on test images if they exist
    critique_result = critique_test_images(ssh_config, version)

    if critique_result != 0:
        print(f"\n⚠️  Test image critique failed but continuing with download...")

    # Download results (model + test_reviews if they exist)
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
    print(f"Final samples: versions/{version}/samples/")
    print(f"Final reviews: versions/{version}/samples/*.txt")
    print(f"Training samples: versions/{version}/training_samples/")
    print(f"Training reviews: versions/{version}/training_samples/*.txt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
