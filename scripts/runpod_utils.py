#!/usr/bin/env python3
"""Shared utilities for RunPod deployment scripts."""

import sys
import json
import subprocess
from pathlib import Path
from threading import Thread, Event

# Default SSH connection details
CONFIG_FILE = Path.home() / ".runpod_pet_config.json"


def load_ssh_config():
    """Load SSH config from file or use defaults."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "SSH_HOST": "localhost",
        "SSH_PORT": "22",
        "SSH_KEY": "~/.ssh/id_ed25519",
    }


def save_ssh_config(config):
    """Save SSH config to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config to {CONFIG_FILE}")


def prompt_ssh_config():
    """Prompt user for SSH connection details."""
    config = load_ssh_config()

    print(f"\nCurrent SSH configuration:")
    print(f"  Host: {config['SSH_HOST']}")
    print(f"  Port: {config['SSH_PORT']}")
    print(f"  Key: {config['SSH_KEY']}")
    print(f"  Pod ID: {config.get('POD_ID', '')}")
    wandb_key = config.get('WANDB_API_KEY', '')
    wandb_display = f"{wandb_key[:8]}..." if wandb_key else "(not set)"
    print(f"  WandB API Key: {wandb_display}")
    print()

    response = input("Use current config? [Y/n]: ").strip().lower()

    if response in ['n', 'no']:
        config['SSH_HOST'] = input(f"Enter host [{config['SSH_HOST']}]: ").strip() or config['SSH_HOST']
        config['SSH_PORT'] = input(f"Enter port [{config['SSH_PORT']}]: ").strip() or config['SSH_PORT']
        config['SSH_KEY'] = input(f"Enter SSH key path [{config['SSH_KEY']}]: ").strip() or config['SSH_KEY']
        config['POD_ID'] = input(f"Enter pod ID [{config.get('POD_ID', '')}]: ").strip() or config.get('POD_ID', '')

        # Prompt for WandB API key (optional)
        current_wandb = config.get('WANDB_API_KEY', '')
        wandb_prompt = f"Enter WandB API key (press Enter to skip): " if not current_wandb else f"Enter WandB API key (press Enter to keep current): "
        new_wandb = input(wandb_prompt).strip()
        if new_wandb:
            config['WANDB_API_KEY'] = new_wandb
        elif not current_wandb:
            config['WANDB_API_KEY'] = ''

        save_ssh_config(config)

    return config


def get_ssh_cmd(ssh_config):
    """Get base SSH command."""
    return [
        "ssh",
        f"root@{ssh_config['SSH_HOST']}",
        "-p", ssh_config['SSH_PORT'],
        "-i", ssh_config['SSH_KEY']
    ]


def get_scp_cmd(ssh_config):
    """Get base SCP command."""
    return [
        "scp",
        "-P", ssh_config['SSH_PORT'],
        "-i", ssh_config['SSH_KEY']
    ]


def validate_version(version):
    """Validate version directory exists and has required files."""
    version_dir = Path("versions") / version

    if not version_dir.exists():
        print(f"Error: {version_dir} does not exist")
        return False

    required_files = [
        version_dir / "config.json",
        version_dir / "metadata.json"
    ]

    for file_path in required_files:
        if not file_path.exists():
            print(f"Error: Required file missing: {file_path}")
            return False

    print(f"✓ Version {version} validated")
    return True


def check_ssh_connection(ssh_config):
    """Test SSH connection to pod."""
    ssh_host = ssh_config['SSH_HOST']
    print(f"\nTesting SSH connection to {ssh_host}...")

    try:
        cmd = get_ssh_cmd(ssh_config) + ["echo 'Connected successfully'"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print(f"✓ SSH connection successful")
            return True
        else:
            print(f"Error: SSH connection failed")
            print(f"Output: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Error: SSH connection timed out")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def create_directories(ssh_config, version):
    """Create workspace directories on pod."""
    print(f"\nCreating directories on pod...")

    cmd = get_ssh_cmd(ssh_config) + [
        f"mkdir -p /workspace/{version}/models /workspace/scripts"
    ]

    subprocess.run(cmd, check=True)
    print(f"✓ Directories created")


def upload_file(ssh_config, local_path, remote_path):
    """Upload a file to the pod using SCP."""
    cmd = get_scp_cmd(ssh_config) + [
        str(local_path),
        f"root@{ssh_config['SSH_HOST']}:{remote_path}"
    ]

    subprocess.run(cmd, check=True)


def install_dependencies(ssh_config):
    """Install Python dependencies on pod."""
    print(f"\nInstalling dependencies...")

    # Install system packages (libcairo2 for SVG rendering, python3-venv for virtual environments)
    print(f"  Installing system packages (libcairo2, python3-venv)...")
    system_cmd = get_ssh_cmd(ssh_config) + [
        "apt-get update && apt-get install -y libcairo2 python3-venv"
    ]
    result = subprocess.run(system_cmd)

    if result.returncode == 0:
        print(f"  ✓ System packages installed")
    else:
        print(f"  Warning: System package installation failed with exit code {result.returncode}")

    # Create virtual environment with system site packages (to inherit torch, torchvision, etc.)
    print(f"  Creating virtual environment (inheriting system packages)...")
    venv_cmd = get_ssh_cmd(ssh_config) + [
        "python3 -m venv --system-site-packages /workspace/venv"
    ]
    result = subprocess.run(venv_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Virtual environment created")
    else:
        print(f"ERROR: Failed to create virtual environment")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)

    # Install Python packages in virtual environment
    print(f"  Installing Python packages in venv (transformers, trl, peft, bitsandbytes, etc.)...")
    pip_cmd = (
        "/workspace/venv/bin/pip install --upgrade pip && "
        "/workspace/venv/bin/pip install --upgrade transformers trl peft bitsandbytes accelerate datasets "
        "cairosvg pillow wandb safetensors"
    )
    ssh_base = f"ssh root@{ssh_config['SSH_HOST']} -p {ssh_config['SSH_PORT']} -i {ssh_config['SSH_KEY']}"
    full_cmd = f"{ssh_base} '{pip_cmd}'"
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Dependencies installed")
    else:
        print(f"ERROR: Dependency installation failed with exit code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"\nCannot continue without dependencies. Exiting.")
        sys.exit(1)


def get_wandb_key(config):
    """Get WANDB API key from config or environment variable."""
    import os
    wandb_key = config.get('WANDB_API_KEY') or os.environ.get('WANDB_API_KEY')
    if wandb_key:
        print(f"\n✓ WandB API key found - will be passed to training script")
    else:
        print(f"\n⚠️  Warning: WANDB_API_KEY not set - training will run without experiment tracking")
        print(f"  Set it in the config during SSH setup or via WANDB_API_KEY environment variable")
    return wandb_key


def download_metadata(ssh_config, version):
    """Download metadata from pod."""
    print(f"  Downloading metadata...")
    version_dir = Path("versions") / version

    cmd = get_scp_cmd(ssh_config) + [
        f"root@{ssh_config['SSH_HOST']}:/workspace/{version}/metadata.json",
        str(version_dir / "metadata.json")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Metadata downloaded")
    else:
        print(f"  ⚠️  Could not download metadata")


def prompt_shutdown_with_timeout():
    """Prompt user to shut down pod with 5-minute timeout."""
    print(f"\n{'='*60}")
    print(f"Training and sample generation complete!")
    print(f"{'='*60}\n")

    print(f"Do you want to shut down the pod? (y/n)")
    print(f"Waiting 5 minutes for response...")
    print(f"(No response = automatic shutdown)\n")

    user_response = [None]
    input_received = Event()

    def get_input():
        try:
            response = input("Your choice [y/n]: ").strip().lower()
            user_response[0] = response
            input_received.set()
        except:
            pass

    # Start input thread (daemon so it won't prevent script exit)
    input_thread = Thread(target=get_input, daemon=True)
    input_thread.start()

    # Wait up to 300 seconds (5 minutes) for input
    if input_received.wait(timeout=300):
        # User responded within timeout
        if user_response[0] in ['y', 'yes', '']:
            return True
        else:
            print(f"\nPod will remain running.")
            return False
    else:
        # Timeout - no response
        print(f"\n⏰ Timeout reached (5 minutes). Shutting down pod automatically...")
        return True


def stop_pod(pod_id, pod_name):
    """Stop the RunPod instance."""
    print(f"\nStopping RunPod instance {pod_id}...")

    cmd = ["runpodctl", "stop", "pod", pod_id]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Pod {pod_name} stopped successfully")
    else:
        print(f"Warning: Failed to stop pod")
        print(f"Error: {result.stderr}")
        print(f"You can manually stop it with: runpodctl stop pod {pod_id}")
