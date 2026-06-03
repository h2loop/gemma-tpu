#!/usr/bin/env python3
"""
CLI entry point for Gemma TPU training.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume latest
    python train.py --config configs/default.yaml --max-steps 100  # override config
"""

import argparse
import os
import sys
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Merge CLI arguments into config, CLI takes precedence."""
    if args.model_path:
        config.setdefault("model", {})["path"] = args.model_path
    if args.dataset_path:
        config.setdefault("dataset", {})["path"] = args.dataset_path
    if args.output_dir:
        config.setdefault("checkpointing", {})["output_dir"] = args.output_dir
    if args.max_steps:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    if args.learning_rate:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.batch_size:
        config.setdefault("training", {})["per_device_batch_size"] = args.batch_size
    if args.max_seq_len:
        config.setdefault("training", {})["max_seq_len"] = args.max_seq_len
    if args.resume:
        config.setdefault("checkpointing", {})["resume_from"] = args.resume
    if args.lora_rank:
        config.setdefault("lora", {})["rank"] = args.lora_rank
    if args.lora_alpha:
        config.setdefault("lora", {})["alpha"] = args.lora_alpha
    return config


def validate_config(config: dict) -> None:
    """Validate required config fields."""
    required = [
        ("model.path", config.get("model", {}).get("path")),
        ("dataset.path", config.get("dataset", {}).get("path")),
        ("checkpointing.output_dir", config.get("checkpointing", {}).get("output_dir")),
    ]
    missing = [name for name, val in required if not val or "YOUR_" in str(val)]
    if missing:
        print(f"Error: Missing or placeholder values in config: {missing}")
        print("Please update your config file or provide CLI overrides.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Gemma TPU LoRA fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config file
  python train.py --config configs/default.yaml

  # Override config values via CLI
  python train.py --config configs/default.yaml --max-steps 500 --learning-rate 5e-5

  # Resume from checkpoint
  python train.py --config configs/default.yaml --resume latest
        """
    )

    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    parser.add_argument("--model-path", help="Override model.path")
    # model-variant removed: script only supports gemma4_31b currently
    parser.add_argument("--dataset-path", help="Override dataset.path")
    parser.add_argument("--output-dir", help="Override checkpointing.output_dir")
    parser.add_argument("--max-steps", type=int, help="Override training.max_steps")
    parser.add_argument("--learning-rate", type=float, help="Override training.learning_rate")
    parser.add_argument("--batch-size", type=int, help="Override training.per_device_batch_size")
    parser.add_argument("--max-seq-len", type=int, help="Override training.max_seq_len")
    parser.add_argument("--resume", help="Resume from checkpoint ('latest' or step number)")
    parser.add_argument("--lora-rank", type=int, help="Override lora.rank")
    parser.add_argument("--lora-alpha", type=int, help="Override lora.alpha")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")

    args = parser.parse_args()

    # Load and merge config
    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    if args.dry_run:
        print("=== Effective Configuration ===")
        print(yaml.dump(config, default_flow_style=False))
        return

    validate_config(config)

    # Set environment variables
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TMPDIR", "/dev/shm")

    # Import training module and run
    print(f"Loading config from: {args.config}")
    print(f"Model: {config['model']['path']}")
    print(f"Dataset: {config['dataset']['path']}")
    print(f"Output: {config['checkpointing']['output_dir']}")

    # Import here to avoid slow JAX import if just checking --help
    from gemma4_lora_sft import run_training
    run_training(config)


if __name__ == "__main__":
    main()
