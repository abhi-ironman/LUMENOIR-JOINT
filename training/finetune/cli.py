"""Command-line entry point for Zero-DCE++ fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, TrainConfig
from .data import load_train_validation_pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _apply_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    changes = {}
    if args.output_dir is not None:
        changes["output_dir"] = args.output_dir
    if args.resume is not None:
        changes["resume"] = args.resume
    if args.device is not None:
        changes["device"] = args.device
    if args.epochs is not None:
        changes["epochs"] = args.epochs
    if args.batch_size is not None:
        changes["batch_size"] = args.batch_size
    if args.max_train_images is not None:
        changes["max_train_images"] = args.max_train_images
    if args.max_val_images is not None:
        changes["max_val_images"] = args.max_val_images
    return replace(config, **changes)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = TrainConfig.from_file(args.config)
        config = _apply_overrides(config, args)
        config.validate()
        train, validation = load_train_validation_pairs(
            config.train_manifest,
            config.validation_manifest,
            config.data_root,
        )
    except (OSError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    train_paired = sum(pair.has_reference for pair in train)
    validation_paired = sum(pair.has_reference for pair in validation)
    print(
        f"train records: {len(train)} "
        f"({train_paired} paired, {len(train) - train_paired} zero-reference)"
    )
    print(
        f"validation records: {len(validation)} "
        f"({validation_paired} paired metric records, "
        f"{len(validation) - validation_paired} zero-reference records)"
    )
    print(f"datasets: {sorted({pair.dataset for pair in (*train, *validation)})}")
    print(f"output: {config.output_dir}")
    if args.dry_run:
        return 0

    # reference_metrics reads this at import time. Set it before importing the
    # trainer so LPIPS uses the requested GPU when available.
    os.environ.setdefault("LUMENOIR_EVAL_DEVICE", config.metrics_device)
    from .trainer import FineTuner

    try:
        summary = FineTuner(config).fit()
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
