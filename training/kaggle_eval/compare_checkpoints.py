"""Run a controlled multi-checkpoint / Retinex / HueLock comparison matrix.

Example:
    python -m training.kaggle_eval.compare_checkpoints \
        --config training/kaggle_eval/example_config.json \
        --checkpoint friend=/kaggle/input/friend-model/zero_dce_plus_plus.pth \
        --checkpoint abhi=/kaggle/input/abhi-model/zero_dce_plus_plus.pth \
        --datasets lol_v1,lol_v2_real,lol_v2_syn \
        --device cuda \
        --output-dir /kaggle/working/lumenoir-joint-matrix

This runs an original/classical reference once, then evaluates deep-learning and
hybrid modes for each checkpoint under every requested parameter combination.
The two environment-controlled extensions remain opt-in; no weights are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from .config import EvalConfig
from .runner import RunSummary, run_evaluation

RETINEX_ENV = "LUMENOIR_RETINEX_DOWNSAMPLE"
HUE_ANCHOR_ENV = "LUMENOIR_HUELOCK_SOURCE_ANCHOR"\nMODEL_SCALE_ENV = "LUMENOIR_ZERODCE_SCALE_FACTOR"
DEFAULT_DATASETS = ("lol_v1", "lol_v2_real", "lol_v2_syn")


def _checkpoint_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Checkpoint must use LABEL=/absolute/path/to/checkpoint.pth"
        )
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", label):
        raise argparse.ArgumentTypeError(
            "Checkpoint label may contain only letters, digits, '_' and '-'."
        )
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Checkpoint file does not exist: {path}")
    return label, path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _pipeline_environment(\n    retinex_downsample: int, hue_anchor: float, model_scale_factor: int = 1\n) -> Iterator[None]:
    if not 1 <= retinex_downsample <= 4:
        raise ValueError("Retinex downsample must be in [1, 4].")
    if not 0.0 <= hue_anchor <= 0.55:
        raise ValueError("HueLock source anchor must be in [0, 0.55].")
    if not 1 <= model_scale_factor <= 12:
        raise ValueError("Zero-DCE++ scale factor must be in [1, 12].")

    keys = (RETINEX_ENV, HUE_ANCHOR_ENV, MODEL_SCALE_ENV)
    previous = {key: os.environ.get(key) for key in keys}
    os.environ[RETINEX_ENV] = str(retinex_downsample)
    os.environ[HUE_ANCHOR_ENV] = f"{hue_anchor:.4f}".rstrip("0").rstrip(".")
    os.environ[MODEL_SCALE_ENV] = str(model_scale_factor)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _summary_dict(summary: RunSummary) -> dict[str, object]:
    return {
        "output_dir": str(summary.output_dir),
        "record_count": summary.record_count,
        "failure_count": summary.failure_count,
        "skipped_modes": list(summary.skipped_modes),
        "warnings": list(summary.warnings),
        "aggregate_paths": [str(path) for path in summary.aggregate_paths],
    }


def _run_one(
    config: EvalConfig,
    output_dir: Path,
    *,
    retinex_downsample: int,
    hue_anchor: float,
    model_scale_factor: int,
) -> dict[str, object]:
    with _pipeline_environment(retinex_downsample, hue_anchor, model_scale_factor):
        summary = run_evaluation(config.replace(output_dir=output_dir))
    return {
        "retinex_downsample": retinex_downsample,
        "hue_lock_source_anchor": hue_anchor,
        "model_scale_factor": model_scale_factor,
        **_summary_dict(summary),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_checkpoint_arg,
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each checkpoint; each file is SHA-256 recorded and left untouched.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated paired dataset keys (default: LOL-v1, LOL-v2 Real, LOL-v2 Synthetic).",
    )
    parser.add_argument("--device", default=None, help="Override configured device.")
    parser.add_argument("--latency-warmups", type=int, default=None)
    parser.add_argument("--latency-runs", type=int, default=None)
    parser.add_argument(
        "--model-scale-factor", type=int, default=1,
        help="Use the same Zero-DCE++ inference scale for every checkpoint (default 1, matching the fine-tuning config).",
    )
    parser.add_argument(
        "--retinex-downsample",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Retinex illumination divisors to compare (1-4).",
    )
    parser.add_argument(
        "--hue-anchor",
        type=float,
        nargs="+",
        default=[0.0, 0.45],
        help="HueLock source-anchor strengths (0-0.55).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/reports/joint-comparison"),
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip the one-time Original/Classical reference run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        base_config = EvalConfig.from_file(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"Could not load evaluation config: {exc}")

    if not 1 <= args.model_scale_factor <= 12:
        parser.error("--model-scale-factor must be between 1 and 12.")
    if len({label for label, _ in args.checkpoint}) != len(args.checkpoint):
        parser.error("Checkpoint labels must be unique.")
    if len({path for _, path in args.checkpoint}) != len(args.checkpoint):
        parser.error("The same checkpoint path was supplied more than once.")
    if any(value not in range(1, 5) for value in args.retinex_downsample):
        parser.error("--retinex-downsample values must be between 1 and 4.")
    if any(not 0.0 <= value <= 0.55 for value in args.hue_anchor):
        parser.error("--hue-anchor values must be between 0 and 0.55.")
    if len(set(args.retinex_downsample)) != len(args.retinex_downsample):
        parser.error("--retinex-downsample values must be unique.")
    if len(set(args.hue_anchor)) != len(args.hue_anchor):
        parser.error("--hue-anchor values must be unique.")

    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    if not datasets:
        parser.error("At least one dataset must be selected.")
    try:
        base_config = base_config.replace(
            selected_datasets=datasets,
            modes=("original", "classical"),
            output_dir=args.output_dir / "reference",
            weights=None,
            allow_missing_model=True,
        )
        if args.device is not None:
            base_config = base_config.replace(device=args.device)
        if args.latency_warmups is not None:
            base_config = base_config.replace(latency_warmups=args.latency_warmups)
        if args.latency_runs is not None:
            base_config = base_config.replace(latency_runs=args.latency_runs)
    except ValueError as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix: dict[str, object] = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_file": str(args.config.resolve()),
        "selected_datasets": list(datasets),
        "device": base_config.device,
        "model_scale_factor": args.model_scale_factor,
        "latency_warmups": base_config.latency_warmups,
        "latency_runs": base_config.latency_runs,
        "retinex_downsample_values": args.retinex_downsample,
        "hue_anchor_values": args.hue_anchor,
        "checkpoints": [
            {"label": label, "path": str(path), "sha256": _sha256(path)}
            for label, path in args.checkpoint
        ],
        "reference_run": None,
        "model_runs": [],
        "errors": [],
    }

    if not args.skip_reference:
        try:
            with _pipeline_environment(2, 0.0, args.model_scale_factor):
                reference = run_evaluation(base_config)
            matrix["reference_run"] = _summary_dict(reference)
        except Exception as exc:
            matrix["errors"].append({"run": "original-classical-reference", "error": str(exc)})

    model_config = base_config.replace(modes=("deep_learning", "hybrid"))
    for label, checkpoint in args.checkpoint:
        for downsample in args.retinex_downsample:
            for anchor in args.hue_anchor:
                run_label = (
                    f"{label}/retinex-{downsample}/hue-anchor-{anchor:g}"
                )
                run_dir = args.output_dir / label / f"retinex-{downsample}_anchor-{anchor:g}"
                try:
                    run_config = model_config.replace(
                        weights=checkpoint,
                        output_dir=run_dir,
                        allow_missing_model=False,
                    )
                    outcome = _run_one(
                        run_config,
                        run_dir,
                        retinex_downsample=downsample,
                        hue_anchor=anchor,
                        model_scale_factor=args.model_scale_factor,
                    )
                    outcome.update({"checkpoint_label": label, "checkpoint_sha256": _sha256(checkpoint)})
                    matrix["model_runs"].append(outcome)
                    print(
                        f"[{run_label}] records={outcome['record_count']} "
                        f"failures={outcome['failure_count']} -> {run_dir}"
                    )
                except Exception as exc:
                    matrix["errors"].append({"run": run_label, "error": str(exc)})
                    print(f"[{run_label}] FAILED: {exc}", file=sys.stderr)

    matrix["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    report_path = args.output_dir / "matrix_manifest.json"
    report_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"Matrix manifest: {report_path}")

    if not matrix["model_runs"]:
        return 1
    if matrix["errors"]:
        print(f"Completed with {len(matrix['errors'])} error(s); inspect matrix_manifest.json.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
