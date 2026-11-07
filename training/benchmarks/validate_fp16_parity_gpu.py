"""FP32 vs FP16 numerical and visual parity validation for Lumenoir.

Runs the complete enhancement pipeline twice on the *same* CUDA device — once
in float32, once in float16 — over the *same* input images, and compares the
final enhanced images for classical, deep-learning and hybrid processing.

For every mode the report contains MAE, maximum absolute pixel difference,
PSNR (dB) and SSIM, both per image and aggregated, plus a side-by-side
comparison panel per image (input | float32 | float16 | amplified difference).
Results are written as JSON and Markdown.

CUDA only:

* ``--device`` must be ``cuda`` or ``cuda:N``; ``cpu``, ``mps`` and ``auto``
  are refused instead of silently validating the host;
* both precisions must resolve to the same CUDA device;
* the NumPy/OpenCV stage entry points are trapped, so a stage that executes its
  CPU implementation fails the run instead of contaminating the comparison;
* the loaded checkpoint is audited so the float16 run is proven to hold
  ``torch.float16`` parameters and the float32 run ``torch.float32``.

This script does not measure latency, does not modify enhancement parameters,
weights or outputs, and the images it compares are always *copied out of the
pipeline result before the result is released*.

Example (Kaggle T4):

    python -m training.benchmarks.validate_fp16_parity_gpu \\
        --device cuda \\
        --images datasets/raw/LOL-v2-Real/Test/Low \\
        --count 10 \\
        --output-dir /kaggle/working/fp16_parity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import deep_learning  # noqa: E402,F401  (registers zero_dce_plus_plus)
from backend.config import InferenceSettings  # noqa: E402
from backend.enhancement.pipeline import EnhancementPipeline  # noqa: E402
from backend.evaluation.reference_metrics import compute_psnr, compute_ssim  # noqa: E402
from backend.evaluation.types import MetricInput  # noqa: E402
from backend.inference.manager import ModelManager  # noqa: E402
from backend.inference.registry import available_models  # noqa: E402
from backend.runtime.config import PipelineRuntime  # noqa: E402
from training.benchmarks import benchmark_full_pipeline_gpu as gpu_bench  # noqa: E402

MODEL_ID = gpu_bench.MODEL_ID
MODES: tuple[str, ...] = gpu_bench.MODES
DEFAULT_WEIGHTS = gpu_bench.DEFAULT_WEIGHTS
DEFAULT_IMAGE_DIR = gpu_bench.DEFAULT_IMAGE_DIR
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training" / "reports" / "fp16_parity"
DEFAULT_DEVICE = "cuda"
DEFAULT_COUNT = 10

REFERENCE_PRECISION = "float32"
CANDIDATE_PRECISION = "float16"
DIFF_GAIN = 8

REPORT_STEM = "fp32_vs_fp16_parity"
PANELS_DIRNAME = "panels"


class ParityValidationError(gpu_bench.GpuBenchmarkError):
    """The comparison could not be produced faithfully."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class ValidationSettings:
    """Resolved validation inputs."""

    images: Path | None = None
    count: int | None = DEFAULT_COUNT
    modes: tuple[str, ...] = MODES
    weights: Path = DEFAULT_WEIGHTS
    output_dir: Path = DEFAULT_OUTPUT_DIR
    device: str = DEFAULT_DEVICE


def settings_from_args(args: argparse.Namespace) -> ValidationSettings:
    """Validate CLI values and fall back to the documented defaults."""
    weights = (args.weights or DEFAULT_WEIGHTS).expanduser()
    output_dir = (args.output_dir or DEFAULT_OUTPUT_DIR).expanduser()
    images = args.images.expanduser() if args.images is not None else None
    device = (args.device or DEFAULT_DEVICE).strip()
    modes = gpu_bench._parse_modes(args.modes)

    count = args.count
    if count is not None:
        count = int(count)
        if count <= 0:
            raise ParityValidationError("--count must be a positive integer.")
    if not device:
        raise ParityValidationError("--device must name a CUDA device.")

    return ValidationSettings(
        images=images,
        count=count,
        modes=modes,
        weights=weights,
        output_dir=output_dir,
        device=device,
    )


# ---------------------------------------------------------------------------
# image capture
# ---------------------------------------------------------------------------



def capture_rgb(image: Image.Image) -> np.ndarray:
    """Copy a PIL image into a float32 RGB array in [0, 1].

    ``PipelineResult.release()`` closes the PIL handles the pipeline produced,
    so every comparison buffer is materialised here — with an explicit
    ``tobytes()`` copy — before the caller is allowed to release anything.
    """
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    raw = np.frombuffer(rgb.tobytes(), dtype=np.uint8)
    expected = rgb.height * rgb.width * 3
    if raw.size != expected:
        raise ParityValidationError(
            f"Captured {raw.size} bytes for a {rgb.width}x{rgb.height} RGB image "
            f"(expected {expected})."
        )
    return raw.reshape(rgb.height, rgb.width, 3).astype(np.float32) / 255.0


def load_inputs(images: list[Path]) -> dict[str, np.ndarray]:
    """Load every input once, as an independent copy, for the panels."""
    captured: dict[str, np.ndarray] = {}
    for path in images:
        with Image.open(path) as handle:
            captured[path.name] = capture_rgb(handle)
    return captured


def input_digests(images: list[Path]) -> list[dict[str, Any]]:
    """Record SHA-256 digests so the report proves the inputs were identical."""
    entries: list[dict[str, Any]] = []
    for path in images:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"name": path.name, "path": str(path), "sha256": digest})
    return entries


# ---------------------------------------------------------------------------
# running both precisions
# ---------------------------------------------------------------------------



@dataclass
class PrecisionRun:
    """Everything one precision produced, with no live pipeline handles."""

    precision: str
    device: str
    outputs: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    cpu_stage_calls: int = 0
    statuses: dict[str, dict[str, str]] = field(default_factory=dict)


def assert_model_precision(audit: dict[str, Any], precision: str) -> None:
    """Fail when the loaded checkpoint is not in the requested precision."""
    expected = "torch.float16" if precision == "float16" else "torch.float32"
    if not audit.get("module_inspected"):
        raise ParityValidationError(
            "Could not inspect the model's parameters, so the "
            f"{precision} run cannot be verified as {expected}."
        )
    dtypes = set(audit.get("parameter_dtypes") or [])
    if dtypes != {expected}:
        raise ParityValidationError(
            f"The {precision} run loaded parameters as {sorted(dtypes) or ['none']}; "
            f"expected exactly {expected}. A precision mismatch would make the "
            "comparison meaningless."
        )


def collect_precision_outputs(
    runtime: PipelineRuntime,
    images: list[Path],
    modes: tuple[str, ...],
    weights: Path,
) -> PrecisionRun:
    """Run every mode over every image at ``runtime.precision``."""
    trap = gpu_bench.CpuFallbackTrap()
    run = PrecisionRun(precision=runtime.precision, device=runtime.device)
    manager = ModelManager(
        InferenceSettings(
            model_id=MODEL_ID,
            weights_path=weights,
            device=runtime.device,
            precision=runtime.precision,
        )
    )
    with trap:
        status = manager.load()
        if not status.ready:
            raise ParityValidationError(f"The checkpoint could not be loaded: {status.message}")
        run.audit = gpu_bench.audit_model_device(manager, runtime, runtime.precision)
        pipeline = EnhancementPipeline(model_manager=manager, runtime=runtime)
        gpu_bench.audit_pipeline_ops(pipeline, runtime)
        try:
            for mode in modes:
                per_image: dict[str, np.ndarray] = {}
                statuses: dict[str, str] = {}
                for path in images:
                    with Image.open(path) as handle:
                        request = handle.convert("RGB")
                    try:
                        result = pipeline.run(request, mode=mode)
                        try:
                            if result.status.value != "completed":
                                raise ParityValidationError(
                                    f"{mode} on {path.name} returned status "
                                    f"'{result.status.value}': {result.message}"
                                )
                            if result.final_image is None:
                                raise ParityValidationError(
                                    f"{mode} on {path.name} produced no final image."
                                )
                            per_image[path.name] = capture_rgb(result.final_image)
                            statuses[path.name] = result.status.value
                        finally:
                            result.release()
                    finally:
                        request.close()
                run.outputs[mode] = per_image
                run.statuses[mode] = statuses
        finally:
            manager.close()

    run.cpu_stage_calls = len(trap.calls)
    if trap.calls:  # pragma: no cover - the trap raises first
        raise ParityValidationError(
            f"CPU stage implementations were called: {sorted(set(trap.calls))}"
        )
    return run


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------



def compare_images(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Compare two float RGB images in [0, 1] on four pixel-level measures."""
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    candidate = np.ascontiguousarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ParityValidationError(
            f"Cannot compare images of different shapes: "
            f"{reference.shape} vs {candidate.shape}."
        )
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ParityValidationError(
            f"Expected an (H, W, 3) RGB image, got shape {reference.shape}."
        )

    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    metric_input = MetricInput(image=candidate, reference=reference)
    psnr = float(compute_psnr(metric_input))
    ssim = float(compute_ssim(metric_input))
    identical = float(difference.max()) == 0.0

    return {
        "mae": float(difference.mean()),
        "mae_8bit": float(difference.mean()) * 255.0,
        "max_abs": float(difference.max()),
        "max_abs_8bit": float(difference.max()) * 255.0,
        "psnr_db": None if not np.isfinite(psnr) else psnr,
        "ssim": None if not np.isfinite(ssim) else ssim,
        "identical": identical,
    }


def aggregate_metrics(per_image: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image measurements into one summary per mode."""
    if not per_image:
        raise ParityValidationError("Cannot aggregate an empty metric list.")

    def values(key: str) -> list[float]:
        return [
            float(item[key]) for item in per_image if item.get(key) is not None
        ]

    psnr = values("psnr_db")
    ssim = values("ssim")
    mae = values("mae")
    max_abs = values("max_abs")
    return {
        "images": len(per_image),
        "identical_images": sum(1 for item in per_image if item["identical"]),
        "mae_mean": float(np.mean(mae)),
        "mae_mean_8bit": float(np.mean(mae)) * 255.0,
        "mae_max": float(np.max(mae)),
        "max_abs": float(np.max(max_abs)),
        "max_abs_8bit": float(np.max(max_abs)) * 255.0,
        "psnr_mean_db": float(np.mean(psnr)) if psnr else None,
        "psnr_min_db": float(np.min(psnr)) if psnr else None,
        "ssim_mean": float(np.mean(ssim)) if ssim else None,
        "ssim_min": float(np.min(ssim)) if ssim else None,
    }


# ---------------------------------------------------------------------------
# comparison panels
# ---------------------------------------------------------------------------



def _label(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, len(text) * 7 + 8, 14], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return image


def difference_map(
    reference: np.ndarray, candidate: np.ndarray, gain: int = DIFF_GAIN
) -> np.ndarray:
    """Amplified absolute difference as a uint8 RGB image."""
    difference = np.abs(
        candidate.astype(np.float32) - reference.astype(np.float32)
    ).max(axis=2)
    amplified = np.clip(difference * float(gain) * 255.0, 0.0, 255.0)
    grey = amplified.astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def build_panel(
    original: np.ndarray | None,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    gain: int = DIFF_GAIN,
) -> Image.Image:
    """Build the input | float32 | float16 | |diff| x gain comparison strip."""
    tiles: list[Image.Image] = []
    if original is not None:
        tiles.append(_label(_to_pil(original), "input"))
    tiles.append(_label(_to_pil(reference), "float32"))
    tiles.append(_label(_to_pil(candidate), "float16"))
    tiles.append(_label(_to_pil(difference_map(reference, candidate, gain)), f"|diff| x{gain}"))

    height = max(tile.height for tile in tiles)
    width = sum(tile.width for tile in tiles)
    panel = Image.new("RGB", (width, height), (0, 0, 0))
    offset = 0
    for tile in tiles:
        panel.paste(tile, (offset, 0))
        offset += tile.width
    return panel


def _to_pil(array: np.ndarray) -> Image.Image:
    data = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(data)


def panel_path(panels_dir: Path, mode: str, name: str) -> Path:
    return panels_dir / mode / f"{Path(name).stem}__{mode}.png"


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------



def compare_runs(
    reference: PrecisionRun,
    candidate: PrecisionRun,
    inputs: dict[str, np.ndarray],
    panels_dir: Path,
    modes: tuple[str, ...],
) -> dict[str, Any]:
    """Produce the per-mode comparison section for both precision runs."""
    sections: dict[str, Any] = {}
    for mode in modes:
        if mode not in reference.outputs or mode not in candidate.outputs:
            raise ParityValidationError(f"Mode '{mode}' is missing from one of the runs.")
        entries: list[dict[str, Any]] = []
        for name, reference_image in reference.outputs[mode].items():
            candidate_image = candidate.outputs[mode].get(name)
            if candidate_image is None:
                raise ParityValidationError(
                    f"{mode} produced no float16 output for {name}."
                )
            metrics = compare_images(reference_image, candidate_image)
            destination = panel_path(panels_dir, mode, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            panel = build_panel(
                inputs.get(name), reference_image, candidate_image
            )
            try:
                panel.save(destination)
            finally:
                panel.close()
            entries.append(
                {
                    "image": name,
                    "shape": list(reference_image.shape),
                    "panel": str(destination),
                    **metrics,
                }
            )
        sections[mode] = {
            "mode": mode,
            "metrics": aggregate_metrics(entries),
            "images": entries,
        }
    return sections


def run_validation(settings: ValidationSettings) -> dict[str, Any]:
    """Run both precisions on CUDA and compare every mode."""
    registered = available_models()
    if MODEL_ID not in registered:
        raise ParityValidationError(
            f"The model registry has no '{MODEL_ID}' implementation "
            f"(registered: {', '.join(registered) or 'none'}). "
            "backend.deep_learning must be imported before the checkpoint is loaded."
        )

    images = gpu_bench.collect_images(settings.images, settings.count)
    weights = (settings.weights or DEFAULT_WEIGHTS).expanduser()
    if not weights.is_file():
        raise ParityValidationError(f"Checkpoint not found: {weights}")

    reference_runtime = gpu_bench.require_cuda(settings.device, REFERENCE_PRECISION)
    candidate_runtime = gpu_bench.require_cuda(settings.device, CANDIDATE_PRECISION)
    if reference_runtime.device != candidate_runtime.device:
        raise ParityValidationError(
            f"The two precisions resolved to different devices "
            f"({reference_runtime.device} vs {candidate_runtime.device}); "
            "the comparison must run on one GPU."
        )
    if candidate_runtime.precision != CANDIDATE_PRECISION:
        raise ParityValidationError(
            f"Requested {CANDIDATE_PRECISION} but resolved {candidate_runtime.precision}."
        )

    digests = input_digests(images)
    inputs = load_inputs(images)
    panels_dir = settings.output_dir / PANELS_DIRNAME

    reference_run = collect_precision_outputs(
        reference_runtime, images, settings.modes, weights
    )
    assert_model_precision(reference_run.audit, REFERENCE_PRECISION)
    candidate_run = collect_precision_outputs(
        candidate_runtime, images, settings.modes, weights
    )
    assert_model_precision(candidate_run.audit, CANDIDATE_PRECISION)

    modes = compare_runs(reference_run, candidate_run, inputs, panels_dir, settings.modes)

    return {
        "benchmark": "lumenoir.fp16_parity.gpu",
        "description": (
            "Numerical and visual agreement between the float32 and float16 "
            "full enhancement pipeline. Metrics compare FP16 output against "
            "FP32 output, not against a ground-truth reference."
        ),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": gpu_bench._git_commit(),
        "gpu": gpu_bench.gpu_facts(reference_runtime),
        "settings": {
            "device": reference_runtime.device,
            "reference_precision": REFERENCE_PRECISION,
            "candidate_precision": CANDIDATE_PRECISION,
            "weights": str(weights),
            "image_source": str(settings.images or DEFAULT_IMAGE_DIR),
            "count": len(images),
            "modes": list(settings.modes),
            "difference_gain": DIFF_GAIN,
            "duplicate_inputs_per_precision": True,
        },
        "checks": {
            "cuda_required": True,
            "cpu_fallback_allowed": False,
            "same_device": reference_runtime.device == candidate_runtime.device,
            "reference_model": reference_run.audit,
            "candidate_model": candidate_run.audit,
            "cpu_stage_calls": {
                REFERENCE_PRECISION: reference_run.cpu_stage_calls,
                CANDIDATE_PRECISION: candidate_run.cpu_stage_calls,
            },
            "outputs_captured_before_release": True,
        },
        "inputs": digests,
        "modes": modes,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------



def _format(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the JSON report as a Markdown document."""
    settings = report["settings"]
    gpu = report["gpu"]
    lines: list[str] = [
        "# Lumenoir FP32 vs FP16 full-pipeline parity",
        "",
        "Both precisions ran the complete enhancement pipeline on the same CUDA "
        "device with identical input images. The table values compare the FP16 "
        "output against the FP32 output; they are *not* enhancement-quality "
        "scores against ground truth.",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Commit: `{report.get('git_commit') or 'unknown'}`",
        f"- Device: `{settings['device']}` ({gpu['name']})",
        f"- Reference: `{settings['reference_precision']}` / "
        f"Candidate: `{settings['candidate_precision']}`",
        f"- Images: {settings['count']} from `{settings['image_source']}`",
        f"- Modes: {', '.join(settings['modes'])}",
        "- Panels: input | float32 | float16 | |difference| x"
        f"{settings['difference_gain']}",
        "",
        "## Agreement by mode",
        "",
        "| Mode | Images identical | MAE | MAE (8-bit) | Max abs | Max abs (8-bit) | "
        "PSNR mean dB | PSNR min dB | SSIM mean | SSIM min |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, section in report["modes"].items():
        metrics = section["metrics"]
        lines.append(
            f"| {mode} | {metrics['identical_images']}/{metrics['images']} | "
            f"{_format(metrics['mae_mean'])} | {_format(metrics['mae_mean_8bit'], 2)} | "
            f"{_format(metrics['max_abs'])} | {_format(metrics['max_abs_8bit'], 2)} | "
            f"{_format(metrics['psnr_mean_db'], 2)} | "
            f"{_format(metrics['psnr_min_db'], 2)} | "
            f"{_format(metrics['ssim_mean'], 6)} | {_format(metrics['ssim_min'], 6)} |"
        )
    lines.append("")

    for mode, section in report["modes"].items():
        lines.extend(
            [
                f"## {mode}",
                "",
                "| Image | MAE | Max abs | Max abs (8-bit) | PSNR dB | SSIM | Panel |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for entry in section["images"]:
            lines.append(
                f"| {entry['image']} | {_format(entry['mae'])} | "
                f"{_format(entry['max_abs'])} | {_format(entry['max_abs_8bit'], 2)} | "
                f"{_format(entry['psnr_db'], 2)} | {_format(entry['ssim'], 6)} | "
                f"`{entry['panel']}` |"
            )
        lines.append("")

    checks = report["checks"]
    lines.extend(
        [
            "## Execution checks",
            "",
            "| Check | Result |",
            "| --- | --- |",
            f"| CUDA required | {'yes' if checks['cuda_required'] else 'no'} |",
            f"| CPU fallback allowed | {'yes' if checks['cpu_fallback_allowed'] else 'no'} |",
            f"| Both precisions on one device | {'yes' if checks['same_device'] else 'no'} |",
            f"| float32 parameter dtypes | "
            f"{checks['reference_model']['parameter_dtypes']} |",
            f"| float16 parameter dtypes | "
            f"{checks['candidate_model']['parameter_dtypes']} |",
            f"| CPU stage implementation calls | {checks['cpu_stage_calls']} |",
            f"| Outputs copied before release | "
            f"{'yes' if checks['outputs_captured_before_release'] else 'no'} |",
            "",
            "## Method",
            "",
            "- Each precision loads its own checkpoint copy, builds its own "
            "pipeline and processes the same image files in the same order.",
            "- Final images are copied out of the pipeline result before "
            "`release()` closes the underlying buffers.",
            "- PSNR and SSIM use `backend.evaluation.reference_metrics` with "
            "`data_range = 1` on float32 RGB in [0, 1]; PSNR is `n/a` when the two "
            "outputs are bit-identical (infinite dB).",
            "- MAE and max abs are reported both in [0, 1] and in 8-bit levels "
            "(x255), so a difference of 1.0 means one 8-bit step.",
            "",
            "## Scope and limitations",
            "",
            "- Latency is not measured here; use "
            "`training.benchmarks.benchmark_full_pipeline_gpu` for timings.",
            "- Agreement metrics say nothing about enhancement quality versus a "
            "ground-truth reference; they only quantify the precision gap.",
            "- Final images are 8-bit, so differences below one level are "
            "invisible after quantisation even when the float outputs differ.",
            "- Results apply to this GPU, driver and PyTorch/CUDA build only.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    report: dict[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    """Write the JSON results and the Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{REPORT_STEM}.json"
    markdown_path = output_dir / f"{REPORT_STEM}.md"
    report.setdefault("artifacts", {})
    report["artifacts"].update(
        {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "panels": str(output_dir / PANELS_DIRNAME),
        }
    )
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Image directory, single image or JSON manifest of images.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=f"CUDA device: 'cuda' or 'cuda:N' (default {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Where to write the report and panels (default {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help=f"Maximum number of images (default {DEFAULT_COUNT}).",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated subset of classical,deep_learning,hybrid.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=f"Zero-DCE++ checkpoint (default {DEFAULT_WEIGHTS}).",
    )
    return parser


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"device {report['settings']['device']} ({report['gpu']['name']}), "
        f"{report['settings']['count']} images, "
        f"{report['settings']['reference_precision']} vs "
        f"{report['settings']['candidate_precision']}"
    )
    print(
        f"{'mode':>14} {'MAE':>9} {'MAE/8b':>8} {'max':>9} {'max/8b':>8} "
        f"{'PSNR':>8} {'SSIM':>9}"
    )
    for mode, section in report["modes"].items():
        metrics = section["metrics"]
        print(
            f"{mode:>14} {metrics['mae_mean']:>9.5f} {metrics['mae_mean_8bit']:>8.3f} "
            f"{metrics['max_abs']:>9.5f} {metrics['max_abs_8bit']:>8.3f} "
            f"{_format(metrics['psnr_mean_db'], 2):>8} "
            f"{_format(metrics['ssim_mean'], 6):>9}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = settings_from_args(args)
        report = run_validation(settings)
        json_path, markdown_path = write_outputs(report, settings.output_dir)
    except gpu_bench.CpuExecutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except gpu_bench.GpuBenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; no report written", file=sys.stderr)
        return 130

    _print_summary(report)
    print(f"\nJSON: {json_path}\nMarkdown: {markdown_path}")
    print(f"Panels: {settings.output_dir / PANELS_DIRNAME}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
