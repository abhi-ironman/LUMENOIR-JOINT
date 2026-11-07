"""Bounded, proxy-driven quality control for the enhancement pipeline.

The loop follows final_decision.md §5.7. It evaluates a completed result,
chooses the first matching corrective action, and allows at most two retries.
It deliberately uses inexpensive image proxies rather than NIQE or LPIPS.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image

from ..classical_cv.arrays import as_float_rgb, luminance
from ..classical_cv.noise import noise_sigma
from ..classical_cv.scan import LumenScanResult
from .types import PipelineResult


@dataclass(frozen=True)
class PhaseOverrides:
    """Per-pass adjustments; identity values preserve the existing pipeline."""

    gain_cap_scale: float = 1.0
    shoulder_start_delta: float = 0.0
    target_brightness: float | None = None
    denoise_strength_scale: float = 1.0
    hue_beta_scale: float = 1.0
    dl_strength_scale: float = 1.0

    def as_dict(self) -> dict[str, float | None]:
        return {
            "gain_cap_scale": self.gain_cap_scale,
            "shoulder_start_delta": self.shoulder_start_delta,
            "target_brightness": self.target_brightness,
            "denoise_strength_scale": self.denoise_strength_scale,
            "hue_beta_scale": self.hue_beta_scale,
            "dl_strength_scale": self.dl_strength_scale,
        }


@dataclass(frozen=True)
class PhaseEngineConfig:
    """Starting thresholds and conservative correction factors from §5.7."""

    max_extra_passes: int = 2
    max_new_clipped_fraction: float = 0.01
    median_luminance_min: float = 0.30
    median_luminance_max: float = 0.60
    max_noise_sigma: float = 0.025
    max_mean_hue_shift_degrees: float = 12.0
    default_target_brightness: float = 0.45
    brightness_raise_step: float = 0.05
    brightness_lower_step: float = 0.07
    clip_gain_scale: float = 0.80
    clip_shoulder_delta: float = -0.10
    noise_gain_scale: float = 0.85
    noise_denoise_scale: float = 1.50
    hue_strength_scale: float = 0.80
    noise_cap_binding_sigma: float = 0.02
    proxy_improvement_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if not 0 <= self.max_extra_passes <= 2:
            raise ValueError("max_extra_passes must be between 0 and 2.")
        if not 0.0 <= self.max_new_clipped_fraction <= 1.0:
            raise ValueError("max_new_clipped_fraction must be in [0, 1].")
        if not 0.0 <= self.median_luminance_min < self.median_luminance_max <= 1.0:
            raise ValueError("Median-luminance bounds must be ordered within [0, 1].")
        if self.max_noise_sigma < 0.0 or self.max_mean_hue_shift_degrees < 0.0:
            raise ValueError("Noise and hue thresholds must be non-negative.")


@dataclass(frozen=True)
class PhaseMetrics:
    """Cheap, full-resolution proxies measured against the source image."""

    new_clipped_fraction: float
    median_luminance: float
    noise_sigma: float
    mean_hue_shift_degrees: float
    proxy_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "new_clipped_fraction": self.new_clipped_fraction,
            "median_luminance": self.median_luminance,
            "noise_sigma": self.noise_sigma,
            "mean_hue_shift_degrees": self.mean_hue_shift_degrees,
            "proxy_score": self.proxy_score,
        }

    def within_thresholds(self, config: PhaseEngineConfig) -> bool:
        return (
            self.new_clipped_fraction <= config.max_new_clipped_fraction
            and config.median_luminance_min
            <= self.median_luminance
            <= config.median_luminance_max
            and self.noise_sigma <= config.max_noise_sigma
            and self.mean_hue_shift_degrees <= config.max_mean_hue_shift_degrees
        )


def _as_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return as_float_rgb(image)


def _mean_hue_shift(source: np.ndarray, output: np.ndarray) -> float:
    """Mean circular hue change, measured only on source pixels with C* > 10."""
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
    chroma = np.hypot(source_lab[..., 1], source_lab[..., 2])
    coloured = chroma > 10.0
    if not np.any(coloured):
        return 0.0

    source_hue = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)[..., 0]
    output_hue = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)[..., 0]
    delta = np.abs(output_hue - source_hue)
    circular_delta = np.minimum(delta, 360.0 - delta)
    return float(np.mean(circular_delta[coloured]))


def measure_phase_metrics(
    source: Image.Image | np.ndarray,
    output: Image.Image | np.ndarray,
) -> PhaseMetrics:
    """Measure new clipping, median luma, Immerkaer noise and hue shift."""
    source_rgb = _as_rgb(source)
    output_rgb = _as_rgb(output)
    if source_rgb.shape != output_rgb.shape:
        raise ValueError(
            f"Source shape {source_rgb.shape} must match output {output_rgb.shape}."
        )

    source_clip = float((source_rgb.max(axis=2) >= 0.98).mean())
    output_clip = float((output_rgb.max(axis=2) >= 0.98).mean())
    new_clip = max(0.0, output_clip - source_clip)
    output_luma = luminance(output_rgb)
    median = float(np.median(output_luma))
    sigma = noise_sigma(output_luma)
    hue_shift = _mean_hue_shift(source_rgb, output_rgb)
    score = 10.0 * new_clip + 2.0 * abs(median - 0.45) + 10.0 * sigma
    return PhaseMetrics(
        new_clipped_fraction=new_clip,
        median_luminance=median,
        noise_sigma=sigma,
        mean_hue_shift_degrees=hue_shift,
        proxy_score=float(score),
    )


def apply_phase_overrides(
    scan_result: LumenScanResult,
    overrides: PhaseOverrides,
) -> LumenScanResult:
    """Apply retry-only parameter changes without mutating the scan result."""
    parameters = scan_result.parameters
    gain_cap = max(1.0, parameters.gain_cap * overrides.gain_cap_scale)
    hue_beta = float(
        np.clip(parameters.hue_lock_beta * overrides.hue_beta_scale, 0.0, 0.5)
    )

    gamma = parameters.gamma
    if overrides.target_brightness is not None:
        target = float(np.clip(overrides.target_brightness, 0.25, 0.70))
        p50 = float(scan_result.features.p50)
        if p50 >= target:
            gamma = 1.0
        elif p50 <= 1e-6:
            gamma = 0.30
        else:
            gamma = float(np.clip(math.log(target) / math.log(p50), 0.30, 1.0))

    adjusted = replace(
        parameters,
        gain_cap=gain_cap,
        gamma=gamma,
        hue_lock_beta=hue_beta,
    )
    return replace(scan_result, parameters=adjusted)


def input_noise_cap_is_binding(
    source: Image.Image | np.ndarray,
    config: PhaseEngineConfig | None = None,
) -> bool:
    """The LumenScan gain cap reaches its lower bound when sigma >= 0.02."""
    config = config or PhaseEngineConfig()
    source_rgb = _as_rgb(source)
    sigma = noise_sigma(luminance(source_rgb))
    return sigma >= config.noise_cap_binding_sigma


class PhaseEngine:
    """Run one baseline and up to two corrective passes, retaining the lowest Q."""

    def __init__(self, config: PhaseEngineConfig | None = None) -> None:
        self.config = config or PhaseEngineConfig()

    def _next_overrides(
        self,
        metrics: PhaseMetrics,
        current: PhaseOverrides,
        *,
        noise_cap_binding: bool,
    ) -> tuple[str | None, PhaseOverrides]:
        """Apply the first matching §5.7 rule, in its documented priority order."""
        config = self.config

        if metrics.new_clipped_fraction > config.max_new_clipped_fraction:
            return (
                "reduce_highlight_gain",
                replace(
                    current,
                    gain_cap_scale=current.gain_cap_scale * config.clip_gain_scale,
                    shoulder_start_delta=(
                        current.shoulder_start_delta + config.clip_shoulder_delta
                    ),
                ),
            )

        if (
            not noise_cap_binding
            and (
                metrics.median_luminance < config.median_luminance_min
                or metrics.median_luminance > config.median_luminance_max
            )
        ):
            target = (
                config.default_target_brightness
                if current.target_brightness is None
                else current.target_brightness
            )
            step = (
                config.brightness_raise_step
                if metrics.median_luminance < config.median_luminance_min
                else -config.brightness_lower_step
            )
            return (
                "adjust_target_brightness",
                replace(current, target_brightness=target + step),
            )

        if metrics.noise_sigma > config.max_noise_sigma:
            return (
                "increase_denoising_and_reduce_gain",
                replace(
                    current,
                    gain_cap_scale=current.gain_cap_scale * config.noise_gain_scale,
                    denoise_strength_scale=(
                        current.denoise_strength_scale * config.noise_denoise_scale
                    ),
                ),
            )

        if metrics.mean_hue_shift_degrees > config.max_mean_hue_shift_degrees:
            return (
                "reduce_hue_and_dl_strength",
                replace(
                    current,
                    hue_beta_scale=current.hue_beta_scale * config.hue_strength_scale,
                    dl_strength_scale=current.dl_strength_scale * config.hue_strength_scale,
                ),
            )

        return None, current

    def run(
        self,
        *,
        original: Image.Image | np.ndarray,
        run_pass: Callable[[PhaseOverrides], PipelineResult],
    ) -> PipelineResult:
        """Execute the bounded loop and return the result with the lowest proxy Q."""
        source = _as_rgb(original)
        source_luma = luminance(source)
        source_p50 = float(np.median(source_luma))
        source_dynamic_range = float(
            np.percentile(source_luma, 95.0) - np.percentile(source_luma, 5.0)
        )
        source_highlights = float((source.max(axis=2) >= 0.98).mean())
        bypass_gate = (
            source_p50 >= 0.45
            and source_dynamic_range >= 0.45
            and source_highlights < 0.01
        )
        noise_cap_binding = input_noise_cap_is_binding(source, self.config)

        started = time.perf_counter()
        attempts: list[dict[str, object]] = []
        best_result: PipelineResult | None = None
        best_score = float("inf")
        selected_pass = 0
        previous_score: float | None = None
        total_pipeline_ms = 0.0
        overrides = PhaseOverrides()
        stop_reason = "max_extra_passes"

        for pass_index in range(self.config.max_extra_passes + 1):
            try:
                candidate = run_pass(overrides)
            except Exception as exc:
                if best_result is None:
                    raise
                attempts.append(
                    {
                        "pass": pass_index,
                        "overrides": overrides.as_dict(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                stop_reason = "retry_failed"
                break

            total_pipeline_ms += float(candidate.total_ms)
            if not candidate.completed or candidate.final_image is None:
                attempts.append(
                    {
                        "pass": pass_index,
                        "overrides": overrides.as_dict(),
                        "status": candidate.status.value,
                        "message": candidate.message,
                    }
                )
                if best_result is None:
                    return candidate
                candidate.release()
                stop_reason = "retry_incomplete"
                break

            output = _as_rgb(candidate.final_image)
            metrics = measure_phase_metrics(source, output)
            improved_since_previous = (
                previous_score is None
                or metrics.proxy_score
                < previous_score - self.config.proxy_improvement_tolerance
            )

            if metrics.proxy_score < best_score - self.config.proxy_improvement_tolerance:
                if best_result is not None:
                    best_result.release()
                best_result = candidate
                best_score = metrics.proxy_score
                selected_pass = pass_index
            else:
                candidate.release()

            action, next_overrides = self._next_overrides(
                metrics,
                overrides,
                noise_cap_binding=noise_cap_binding,
            )
            attempts.append(
                {
                    "pass": pass_index,
                    "overrides": overrides.as_dict(),
                    "metrics": metrics.as_dict(),
                    "next_action": action,
                }
            )

            if bypass_gate:
                stop_reason = "do_no_harm_bypass"
                break
            if metrics.within_thresholds(self.config):
                stop_reason = "proxy_thresholds_met"
                break
            if pass_index >= self.config.max_extra_passes:
                stop_reason = "max_extra_passes"
                break
            if not improved_since_previous:
                stop_reason = "proxy_score_did_not_improve"
                break
            if action is None:
                stop_reason = "no_matching_correction"
                break

            previous_score = metrics.proxy_score
            overrides = next_overrides

        if best_result is None:
            raise RuntimeError("Phase Engine completed without a usable pipeline result.")

        diagnostics = dict(best_result.diagnostics)
        diagnostics["phase_engine"] = {
            "attempt_count": len(attempts),
            "extra_passes": max(0, len(attempts) - 1),
            "selected_pass": selected_pass,
            "selected_proxy_score": best_score,
            "stop_reason": stop_reason,
            "noise_cap_binding": noise_cap_binding,
            "attempts": attempts,
        }
        wall_ms = (time.perf_counter() - started) * 1000.0
        return replace(
            best_result,
            total_ms=max(total_pipeline_ms, wall_ms),
            diagnostics=diagnostics,
        )


__all__ = [
    "PhaseEngine",
    "PhaseEngineConfig",
    "PhaseMetrics",
    "PhaseOverrides",
    "apply_phase_overrides",
    "input_noise_cap_is_binding",
    "measure_phase_metrics",
]
