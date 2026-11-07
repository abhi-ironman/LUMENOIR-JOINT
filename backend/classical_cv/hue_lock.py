"""HueLock: guarded LAB cast correction plus optional source-chroma anchoring."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .arrays import as_float_rgb
from .colour import mid_tone_lab_stats
from .scan import LumenScanResult, scan


@dataclass(frozen=True)
class HueLockConfig:
    """HueLock guards and optional source-aware chroma anchoring.

    Source anchoring is disabled when source_anchor_strength is zero.
    """

    spread_threshold: float = 15.0
    max_shift: float = 8.0
    halve_on_spread: bool = True
    source_anchor_strength: float = 0.0
    analysis_max_edge: int = 320


@dataclass(frozen=True)
class HueLockResult:
    image: np.ndarray
    cast_a: float
    cast_b: float
    cast_magnitude: float
    beta_raw: float
    beta_applied: float
    shift_magnitude: float
    spread: float
    spread_guarded: bool
    source_anchor_mean_weight: float = 0.0
    source_anchor_coverage: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.image.shape),
            "cast": [self.cast_a, self.cast_b],
            "cast_magnitude": self.cast_magnitude,
            "beta_raw": self.beta_raw,
            "beta_applied": self.beta_applied,
            "shift_magnitude": self.shift_magnitude,
            "spread": self.spread,
            "spread_guarded": self.spread_guarded,
            "source_anchor_mean_weight": self.source_anchor_mean_weight,
            "source_anchor_coverage": self.source_anchor_coverage,
        }


def _analysis_view(rgb: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge < 32:
        raise ValueError("analysis_max_edge must be at least 32")
    height, width = rgb.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    if scale >= 1.0:
        return rgb
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _source_anchor_maps(
    reference_rgb: np.ndarray,
    *,
    config: HueLockConfig,
    noise_sigma: float,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_small = _analysis_view(reference_rgb, config.analysis_max_edge)
    source_lab = cv2.cvtColor(source_small, cv2.COLOR_RGB2LAB)
    source_a = source_lab[..., 1].astype(np.float32)
    source_b = source_lab[..., 2].astype(np.float32)
    if min(source_small.shape[:2]) >= 3:
        source_a = cv2.medianBlur(source_a, 3)
        source_b = cv2.medianBlur(source_b, 3)

    chroma = np.hypot(source_a, source_b)
    luma = (
        0.299 * source_small[..., 0]
        + 0.587 * source_small[..., 1]
        + 0.114 * source_small[..., 2]
    )
    luminance_confidence = np.clip((luma - 0.025) / 0.18, 0.0, 1.0)
    neutral_confidence = np.exp(-np.square(chroma / 18.0))
    noise_confidence = float(np.clip(1.0 - 12.0 * max(0.0, noise_sigma), 0.55, 1.0))
    strength = float(np.clip(config.source_anchor_strength, 0.0, 0.55))
    anchor_small = strength * luminance_confidence * neutral_confidence * noise_confidence

    height, width = output_shape
    if anchor_small.shape != output_shape:
        anchor = cv2.resize(anchor_small, (width, height), interpolation=cv2.INTER_LINEAR)
        source_a = cv2.resize(source_a, (width, height), interpolation=cv2.INTER_LINEAR)
        source_b = cv2.resize(source_b, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        anchor = anchor_small
    return (
        np.ascontiguousarray(anchor, dtype=np.float32),
        np.ascontiguousarray(source_a, dtype=np.float32),
        np.ascontiguousarray(source_b, dtype=np.float32),
    )


def correct(
    image: np.ndarray,
    *,
    scan_result: LumenScanResult | None = None,
    config: HueLockConfig | None = None,
    reference_rgb: np.ndarray | None = None,
) -> HueLockResult:
    """Correct neutral casts and optionally anchor reliable source chroma."""
    config = config or HueLockConfig()
    rgb = as_float_rgb(image)
    result = scan_result or scan(rgb, None)
    features = result.features
    parameters = result.parameters

    cast_a = float(features.colour_cast_a)
    cast_b = float(features.colour_cast_b)
    cast_magnitude = float(features.colour_cast_magnitude)
    beta_raw = float(parameters.hue_lock_beta)
    if not 0.0 <= config.source_anchor_strength <= 0.55:
        raise ValueError("source_anchor_strength must be in [0, 0.55]")

    analysis_rgb = _analysis_view(rgb, config.analysis_max_edge)
    stats = mid_tone_lab_stats(analysis_rgb, p50=float(features.p50))
    spread = max(stats.std_a, stats.std_b)
    spread_guarded = spread > config.spread_threshold

    beta = beta_raw / 2.0 if (spread_guarded and config.halve_on_spread) else beta_raw
    if cast_magnitude > 0.0:
        beta = min(beta, float(config.max_shift) / cast_magnitude)
    beta = max(0.0, beta)

    anchor_enabled = reference_rgb is not None and config.source_anchor_strength > 0.0
    if reference_rgb is not None:
        reference = as_float_rgb(reference_rgb, name="reference_rgb")
        if reference.shape != rgb.shape:
            raise ValueError(f"reference_rgb shape {reference.shape} must match image {rgb.shape}.")
    else:
        reference = None

    if beta <= 0.0 and not anchor_enabled:
        return HueLockResult(
            image=rgb.copy(),
            cast_a=cast_a,
            cast_b=cast_b,
            cast_magnitude=cast_magnitude,
            beta_raw=beta_raw,
            beta_applied=0.0,
            shift_magnitude=0.0,
            spread=spread,
            spread_guarded=spread_guarded,
        )

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    if beta > 0.0 and cast_magnitude > 0.0:
        lab[..., 1] -= np.float32(beta * cast_a)
        lab[..., 2] -= np.float32(beta * cast_b)

    anchor_mean = 0.0
    anchor_coverage = 0.0
    if anchor_enabled and reference is not None:
        anchor, source_a, source_b = _source_anchor_maps(
            reference,
            config=config,
            noise_sigma=float(features.noise_sigma),
            output_shape=rgb.shape[:2],
        )
        lab[..., 1] = (1.0 - anchor) * lab[..., 1] + anchor * source_a
        lab[..., 2] = (1.0 - anchor) * lab[..., 2] + anchor * source_b
        anchor_mean = float(anchor.mean())
        anchor_coverage = float(np.mean(anchor > 0.05))

    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    corrected = np.ascontiguousarray(np.clip(corrected, 0.0, 1.0).astype(np.float32))
    if corrected.shape != rgb.shape:
        raise RuntimeError(f"HueLock changed the image shape: {rgb.shape} -> {corrected.shape}.")

    return HueLockResult(
        image=corrected,
        cast_a=cast_a,
        cast_b=cast_b,
        cast_magnitude=cast_magnitude,
        beta_raw=beta_raw,
        beta_applied=beta,
        shift_magnitude=beta * cast_magnitude,
        spread=spread,
        spread_guarded=spread_guarded,
        source_anchor_mean_weight=anchor_mean,
        source_anchor_coverage=anchor_coverage,
    )


__all__ = ["HueLockConfig", "HueLockResult", "correct"]
